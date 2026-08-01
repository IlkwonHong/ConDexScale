"""GT 표본의 물리 평형을 눈으로 확인하는 시각화.

레코드 몇 개를 골라 MuJoCo 씬(실물 충돌 형상 + 구형 팁 + LP 평형 예압)을 만들고
4단계 영상으로 렌더링한다:
  ① 정착(팁 접근·쥐기) → ② 무외란 평형 유지 → ③ 최약 방향 외란 = 합격 시험
  수준(주장 α × 0.25) 버팀 → ④ 같은 방향 3×α 과부하 → 놓침 (한계 실증)

실행:
  ./venv/bin/python -m contraction_gt.visualize_equilibrium            # mp4 6개 렌더
  ./venv/bin/python -m contraction_gt.visualize_equilibrium --xml-only # 뷰어용 XML만
인터랙티브 (사용자 직접, GUI 필요):
  ./venv/bin/python -m mujoco.viewer --mjcf=md_claude/gt_viz/<이름>.xml
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mujoco
import imageio

from .metric import direction_set
from .pilot_sim import _equilibrium_preload, ASSET_ROOT
from .sim_eval import SimConfig, build_scene_xml

ROOT = Path("data/contraction_gt_300k")
OUT = Path("md_claude/gt_viz")
FPS = 30


def pick_samples():
    """합격 레코드에서 N=2/3/4 × 2개, 타입 다양하게."""
    dfs = [pd.read_parquet(f) for f in sorted(Path("data/contraction_gt_v1_100k").glob("*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    ok = df[df.gate_verdict.isin(["pass", "exempt"]) & (df.min_alpha > 0.05)]
    picks = []
    for n in (2, 3, 4):
        sub = ok[ok.n_anchors == n].sort_values("min_alpha", ascending=False)
        sub = sub.drop_duplicates("type")
        picks.append(sub.iloc[[0, len(sub) // 2]])   # 최상 + 중간
    return pd.concat(picks)


def build_record_scene(row, mmap):
    ds, oid = mmap[row.grasp_id.split("#")[0]]
    n = int(row.n_anchors)
    pos = np.array([[row[f"anchor{i}_{a}"] for a in "xyz"] for i in range(n)])
    nrm = np.array([[row[f"normal{i}_{a}"] for a in "xyz"] for i in range(n)])
    r_obj = float(row.r_obj)
    cache = np.load(Path("data/contraction_gt_pilot/object_cache") / ds / f"{oid}.npz")
    com = cache["com"] * float(row.total_scale)
    cfg = SimConfig(mu_tor=float(row.mu_tor),
                    tip_radius=max(float(row.pad_rel) * r_obj, 0.002),
                    timestep=1e-3, settle_steps=600, hold_steps=300)
    coacd = ASSET_ROOT / ds / "processed_data" / oid / "urdf" / "meshes"
    xml = build_scene_xml(("coacd", str(coacd), float(row.total_scale)), pos, nrm, cfg)
    preload, _x, _W = _equilibrium_preload(
        (pos - com) / r_obj, nrm, 0.6,
        float(row.mu_tor) / r_obj, cfg.f_cmd)
    alpha = np.array([row[f"alpha_{j}"] for j in range(14)])
    pos_n = torch.as_tensor((pos - com) / r_obj, dtype=torch.float64).unsqueeze(0)
    dirs = direction_set(pos_n)[0].numpy()
    wd = int(alpha.argmin())
    return xml, preload, cfg, r_obj, dirs[wd], float(alpha[wd]), n


def render_record(row, mmap, out_path, xml_only=False):
    xml, preload, cfg, r_obj, weak_dir, a_min, n = build_record_scene(row, mmap)
    if xml_only:
        Path(out_path).with_suffix(".xml").write_text(xml)
        return
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

    # 색: 팁 = 빨강, 물체 = 하늘색
    for gi in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        model.geom_rgba[gi] = [0.9, 0.2, 0.2, 1.0] if name.startswith("tip_g") \
            else [0.55, 0.75, 0.95, 1.0]

    # 예압 보정 정착 (sim_eval 과 같은 원리, 간이판)
    target = np.clip(preload, 0.02 * cfg.f_cmd, cfg.f_cmd)
    ctrl = cfg.start_gap + target / cfg.kp
    for _ in range(4):
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = ctrl
        for _ in range(cfg.settle_steps):
            mujoco.mj_step(model, data)
        ctrl = ctrl + 0.8 * (target - data.actuator_force) / cfg.kp
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = ctrl

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0]
    cam.distance = 4.5 * r_obj
    cam.azimuth, cam.elevation = 130, -20

    renderer = mujoco.Renderer(model, 480, 640)
    frames = []
    # 단계: (지속초, 외란 배율)
    phases = [(1.5, 0.0), (1.5, 0.0), (2.0, 0.25 * a_min), (2.0, 3.0 * a_min)]
    for dur, level in phases:
        data.xfrc_applied[obj_body, :3] = level * cfg.f_cmd * weak_dir[:3]
        data.xfrc_applied[obj_body, 3:] = level * cfg.f_cmd * r_obj * weak_dir[3:]
        steps = int(dur / cfg.timestep)
        for s in range(steps):
            mujoco.mj_step(model, data)
            if s % int(1 / (FPS * cfg.timestep)) == 0:
                cam.azimuth += 0.15          # 천천히 돌려서 입체감
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
    imageio.mimsave(out_path, frames, fps=FPS, quality=8)
    drift = float(np.linalg.norm(data.qpos[:3])) / r_obj
    print(f"{out_path.name}: N={n}, min-α={a_min:.3f}, 최종 이탈 {drift:.2f}·r_obj "
          f"(과부하 단계 포함이라 큰 값이 정상)", flush=True)


def main(xml_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [l.split("\t") for l in (ROOT / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    mmap = {r[3]: (r[1], r[2]) for r in rows}
    picks = pick_samples()
    for _, row in picks.iterrows():
        tag = f"N{int(row.n_anchors)}_{row.type[:18]}_a{row.min_alpha:.2f}"
        try:
            render_record(row, mmap, OUT / f"equil_{tag}.mp4", xml_only=xml_only)
        except Exception as e:
            print(f"실패 {tag}: {e!r}", flush=True)


if __name__ == "__main__":
    main(xml_only="--xml-only" in sys.argv)
