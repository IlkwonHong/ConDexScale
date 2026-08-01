"""최종 패키징: 탐색 part 파일 + 도장 결과 → taxonomy 타입별 parquet (D11).

수록 기준: min-α(μ=0.6) > 0.02 (분포·도장 바닥값과 일치. α 벡터가 저장되므로
학습 시 더 엄격한 재필터는 언제든 가능 — D12).
도장 열(`gate_verdict`): N=2 는 'exempt'(파일럿 검증으로 면제), N≥3 대표는
pass/fail/fail_settle/unstable, 대표가 아닌 레코드는 'not_tested'.

실행: ./venv/bin/python -m contraction_gt.package_parquet <manifest_dir> <out_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FLOOR = 0.02
PARAMS = dict(
    version="v1-100k", mu_t_nominal=0.6, mu_t_labels=[0.3, 0.6, 1.0],
    rho_rel=0.15, pad_radius_rel_n2=[0.05, 0.10, 0.20], pad_radius_rel_multi=[0.10],
    hertz_frac=0.5, gate_gamma=0.25, min_alpha_floor=FLOOR,
    calib_ratio_median=dict(force=0.75, torque=0.88, axis=0.81),
    dir_order="0-5 힘 ±x±y±z, 6-11 토크 ±x±y±z (r_obj 정규화), 12-13 앵커 급소축 ±",
    normals="물체 안쪽(접촉힘 방향)", coords="물체 원점 파지 좌표계(m), scale=total_scale",
)


def load_all(manifest_dir):
    merged = {}
    for f in sorted(Path(manifest_dir, "search_records").glob("shard_*part*.npz")):
        d = np.load(f, allow_pickle=False)
        for k in d.files:
            if k in ("times", "n_grasp", "n_fail"):
                continue
            merged.setdefault(k, []).append(d[k])
    return {k: np.concatenate(v) for k, v in merged.items()}


def main(manifest_dir, out_dir):
    m = load_all(manifest_dir)
    R = len(m["min_alpha"])
    keep = m["min_alpha"] > FLOOR
    print(f"레코드 {R:,}개 중 수록(min-α>{FLOOR}) {keep.sum():,}개")

    g = np.load(Path(manifest_dir) / "gate_records.npz", allow_pickle=False)
    gate = {(str(a), int(b)): str(v) for a, b, v in
            zip(g["grasp_id"], g["n_anchors"], g["verdict"])}
    # 대표(도장 대상)는 (grasp_id, N) 키 — 같은 키의 다른 레코드는 not_tested
    rep_min = {(str(a), int(b)): float(c) for a, b, c in
               zip(g["grasp_id"], g["n_anchors"], g["min_alpha"])}

    cols = {}
    idx = np.nonzero(keep)[0]
    cols["grasp_id"] = m["grasp_id"][idx].astype(str)
    cols["type"] = m["type"][idx].astype(str)
    for k in ("n_anchors", "pad_rel", "min_alpha", "weak_dir", "mu_tor",
              "thumbless", "n_units", "r_obj", "total_scale"):
        cols[k] = m[k][idx]
    cols["partition"] = m["partition"][idx].astype(str)
    cols["stage"] = m["stage"][idx].astype(str)
    for j in range(14):
        cols[f"alpha_{j}"] = m["alpha"][idx, j]
        cols[f"alpha_mu03_{j}"] = m["alpha_mu03"][idx, j]
        cols[f"alpha_mu10_{j}"] = m["alpha_mu10"][idx, j]
    for a in range(4):
        for c, ax in enumerate("xyz"):
            cols[f"anchor{a}_{ax}"] = m["anchor_pos"][idx, a, c]
            cols[f"normal{a}_{ax}"] = m["anchor_normal"][idx, a, c]
        cols[f"prior_dist{a}"] = m["prior_dist"][idx, a]

    verdicts = []
    for i in idx:
        key = (str(m["grasp_id"][i]), int(m["n_anchors"][i]))
        if m["n_anchors"][i] == 2:
            verdicts.append("exempt")
        elif key in gate and abs(rep_min[key] - m["min_alpha"][i]) < 1e-12:
            verdicts.append(gate[key])
        else:
            verdicts.append("not_tested")
    cols["gate_verdict"] = verdicts

    df = pd.DataFrame(cols)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for typ, sub in df.groupby("type"):
        sub.to_parquet(out / f"{typ}.parquet", index=False)
    (out / "PARAMS.json").write_text(json.dumps(PARAMS, ensure_ascii=False, indent=1))
    print(f"저장: {out} — 타입 파일 {df['type'].nunique()}개, 총 {len(df):,}행")
    print("gate_verdict 분포:")
    print(df["gate_verdict"].value_counts().to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
