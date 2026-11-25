"""분할 화면 영상 — 왼쪽: GT 재생(MANO 양손 + GT 경로 박스, 물리 없음),
오른쪽: 사다리 1단 Genesis 재생 영상.

시간 정렬: 재생 npz 의 demo_frame(스텝→데모 프레임, 전환 중엔 직전 프레임 유지)으로
GT 패널을 재생 영상의 각 프레임(4스텝 간격)에 맞춘다.
실행: ./venv/bin/python scratchpad/split_view.py box_use_02 [box_grab_01]
"""
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contraction_dyn.hand_contacts import SeqData

MDIR = ROOT / "extlib/dexmachina/dexmachina/assets/arctic/box"


def run(seq, subj="s01", kind="rigid"):
    z = np.load(ROOT / f"data/contraction_dyn/replay/{subj}__{seq}.npz",
                allow_pickle=True)
    demo_f, plat = z["demo_frame"], z["platform"]
    rd = imageio.get_reader(ROOT / f"md_claude/dexmachina/{kind}_replay_{seq}.mp4")
    n_vid = rd.count_frames()

    sd = SeqData(subj, seq)
    mb = trimesh.load(MDIR / "bottom_watertight_tiny_box.obj", force="mesh")
    mt = trimesh.load(MDIR / "top_watertight_tiny_box.obj", force="mesh")
    Vb, Fb = np.array(mb.vertices), np.array(mb.faces)
    Vt, Ft = np.array(mt.vertices), np.array(mt.faces)

    # 스텝→데모 프레임 (전환 -1 은 직전 값 유지)
    df = demo_f.copy()
    for i in range(1, len(df)):
        if df[i] < 0:
            df[i] = df[i - 1]
    steps = np.arange(0, len(df), 4)[:n_vid]
    frames_needed = sorted(set(int(df[s]) for s in steps))
    fmap = {f: k for k, f in enumerate(frames_needed)}
    hv = {s: sd.hand_verts_world(s, frames_needed) for s in ("right", "left")}
    hf = np.array(sd.layers["right"].th_faces)
    hf_l = np.array(sd.layers["left"].th_faces)

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    cx, cy, cz = float(plat[0]), float(plat[1]), float(plat[2])
    ax.set_xlim(cx - 0.4, cx + 0.4)
    ax.set_ylim(cy - 0.4, cy + 0.4)
    ax.set_zlim(cz - 0.15, cz + 0.55)
    ax.set_box_aspect((1, 1, 0.875))
    ax.view_init(elev=17, azim=-90)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.text2D(0.03, 0.94, "GT (human demo, kinematic)", transform=ax.transAxes,
              fontsize=13)
    # 받침 상판
    px, py, pz = cx, cy, cz
    ax.computed_zorder = False           # 자동 깊이 정렬이 테이블을 앞에 그리는 문제
    quad = np.array([[px - .45, py - .45, pz], [px + .45, py - .45, pz],
                     [px + .45, py + .45, pz], [px - .45, py + .45, pz]])
    ax.add_collection3d(Poly3DCollection([quad], facecolor="#e8e8e8",
                                         edgecolor="#bbb", zorder=1))
    cols = {}
    for name, color, zo in [("bot", "#d2601a", 2), ("top", "#e8862e", 3),
                            ("hr", "#c9a189", 4), ("hl", "#c9a189", 4)]:
        pc = Poly3DCollection([], facecolor=color, edgecolor="none",
                              alpha=1.0, zorder=zo)
        ax.add_collection3d(pc)
        cols[name] = pc

    suffix = "" if kind == "rigid" else f"_{kind}"
    wr = imageio.get_writer(
        ROOT / f"md_claude/dexmachina/split{suffix}_{seq}.mp4", fps=15)
    for i, s in enumerate(steps):
        f = int(df[s])
        R, t, th, _v, _c = sd.obj_frame(f)
        vb = (R @ Vb.T).T + t
        vt_ = Vt @ Rotation.from_rotvec([0, 0, -th]).as_matrix().T if abs(th) > 1e-8 else Vt
        vt_ = (R @ vt_.T).T + t
        k = fmap[f]
        cols["bot"].set_verts(vb[Fb])
        cols["top"].set_verts(vt_[Ft])
        cols["hr"].set_verts(hv["right"][k][hf])
        cols["hl"].set_verts(hv["left"][k][hf_l])
        fig.canvas.draw()
        gt = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        rp = rd.get_data(i)
        h = min(gt.shape[0], rp.shape[0])
        wr.append_data(np.hstack([gt[:h], rp[:h]]))
        if i % 100 == 0:
            print(f"  {i}/{len(steps)}", flush=True)
    wr.close()
    print(f"저장: md_claude/dexmachina/split{suffix}_{seq}.mp4")


if __name__ == "__main__":
    kind = "contact" if "--contact" in sys.argv else "rigid"
    seqs = [a for a in sys.argv[1:] if not a.startswith("--")] or ["box_use_02"]
    for seq in seqs:
        run(seq, kind=kind)
