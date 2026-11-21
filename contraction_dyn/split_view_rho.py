"""분할 화면 영상 (ρ판) — 왼쪽: 시연 GT (MANO 양손 + GT 경로의 박스, 물리 없음),
오른쪽: play_contact_rho 재생 영상. 시연 프레임 번호로 시간 정렬 (전환 중에는 직전 프레임 유지).

scratchpad/split_view.py 를 ρ판 산출물(replay_rho 계획 + contact_rho 영상)에 맞게 옮긴 것.

실행: ./venv/bin/python -m contraction_dyn.split_view_rho box_grab_01 r0.15 [r0.05 ...] [--plan-tag __nomove]
출력: md_claude/dexmachina/split_rho_{seq}_{tag}.mp4
"""
import argparse
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import trimesh                                        # noqa: E402
from matplotlib import font_manager                   # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.spatial.transform import Rotation          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contraction_dyn.hand_contacts import SeqData     # noqa: E402

MDIR = ROOT / "extlib/dexmachina/dexmachina/assets/arctic/box"
try:
    _fp = subprocess.check_output(["fc-match", "-f", "%{file}", "NanumGothic"], text=True).strip()
    font_manager.fontManager.addfont(_fp)
    _fam = font_manager.FontProperties(fname=_fp).get_name()
except Exception:  # noqa: BLE001
    _fam, _fp = "DejaVu Sans", None
plt.rcParams["font.family"] = [_fam, "DejaVu Sans"]


def label_frame(img, text, fp):
    """오른쪽 패널 위에 글자 (PIL)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(fp, 22) if fp else ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    dr.rectangle([8, 8, 8 + 12 * len(text) + 16, 44], fill=(0, 0, 0))
    dr.text((16, 12), text, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def run(seq, tag, plan_tag="", subj="s01"):
    z = np.load(ROOT / f"data/contraction_dyn/replay_rho/{subj}__{seq}{plan_tag}.npz", allow_pickle=True)
    demo_f, plat = z["demo_frame"], z["platform"]
    vid = ROOT / f"md_claude/dexmachina/contact_rho_{seq}_{tag}.mp4"
    rd = imageio.get_reader(vid)
    n_vid = rd.count_frames()

    sd = SeqData(subj, seq)
    mb = trimesh.load(MDIR / "bottom_watertight_tiny_box.obj", force="mesh")
    mt = trimesh.load(MDIR / "top_watertight_tiny_box.obj", force="mesh")
    Vb, Fb = np.array(mb.vertices), np.array(mb.faces)
    Vt, Ft = np.array(mt.vertices), np.array(mt.faces)

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

    fig = plt.figure(figsize=(8, 6.08), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    cx, cy, cz = float(plat[0]), float(plat[1]), float(plat[2])
    ax.set_xlim(cx - 0.4, cx + 0.4)
    ax.set_ylim(cy - 0.4, cy + 0.4)
    ax.set_zlim(cz - 0.15, cz + 0.55)
    ax.set_box_aspect((1, 1, 0.875))
    ax.view_init(elev=17, azim=-90)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.text2D(0.03, 0.94, "시연 GT (사람 양손 + 물체 경로, 물리 없음)", transform=ax.transAxes, fontsize=13)
    ax.computed_zorder = False
    quad = np.array([[cx - .45, cy - .45, cz], [cx + .45, cy - .45, cz],
                     [cx + .45, cy + .45, cz], [cx - .45, cy + .45, cz]])
    ax.add_collection3d(Poly3DCollection([quad], facecolor="#e8e8e8", edgecolor="#bbb", zorder=1))
    cols = {}
    for name, color, zo in [("bot", "#d2601a", 2), ("top", "#e8862e", 3),
                            ("hr", "#c9a189", 4), ("hl", "#c9a189", 4)]:
        pc = Poly3DCollection([], facecolor=color, edgecolor="none", alpha=1.0, zorder=zo)
        ax.add_collection3d(pc)
        cols[name] = pc
    ftxt = ax.text2D(0.03, 0.04, "", transform=ax.transAxes, fontsize=11)

    out = ROOT / f"md_claude/dexmachina/split_rho_{seq}_{tag}.mp4"
    wr = imageio.get_writer(out, fps=15)
    right_label = f"재생 결과 (ρ 앵커 + 새 배분기, {tag})"
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
        ftxt.set_text(f"시연 프레임 {f}" + ("  (전환 중: 시연 정지)" if demo_f[s] < 0 else ""))
        fig.canvas.draw()
        gt = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        rp = label_frame(rd.get_data(i), right_label, _fp)
        h = min(gt.shape[0], rp.shape[0])
        wr.append_data(np.hstack([gt[:h], rp[:h]]))
        if i % 100 == 0:
            print(f"  {i}/{len(steps)}", flush=True)
    wr.close()
    print(f"저장: {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("tags", nargs="*", default=["r0.15"])
    ap.add_argument("--plan-tag", default="")
    ap.add_argument("--subj", default="s01")
    a = ap.parse_args()
    for tg in a.tags:
        run(a.seq, tg, a.plan_tag, a.subj)
