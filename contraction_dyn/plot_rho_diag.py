"""ρ판 재생 진단 곡선 — 오차 · ρ(t)(재배분 허용 여유) · r(t)(확보한 배분 여유) · 실측 힘 여유 · 배분 단계.

실행: ./venv/bin/python -m contraction_dyn.plot_rho_diag box_grab_01 r0.15 [r0.15_nomove ...]
출력: md_claude/keyframes/rho_diag_{seq}_{tag}.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# 한글 글꼴: fontconfig 가 아는 나눔고딕을 matplotlib 에 직접 등록 (matplotlib 캐시에 없음)
import subprocess  # noqa: E402
from matplotlib import font_manager  # noqa: E402
try:
    _fp = subprocess.check_output(["fc-match", "-f", "%{file}", "NanumGothic"], text=True).strip()
    font_manager.fontManager.addfont(_fp)
    _fam = font_manager.FontProperties(fname=_fp).get_name()
except Exception:  # noqa: BLE001
    _fam = "Noto Sans CJK JP"
plt.rcParams["font.family"] = [_fam, "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot(seq, tag, subj="s01"):
    z = np.load(ROOT / "data/contraction_dyn/eval_rho" / f"{subj}__{seq}__{tag}.npz", allow_pickle=True)
    s = z["summary"][0]
    df, pe, re_ = z["demo_frame"], z["pos_err"], np.degrees(z["rot_err"])
    rho, ru, rmax, rm, st = z["rho_t"], z["r_used"], z["r_max"], z["r_meas"], z["stage"]
    T = len(df)
    t = np.arange(T) / 60.0
    kf = json.load(open(ROOT / "data/contraction_dyn/keyframes" / f"{subj}__{seq}.json"))
    fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    swap = df < 0
    for a in ax:
        a.fill_between(t, 0, 1, where=swap, transform=a.get_xaxis_transform(), color="0.85", label=None)
        for kfr in kf["keyframes"]:
            j = np.flatnonzero(df == kfr)
            if len(j):
                a.axvline(t[j[0]], color="0.6", lw=0.6, ls=":")
    ax[0].plot(t, 100 * pe, color="C0", label="위치 오차 (cm)")
    ax[0].set_ylabel("위치 오차 (cm)")
    ax0b = ax[0].twinx()
    ax0b.plot(t, re_, color="C3", alpha=0.7, label="회전 오차 (°)")
    ax0b.set_ylabel("회전 오차 (°)")
    ax[0].set_title(f"{seq} / {tag}: 낙하 {'없음' if s['fell_step'] is None else 'step ' + str(s['fell_step'])}, "
                    f"위치 평균 {s['pos_mean_cm']:.1f}cm, 회전 평균 {s['rot_mean_deg']:.1f}°  (회색 = 전환 사건, 점선 = 키프레임)")
    ax[1].plot(t, rho, color="C2", label="ρ(t): 재배분을 허용한 여유 (계획, 명목 렌치)")
    ax[1].plot(t, ru, color="C1", label="r(t): 배분기가 확보한 여유 (요구 여유에서 멈춤)")
    ax[1].plot(t, rmax, color="C1", ls="--", alpha=0.6, label="r_max(t): 요구 여유 미달 시 최대 가능 여유")
    ax[1].axhline(s["r_req"], color="k", lw=0.6, ls="--")
    ax[1].set_ylabel("여유 (ρ 단위)")
    ax[1].set_ylim(-0.05, 0.7)
    ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(t, rm, color="C4", lw=0.7, label="실측 팁 힘의 배분 여유 (음수 = 각뿔 밖)")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_ylim(-1.0, 0.4)
    ax[2].set_ylabel("실측 힘 여유")
    ax[2].legend(loc="upper right", fontsize=8)
    ax[3].step(t, st, color="C5", where="post")
    ax[3].set_yticks([0, 1, 2, 3])
    ax[3].set_yticklabels(["접촉<2", "1단계(요구 여유 달성)", "2단계(최대 r)", "3단계(잔차 폴백)"], fontsize=8)
    ax[3].set_ylabel("배분 단계")
    ax[3].set_xlabel("시간 (s, 60Hz 스텝)")
    # 데모 프레임 눈금
    for f in range(0, int(df.max()) + 1, 100):
        j = np.flatnonzero(df == f)
        if len(j):
            ax[3].annotate(f"f{f}", (t[j[0]], -0.4), fontsize=7, ha="center", annotation_clip=False)
    fig.tight_layout()
    out = ROOT / "md_claude/keyframes" / f"rho_diag_{seq}_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    print("저장:", out)
    return out


if __name__ == "__main__":
    seq = sys.argv[1]
    for tag in sys.argv[2:] or ["r0.15"]:
        plot(seq, tag)
