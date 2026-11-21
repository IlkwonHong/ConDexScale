"""ρ판 라운드 평가 — 최소 분해 세트 7회 (설계 §5) 실행 및 표 작성.

  ① 전체 새 파이프라인 (r_req 0.15)      ② ① − 가속 하중 전방지시
  ③ ① 에서 첫 배치 끝까지 유지(재파지 0)  ④ r_req = 0.05 / 0.10 / 0.20
  ⑤ 눌러앉기 규칙 없음(6팔 튜닝 배치가 있는 구간마다 교체, 이동 6번) × r_req 0.15 / 0.05 (2026-09-05 추가)

실행: ./venv/bin/python -m contraction_dyn.eval_rho_round box_grab_01            # 7회 실행 + 표
      ./venv/bin/python -m contraction_dyn.eval_rho_round box_grab_01 --table-only  # 기존 결과로 표만
출력: data/contraction_dyn/eval_rho/round_{seq}.json, 표준출력 표 (마크다운)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIGS = [
    ("① 전체 새 파이프라인 (r_req 0.15)", ["--r-req", "0.15"], "r0.15"),
    ("② ① − 가속 하중 전방지시", ["--r-req", "0.15", "--no-ff"], "r0.15_noff"),
    ("③ ① 에서 첫 배치 끝까지 유지", ["--r-req", "0.15", "--plan-tag", "__nomove"], "r0.15_nomove"),
    ("④ r_req 0.05", ["--r-req", "0.05"], "r0.05"),
    ("④ r_req 0.10", ["--r-req", "0.10"], "r0.10"),
    ("④ r_req 0.20", ["--r-req", "0.20"], "r0.20"),
    ("⑤ 눌러앉기 없음(키프레임마다 교체), r_req 0.15", ["--r-req", "0.15", "--plan-tag", "__always"], "r0.15_always"),
    ("⑤ 눌러앉기 없음(키프레임마다 교체), r_req 0.05", ["--r-req", "0.05", "--plan-tag", "__always"], "r0.05_always"),
]
BASELINE = dict(name="기준: 옛 α 시간표 + 옛 제어기 (CEM 상수), 2026-09-05 재실행",
                fell_step=None, pos_mean_cm=11.1, pos_max_cm=31.8, rot_mean_deg=8.0, rot_max_deg=58.2)


def load_summary(seq, tag, subj="s01"):
    p = ROOT / "data/contraction_dyn/eval_rho" / f"{subj}__{seq}__{tag}.npz"
    z = np.load(p, allow_pickle=True)
    s = dict(z["summary"][0])
    ok = z["demo_frame"] >= 0
    fn = np.linalg.norm(z["f_meas"], axis=2)[ok]
    s["tip_force_median_N"] = float(np.nanmedian(fn))
    s["r_meas_median"] = float(np.nanmedian(z["r_meas"][ok]))
    return s


def table(rows):
    hdr = ("| 재생 | 낙하 | 위치 오차 평균 / 최대 (cm) | 회전 오차 평균 / 최대 (°) | "
           "실측 팁 힘 중앙값 (N) | 요구 여유 미달(2단계) 비율 |")
    lines = [hdr, "|---|---|---|---|---|---|"]
    lines.append(f"| {BASELINE['name']} | 없음 | {BASELINE['pos_mean_cm']:.1f} / {BASELINE['pos_max_cm']:.1f} | "
                 f"{BASELINE['rot_mean_deg']:.1f} / {BASELINE['rot_max_deg']:.1f} | (옛 배분기) | – |")
    for name, s in rows:
        fell = "없음" if s["fell_step"] is None else f"step {s['fell_step']} 낙하"
        lines.append(f"| {name} | {fell} | {s['pos_mean_cm']:.1f} / {s['pos_max_cm']:.1f} | "
                     f"{s['rot_mean_deg']:.1f} / {s['rot_max_deg']:.1f} | {s['tip_force_median_N']:.1f} | "
                     f"{s['stage_frac'][2]:.2f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", nargs="?", default="box_grab_01")
    ap.add_argument("--subj", default="s01")
    ap.add_argument("--table-only", action="store_true")
    a = ap.parse_args()
    if not a.table_only:
        from contraction_dyn.play_contact_rho import main as play
        for name, argv, tag in CONFIGS:
            print(f"\n##### {name}")
            play([a.seq, "--subj", a.subj, "--tag", tag] + argv)
    rows = [(name, load_summary(a.seq, tag, a.subj)) for name, _argv, tag in CONFIGS]
    md = table(rows)
    print("\n" + md)
    out = ROOT / "data/contraction_dyn/eval_rho" / f"round_{a.seq}.json"
    json.dump(dict(seq=a.seq, baseline=BASELINE,
                   rows=[dict(name=n, **{k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()})
                                         for k, v in s.items()}) for n, s in rows],
                   table_md=md), open(out, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
