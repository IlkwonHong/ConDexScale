"""물성 변주 세트 — 7케이스(명목, 질량 ±10%, 마찰 0.5/0.7, 질량중심 y ±1.5cm) × 요구 여유 {0.05, 0.15}.

제어기·배분기는 명목 물성(질량 1.094kg, 배분 마찰 0.5, URDF 질량중심)을 그대로 믿고, 시뮬만 바꾼다.
시간표는 키프레임마다 교체(__always) 판.

실행: ./venv/bin/python -m contraction_dyn.eval_rho_dr box_grab_01 [--r-req 0.05 0.15] [--table-only] [--jobs 2]
출력: data/contraction_dyn/eval_rho/dr_{seq}.json, 표(마크다운)
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = [
    ("명목", {}),
    ("질량 0.9", {"--dr-mass": "0.9"}),
    ("질량 1.1", {"--dr-mass": "1.1"}),
    ("마찰 0.5", {"--dr-fric": "0.5"}),
    ("마찰 0.7", {"--dr-fric": "0.7"}),
    ("질량중심 +1.5cm", {"--dr-com": "0.015"}),
    ("질량중심 −1.5cm", {"--dr-com": "-0.015"}),
]
PLAN_TAG = "__always"


def tag_of(r_req, case_i):
    return f"r{r_req:.2f}_always_dr{case_i}"


def run_one(seq, subj, r_req, case_i, extra):
    tag = tag_of(r_req, case_i)
    cmd = [str(ROOT / "venv/bin/python"), "-m", "contraction_dyn.play_contact_rho", seq,
           "--subj", subj, "--plan-tag", PLAN_TAG, "--r-req", str(r_req), "--tag", tag]
    for k, v in extra.items():
        cmd += [k, v]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    ok = (ROOT / "data/contraction_dyn/eval_rho" / f"{subj}__{seq}__{tag}.npz").exists()
    print(f"  완료: {tag} ({'OK' if ok else '실패'})", flush=True)
    if not ok:
        print(res.stdout[-2000:], res.stderr[-2000:])
    return tag


def load(seq, subj, tag):
    z = np.load(ROOT / "data/contraction_dyn/eval_rho" / f"{subj}__{seq}__{tag}.npz", allow_pickle=True)
    s = dict(z["summary"][0])
    ok = z["demo_frame"] >= 0
    s["r_meas_min_ok"] = float(np.nanmin(z["r_meas"][ok]))
    s["r_meas_median"] = float(np.nanmedian(z["r_meas"][ok]))
    s["tip_force_median_N"] = float(np.nanmedian(np.linalg.norm(z["f_meas"], axis=2)[ok]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", nargs="?", default="box_grab_01")
    ap.add_argument("--subj", default="s01")
    ap.add_argument("--r-req", type=float, nargs="*", default=[0.05, 0.15])
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    jobs = [(r, i, extra) for r in a.r_req for i, (_n, extra) in enumerate(CASES)]
    if not a.table_only:
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(lambda j: run_one(a.seq, a.subj, j[0], j[1], j[2]), jobs))
    rows = []
    hdr = "| 케이스 | " + " | ".join(f"요구 여유 {r:.2f}: 낙하 / 위치 평균·최대 (cm) / 회전 평균·최대 (°) / 실측 힘 여유 최소" for r in a.r_req) + " |"
    lines = [hdr, "|---|" + "---|" * len(a.r_req)]
    for i, (name, _e) in enumerate(CASES):
        cells = []
        for r in a.r_req:
            s = load(a.seq, a.subj, tag_of(r, i))
            row = {k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}) for k, v in s.items()}
            row["case"] = name
            rows.append(row)
            fell = "없음" if s["fell_step"] is None else f"**낙하(step {s['fell_step']})**"
            cells.append(f"{fell} / {s['pos_mean_cm']:.1f}·{s['pos_max_cm']:.1f} / "
                         f"{s['rot_mean_deg']:.1f}·{s['rot_max_deg']:.1f} / {s['r_meas_min_ok']:+.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    print("\n" + md)
    out = ROOT / "data/contraction_dyn/eval_rho" / f"dr_{a.seq}.json"
    json.dump(dict(seq=a.seq, plan_tag=PLAN_TAG, rows=rows, table_md=md), open(out, "w"),
              ensure_ascii=False, indent=1, default=float)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
