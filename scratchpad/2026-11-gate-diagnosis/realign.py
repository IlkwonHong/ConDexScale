"""팁이 미는 방향을 실제 접촉 법선으로 다시 맞춘 뒤 재시험.

1단계: 원래 장면(위치 서보, 게이트 설정)에서 정착 → 팁별 실제 접촉 위치·법선(힘 가중 평균).
2단계: 앵커 = 실제 접촉 위치, 법선 = 실제 법선으로 장면을 새로 만들고,
       예압·외란별 배분도 새 기하로 다시 계산해 hybrid_check 의 조건 O / C 로 시험.
외란 방향·크기는 원래 게이트 값 그대로.
"""
import pickle
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).parent))
from hybrid_check import run as hy_run  # noqa: E402
from during_load_check import merge_contacts  # noqa: E402

ROOT = Path("data/contraction_gt_300k")
TMP = Path(sys.argv[1])


def realign(job):
    from contraction_gt.sim_eval import SimConfig, measure_capacity_multi
    from contraction_gt.pilot_sim import _equilibrium_preload
    cfg = SimConfig(mu_tor=job["mu_tor"], tip_radius=job["tip_radius"], timestep=1e-3,
                    settle_steps=600, hold_steps=300)
    log = {}
    measure_capacity_multi(("coacd", job["coacd"], job["scale"]), job["pos"], job["nrm"],
                           np.zeros((0, 6)), np.zeros((0, 1)), job["r_obj"], cfg,
                           preload_forces=job["preload"], seed=job["idx"], contact_log=log)
    n = len(job["pos"])
    pos = job["pos"].copy(); nrm = job["nrm"] / np.linalg.norm(job["nrm"], axis=1, keepdims=True)
    moved = np.zeros(n, bool)
    for t in range(n):
        cs = [c for c in log.get("contacts", []) if c["tip"] == t]
        if not cs:
            continue
        P, N, _ = merge_contacts(cs, n)          # 팁 하나만 들어오므로 1행
        pos[t], nrm[t] = P[0], N[0]; moved[t] = True
    ang = np.degrees(np.arccos(np.clip((nrm * (job["nrm"] / np.linalg.norm(job["nrm"], axis=1, keepdims=True))).sum(1), -1, 1)))
    j2 = dict(job)
    j2["pos"], j2["nrm"] = pos, nrm
    j2["pos_n"] = (pos - job["com"]) / job["r_obj"]
    j2["preload"] = _equilibrium_preload(j2["pos_n"], nrm, 0.6, job["mu_tor"] / job["r_obj"], 5.0)[0]
    return j2, dict(moved=moved.mean(), ang=ang[moved])


def work(job):
    j2, info = realign(job)
    return job["idx"], info, hy_run((j2, "O")), hy_run((j2, "C"))


def main():
    from contraction_gt.gate_runner import _load_records, _prepare
    fail8 = [29093, 62729, 102750, 129568, 272962, 342257, 485907, 518538]
    hy = pickle.load(open(TMP / "hybrid_raw.pkl", "rb"))
    fixed8 = sorted({r["idx"] for r in hy if r["cond"] == "C" and all(r["held"])}
                    - {r["idx"] for r in hy if r["cond"] == "O" and all(r["held"])})
    pass10 = sorted({r["idx"] for r in hy if r["cond"] == "O" and all(r.get("held", [False]))})
    m = _load_records(ROOT)
    ids = fail8 + fixed8 + pass10
    jobs = _prepare(np.array(ids), m, ROOT)
    man = [l.split("\t") for l in (ROOT / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    mmap = {r[3]: (r[1], r[2]) for r in man}
    for j in jobs:
        i = j["idx"]
        ds, oid = mmap[str(m["grasp_id"][i]).split("#")[0]]
        j["com"] = np.load(Path("data/contraction_gt_pilot/object_cache") / ds / f"{oid}.npz")["com"] * float(m["total_scale"][i])
        j["pos_n"] = (j["pos"] - j["com"]) / j["r_obj"]
    with get_context("spawn").Pool(8) as pool:
        out = pool.map(work, jobs, chunksize=1)
    pickle.dump(out, open(TMP / "realign_raw.pkl", "wb"))
    R = {o[0]: o for o in out}
    print("(앵커·법선을 쥔 직후 실제 접촉으로 바꾼 뒤 재시험)")
    for name, g in (("하이브리드로도 못 고친 8", fail8), ("하이브리드로 고쳐진 8", fixed8), ("원래 합격 10", pass10)):
        def rate(k):
            rs = [R[i][k] for i in g if R[i][k]["settled"]]
            return (sum(all(r["held"]) for r in rs), len(rs), np.mean([h for r in rs for h in r["held"]]) if rs else np.nan)
        o, c = rate(2), rate(3)
        ang = np.concatenate([R[i][1]["ang"] for i in g])
        print(f"{name:22s} | 법선 바뀐 각도 중앙값 {np.median(ang):.1f}° (최대 {ang.max():.1f}°) | "
              f"위치 서보 {o[0]}/{o[1]} (방향 {o[2]:.2f}) | 하이브리드 C {c[0]}/{c[1]} (방향 {c[2]:.2f})")


if __name__ == "__main__":
    main()
