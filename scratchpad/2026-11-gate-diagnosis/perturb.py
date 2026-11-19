"""앵커를 표면을 따라 흔들었을 때 알파값이 얼마나 무너지는가 (새 시뮬 없음).

표면 = 시뮬과 같은 볼록 조각 메쉬. 앵커마다 접평면 안 무작위 방향으로 ε·팁반지름 이동 →
메쉬 최근접점으로 투영, 그 면 법선(안쪽)을 새 법선으로. ε=0 도 같은 투영으로 계산해 기준으로 씀.
지표: 게이트 방향 세트에서 '방향별 알파 / 시험 강도'의 최솟값 (1 미만 = 수식상으로도 불합격).
"""
import pickle
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
ROOT = Path("data/contraction_gt_300k")
TMP = Path(sys.argv[1])
EPS = [0.0, 0.05, 0.10, 0.20]
K = 32


def work(a):
    import torch
    import trimesh
    from contraction_gt.pilot_sim import ASSET_ROOT
    from contraction_gt.metric import ray_wrench_matrix, alpha_reference
    idx, d, pos, nrm, sc, tr, com, r_obj, mu_tor, dirs, lv = a
    base = ASSET_ROOT / d[0] / "processed_data" / d[1]
    mesh = trimesh.util.concatenate([trimesh.load(p, force="mesh") for p in sorted((base / "urdf/meshes").glob("*.obj"))])
    mesh.apply_scale(sc)
    rng = np.random.default_rng(idx)
    nu = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    h = np.where(np.abs(nu[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])
    t1 = np.cross(h, nu); t1 /= np.linalg.norm(t1, axis=1, keepdims=True); t2 = np.cross(nu, t1)
    out = {}
    for eps in EPS:
        ks = 1 if eps == 0 else K
        P = []; N = []
        for _ in range(ks):
            th = rng.uniform(0, 2 * np.pi, len(pos))
            q = pos + eps * tr * (np.cos(th)[:, None] * t1 + np.sin(th)[:, None] * t2)
            cp, _, fi = trimesh.proximity.closest_point(mesh, q)
            P.append(cp); N.append(-mesh.face_normals[fi])
        P = torch.as_tensor(np.stack(P)); N = torch.as_tensor(np.stack(N))
        C = torch.as_tensor(com).expand(ks, 3)
        W = ray_wrench_matrix(P, N, C, r_obj, 0.6, mu_tor)
        D = torch.as_tensor(dirs).unsqueeze(0).expand(ks, -1, -1)
        A = alpha_reference(W, D).numpy()                          # [ks, Ddir]
        out[eps] = (A / lv[None]).min(axis=1)                      # 표본별 최약 방향 여유
    return idx, out


def main():
    from contraction_gt.gate_runner import _load_records, _prepare
    m = _load_records(ROOT)
    R = [r for r in pickle.load(open(ROOT / "wrench_similarity/realized_contacts.pkl", "rb"))
         if r["verdict"] in ("pass", "fail") and r["settled"]]
    V = {r["idx"]: r["verdict"] for r in R}
    jobs = {j["idx"]: j for j in _prepare(np.array(list(V)), m, ROOT)}
    man = [l.split("\t") for l in (ROOT / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    mmap = {r[3]: (r[1], r[2]) for r in man}
    args = []
    for i, j in jobs.items():
        d = mmap[str(m["grasp_id"][i]).split("#")[0]]
        com = np.load(Path("data/contraction_gt_pilot/object_cache") / d[0] / f"{d[1]}.npz")["com"] * float(m["total_scale"][i])
        args.append((i, d, j["pos"], j["nrm"], float(m["total_scale"][i]), j["tip_radius"], com, j["r_obj"],
                     j["mu_tor"], j["dirs"], j["levels"][:, 0]))
    with get_context("spawn").Pool(8) as pool:
        O = dict(pool.map(work, args, chunksize=4))
    pickle.dump(O, open(TMP / "perturb_raw.pkl", "wb"))
    hy = pickle.load(open(TMP / "hybrid_raw.pkl", "rb"))
    fail8 = [29093, 62729, 102750, 129568, 272962, 342257, 485907, 518538]
    fixed8 = sorted({r["idx"] for r in hy if r["cond"] == "C" and all(r["held"])}
                    - {r["idx"] for r in hy if r["cond"] == "O" and all(r["held"])})
    pass10 = sorted({r["idx"] for r in hy if r["cond"] == "O" and all(r.get("held", [False]))})
    from sklearn.metrics import roc_auc_score
    print("값 = 최약 방향 '알파 / 시험 강도' (1 미만이면 수식상 불합격). 흔든 경우는 32표본 중 최악값.")
    groups = (("하이브리드로도 못 고친 8", fail8), ("하이브리드로 고쳐진 8", fixed8), ("원래 합격 10", pass10),
              ("합격 전체", [i for i in O if V[i] == "pass"]), ("불합격 전체", [i for i in O if V[i] == "fail"]))
    for name, g in groups:
        cells = []
        for e in EPS:
            w = np.array([O[i][e].min() for i in g])
            cells.append(f"ε={e:.2f}: {np.median(w):.2f} (1 미만 {np.mean(w < 1):.2f})")
        print(f"{name:22s} n={len(g):3d} | " + " | ".join(cells))
    y = np.array([V[i] == "pass" for i in O])
    for e in EPS:
        s = np.array([O[i][e].min() for i in O])
        print(f"합격/불합격 AUC ε={e:.2f}: {roc_auc_score(y, s):.3f}")


if __name__ == "__main__":
    main()
