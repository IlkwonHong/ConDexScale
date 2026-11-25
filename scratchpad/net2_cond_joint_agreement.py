"""읽기 전용 집계: 조건부 net2 와 기준 net2 의 '정답과의 일치'를 같은 검증 파지 240개(묶음 A, 팔 3대)에서 비교.
저장된 평가 결과만 읽는다 (새 추론·선형계획·시뮬레이션 없음)."""
import pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/robin/a_workspace/condex")
THR = (0.078, 0.15)

def dist(pts, idx, gt, r):
    return np.linalg.norm(pts[np.asarray(idx)] - gt, axis=-1) / r

def load_base():
    out = {}
    for m in pickle.load(open(ROOT / "scratchpad/net2_viz_cases.pkl", "rb")):
        pts, r = m["pts"].astype(np.float64), float(m["r_obj"])
        for p in m["parts"]:
            if p["n_g"] != 3: continue
            gt = p["gt_anchor"].astype(np.float64)
            top1 = np.array([t[0] for t in p["top8"]])
            d1 = dist(pts, top1, gt, r)                                  # [3]
            reach = np.array([np.linalg.norm(pts[t] - gt[g], axis=1).min() / r for g, t in enumerate(p["top8"])])
            fin = dist(pts, p["combo"], gt, r)
            out[(m["gid"], int(p["cid"]))] = dict(d1=d1, reach=reach, fin=fin)
    return out

def load_cond(path):
    out = {}
    for m in pickle.load(open(ROOT / path, "rb")):
        pts, r = m["pts"].astype(np.float64), float(m["r_obj"])
        best_of_part = {}
        for c in m["cands"]:                      # 파티션마다 다듬은 후보 중 α 가 가장 높은 것
            if c["part"] not in best_of_part or c["min_alpha"] > best_of_part[c["part"]]["min_alpha"]:
                best_of_part[c["part"]] = c
        for pi, p in enumerate(m["parts"]):
            if p["n_g"] != 3: continue
            gt = p["gt_anchor"].astype(np.float64)
            d1 = dist(pts, p["greedy"], gt, r)                           # 앞 자리를 보고 차례로 1등
            dn = np.linalg.norm(pts[p["net_combos"]] - gt[None], axis=-1) / r   # [64,3]
            dl = np.linalg.norm(pts[p["lp_combos"]] - gt[None], axis=-1) / r
            first1 = int(np.argmax(p["lp0"]))                            # 첫 팔은 조건이 없음
            fin = dist(pts, best_of_part[pi]["combo"], gt, r) if pi in best_of_part else np.full(3, np.inf)
            out[(m["gid"], int(p["cid"]))] = dict(d1=d1, dn=dn, dl=dl, fin=fin, n_lp=len(dl))
    return out

def pct(x): return f"{100 * float(np.mean(x)):.1f}%"

def report(name, D, keys):
    d1 = np.array([D[k]["d1"] for k in keys]); fin = np.array([D[k]["fin"] for k in keys])
    print(f"\n[{name}] 파티션 {len(keys)}개")
    for thr in THR:
        ok = d1 <= thr
        marg = ok.mean(0)
        print(f"  물체 반경의 {thr}배 이내")
        print(f"    1등 점, 팔 하나(전체 평균) {pct(ok)} | 자리별 {', '.join(pct(ok[:, j]) for j in range(3))}")
        print(f"    1등 점, 세 팔 모두 {pct(ok.all(1))} | 세 팔이 서로 무관하다면 나올 값(자리별 비율의 곱) {100 * marg.prod():.1f}%")
        a, b, c = ok[:, 0], ok[:, 1], ok[:, 2]
        print(f"    첫째가 맞았을 때 둘째도 맞은 비율 {pct(b[a]) if a.any() else '-'} (둘째 전체 {pct(b)}) | "
              f"첫째·둘째가 맞았을 때 셋째도 맞은 비율 {pct(c[a & b]) if (a & b).any() else '-'} (셋째 전체 {pct(c)}, 해당 {int((a & b).sum())}개)")
        if "reach" in D[keys[0]]:
            rc = np.array([D[k]["reach"] for k in keys]) <= thr
            print(f"    팔마다 상위 8점 안에 정답 근처 점이 있음(세 팔 모두) {pct(rc.all(1))}")
        else:
            print(f"    네트워크가 낸 조합 64개 안에 세 팔 모두 맞은 조합이 있음 {pct([ (D[k]['dn'] <= thr).all(1).any() for k in keys])} | "
                  f"셋째 팔을 범위 안 전부로 넓힌 조합 안에 있음 {pct([ (D[k]['dl'] <= thr).all(1).any() for k in keys])}")
        print(f"    α 로 고르고 주변 7점으로 다듬은 최종 조합, 세 팔 모두 {pct((fin <= thr).all(1))}")

base = load_base()
kb = sorted(base)
for tag, path in (("조건부 net2 v1 (stage2_cond_v1)", "scratchpad/net2_cond_eval_v1.pkl"),
                  ("조건부 net2 v2 (stage2_cond_v2_support)", "scratchpad/net2_cond_eval_v2_support.pkl")):
    cond = load_cond(path)
    kc = sorted(cond)
    both = sorted(set(kb) & set(kc))
    print(f"\n===== {tag}: 파티션 {len(kc)}개, 기준 모델 파티션 {len(kb)}개, 양쪽에 다 있는 파티션 {len(both)}개 =====")
    report("기준 net2 (팔마다 따로) — 양쪽에 다 있는 파티션만", base, both)
    report(f"{tag} — 양쪽에 다 있는 파티션만", cond, both)
    print(f"  조건부 모델이 선형계획으로 채점한 조합 수 중앙값 {int(np.median([cond[k]['n_lp'] for k in both]))}")
