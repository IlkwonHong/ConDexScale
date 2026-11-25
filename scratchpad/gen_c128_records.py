"""후보 128개(인코더 조각 중심) 정답 레코드 생성 (EXP-004, 2026-11-14).

파지마다 팔 3대 레코드에 대해, EXP-003 의 규칙 나(지금 탐색의 범위 반경 + 물체 반경의 0.05배 안, 법선 조건을
만족하고 팁 간섭이 없는 조각 중심)로 후보를 정하고 조합을 전부 선형계획으로 채점한다. 무작위 요소 없음.

출력 (data/contraction_gt_300k/ 아래, 기존 search_records 와 같은 열 + c128_* 열):
  search_records_c128a/     학습 파지 + 모델 선택용 검증 파지. 팔 3대 레코드의 anchor_pos = 새 정답(A)
  search_records_c128b/     같은 파지. 팔 3대 레코드의 anchor_pos = 저장된 정답을 팔마다 가장 가까운 후보로 옮긴 것(B)
  search_records_c128_eval/ 최종 평가용 검증 파지 480개. anchor_pos = A, B 의 번호는 c128_b_idx 에 함께 담는다
  팔 2대·4대 레코드와 min_alpha 등 나머지 열은 원본 그대로 (net1 목표를 바꾸지 않기 위함).
추가 열 (팔 3대 레코드에만 값이 있고 나머지는 -1 / NaN / False):
  c128_ok [행]           후보 조합이 있고 최선 α > 1e-6 (거짓이면 히트맵 과제에서 뺀다)
  c128_cand [행,4,128]   팔별 후보의 표면점 번호 (-1 패딩) — 학습·평가가 이 후보를 그대로 읽는다
  c128_a_idx / c128_b_idx / c128_near_idx / c128_prox_idx [행,4]
                         새 정답 / 옮긴 저장 정답 / 접촉 중심에서 가장 가까운 후보 / 값싼 점수 1등 조합
  c128_alpha / c128_snap_alpha / c128_near_alpha / c128_prox_alpha [행]  위 조합들의 min α (놓을 수 없으면 NaN)
  c128_n_combo [행]      채점한 조합 수

검산 (불통과면 멈춘다):
  1. 학습 파지의 물체와 검증·평가 파지의 물체가 겹치지 않는다 (물체 ID 해시 분리 — Stage1Dataset 과 같은 규칙).
  2. 평가 파지 480개의 새 정답 α 가 EXP-003 에 저장된 규칙 나 결과와 0.000001 이내 (99% 이상).
  3. 평가 파지의 저장된 정답 α 를 다시 계산하면 저장값과 같다 (99% 이상).
  4. 조각 캐시의 중심 좌표 = 그 번호의 표면점 (모든 물체).

이어 하기 (2026-11-14 추가): --resume 은 이미 쓴 묶음의 파지를 건너뛰고 나머지만 만든다. 평가용 폴더가 있으면
  평가용 생성과 검산 2·3 은 건너뛴다 (앞선 실행에서 통과한 것). 학습 파지는 고정된 무작위 순서(씨앗 0)로 처리한다 —
  gid 순서로 처리하면 앞쪽에 조합이 많은 물체가 몰려 있어 초반 속도로 잰 예상 시간이 실제보다 크게 나온다.
  --spent-min 은 앞선 실행에서 이미 쓴 시간(분). 중단 조건은 (1) 이번 실행에서 학습 파지 1,000개를 처리한 시점의
  예상 전체 시간(이미 쓴 시간 포함)이 6시간을 넘으면 멈춤, (2) 실제 누적 시간이 6시간에 닿으면 그때까지 만든 것만 쓰고 끝냄.

실행: ./venv/bin/python scratchpad/gen_c128_records.py [--smoke] [--n-train 20000] [--n-select 500] [--resume] [--spent-min 분]
계획: harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-004-c128-learnability/plan.md
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratchpad"))

import gt_seed_consistency as G                                   # noqa: E402
import gt_coarse_candidate_precheck as P                          # noqa: E402  (rule_sets, arm_geometry)
import contraction_gt.runner_batched as RB                        # noqa: E402
from contraction_gt.search import (SearchConfig, mu_tor_from_pad, _top_combos,   # noqa: E402
                                   _feasible_candidates)
from contraction_gt.object_cache import cache_path                # noqa: E402

DATA = ROOT / "data/contraction_gt_300k"
BASE = DATA / "search_records"
RULE = "나"
N, KMAX = 3, 128
ALPHA_MIN = 1e-6
BASE_COLS = ["grasp_id", "n_anchors", "partition", "pad_rel", "min_alpha", "r_obj", "total_scale",
             "thumbless", "anchor_pos"]
CHUNK, FLUSH = 12, 2000
VERIFY_MIN = 0.99
TIME_CAP_H = 6.0                    # 계획서의 중단 조건: 추정 전체 시간이 6시간을 넘으면 멈춘다


def is_val(oid, seed=0, val_fraction=0.05):
    """Stage1Dataset 의 물체 단위 분리와 같은 규칙."""
    h = int(hashlib.md5(f"{seed}:{oid}".encode()).hexdigest()[:8], 16)
    return (h % 10_000) < val_fraction * 10_000


def spaced(pool, k):
    if len(pool) <= k:
        return list(pool)
    idx = np.unique(np.round(np.linspace(0, len(pool) - 1, k)).astype(int))
    return [pool[i] for i in idx]


def load_base():
    cols = {k: [] for k in BASE_COLS}
    for p in sorted(BASE.glob("shard_*.npz")):
        with np.load(p, allow_pickle=True) as z:
            for k in cols:
                cols[k].append(z[k])
    return {k: np.concatenate(v) for k, v in cols.items()}


def prepare(gid, ds, oid, cfg, centers_of, bad_obj):
    rel, ridx = gid.rsplit("#", 1)
    patches, ts = G.load_grasp_contacts(str(ROOT / G.CONTACT_ROOT), rel, int(ridx))
    samples = G.load_samples(ROOT / G.CACHE_ROOT, ds, oid, scale=ts)
    if (ds, oid) not in centers_of:
        z = np.load(P.PATCH_ROOT / ds / f"{oid}.npz")
        cidx = z["center_idx"].astype(np.int64)
        pts_unit = np.load(cache_path(ROOT / G.CACHE_ROOT, ds, oid))["points"]
        if np.abs(z["centers"] - pts_unit[cidx]).max() >= 1e-5:
            bad_obj.append((ds, oid))
        centers_of[(ds, oid)] = cidx
    pad_rel = cfg.pad_radius_rel_multi[0]
    tip_r = max(pad_rel * samples.r_obj, 0.002)
    mu_tor = mu_tor_from_pad(pad_rel * samples.r_obj, cfg.mu_t, cfg.hertz_frac)
    grid_ok = _feasible_candidates(samples, centers_of[(ds, oid)], tip_r)
    parts = {}
    if len(patches) >= 2:
        for part in G.enumerate_partitions(patches, N):
            arms = [P.arm_geometry(patches, g, samples, cfg) for g in part]
            sets = [P.rule_sets(samples, grid_ok, c, nrm, rho)[RULE] for c, nrm, rho in arms]
            if all(len(s) for s in sets):
                combos = _top_combos(sets, samples, int(np.prod([len(s) for s in sets])), min_sep=2.4 * tip_r)
            else:
                combos = np.zeros((0, N), dtype=np.int64)
            parts[json.dumps(part)] = dict(arms=arms, sets=sets, combos=combos)
    return dict(gid=gid, samples=samples, mu_tor=mu_tor, parts=parts)


def find_row(combos, c):
    r = np.nonzero((combos == c).all(1))[0]
    return int(r[0]) if len(r) else -1


def new_fields():
    return dict(c128_ok=False, c128_cand=np.full((4, KMAX), -1, np.int16),
                c128_a_idx=np.full(4, -1, np.int16), c128_b_idx=np.full(4, -1, np.int16),
                c128_near_idx=np.full(4, -1, np.int16), c128_prox_idx=np.full(4, -1, np.int16),
                c128_alpha=np.nan, c128_snap_alpha=np.nan, c128_near_alpha=np.nan, c128_prox_alpha=np.nan,
                c128_n_combo=0)


def record_fields(p, samples, stored_pos):
    """팔 3대 레코드 하나의 c128_* 값. p = prepare 가 만든 파티션 정보 (+ 채점 결과 p['ma'])."""
    f = new_fields()
    sets, combos = p["sets"], p["combos"]
    for g in range(N):
        f["c128_cand"][g, :len(sets[g])] = sets[g]
    f["c128_n_combo"] = int(len(combos))
    if len(combos) == 0:
        return f
    ma = p["ma"]
    b = int(ma.argmax())
    f["c128_a_idx"][:N] = combos[b]
    f["c128_alpha"] = float(ma[b])
    f["c128_ok"] = bool(ma[b] > ALPHA_MIN)
    f["c128_prox_idx"][:N] = combos[0]
    f["c128_prox_alpha"] = float(ma[0])
    near = np.array([s[np.linalg.norm(samples.points[s] - p["arms"][g][0], axis=1).argmin()]
                     for g, s in enumerate(sets)])
    snap = np.array([s[np.linalg.norm(samples.points[s] - stored_pos[g], axis=1).argmin()]
                     for g, s in enumerate(sets)])
    f["c128_near_idx"][:N], f["c128_b_idx"][:N] = near, snap
    r = find_row(combos, near)
    if r >= 0:
        f["c128_near_alpha"] = float(ma[r])
    r = find_row(combos, snap)
    if r >= 0:
        f["c128_snap_alpha"] = float(ma[r])
    return f


class Writer:
    """레코드 행을 모아 묶음 파일로 쓴다. 변형(A, B)마다 anchor_pos 만 다르다."""

    def __init__(self, dirs, start=0):
        self.dirs, self.k, self.rows, self.n_grasp = dirs, start, [], 0
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def add_grasp(self, rows):
        self.rows += rows
        self.n_grasp += 1
        if self.n_grasp % FLUSH == 0:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        rows = self.rows
        common = {c: np.array([r[c] for r in rows]) for c in BASE_COLS if c != "anchor_pos"}
        for c in new_fields():
            common[c] = np.stack([np.asarray(r[c]) for r in rows])
        for variant, d in self.dirs.items():
            np.savez_compressed(d / f"shard_{self.k:03d}.npz",
                                anchor_pos=np.stack([r["pos_" + variant] for r in rows]), **common)
        self.k += 1
        self.rows = []


def run(gids, rec, by_gid, obj_of, writer, cfg, stat, label, e003=None, verify_stored=False, t_budget=None,
        spent=0.0):
    """spent: 이 단계를 시작하기 전까지 정답 생성에 이미 쓴 시간(초). t_budget(시간)은 그것을 포함한 상한."""
    centers_of, bad_obj = stat["centers_of"], stat["bad_obj"]
    t0 = time.time()
    checked = False
    for c0 in range(0, len(gids), CHUNK):
        ctxs = []
        for gid in gids[c0:c0 + CHUNK]:
            ds, oid = obj_of[gid]
            try:
                ctxs.append(prepare(gid, ds, oid, cfg, centers_of, bad_obj))
            except Exception as e:
                stat["n_fail"] += 1
                print(f"  실패 {gid}: {e!r}", flush=True)
                ctxs.append(dict(gid=gid, samples=None, mu_tor=0.0, parts={}))
        entries, slots = [], []
        for g in ctxs:
            for p in g["parts"].values():
                if len(p["combos"]):
                    entries.append((g["samples"], p["combos"], g["mu_tor"]))
                    slots.append(("part", p))
            if verify_stored and g["samples"] is not None:
                for i in by_gid[g["gid"]]:
                    if int(rec["n_anchors"][i]) != N:
                        continue
                    dd = np.linalg.norm(g["samples"].points[None] - rec["anchor_pos"][i][:N][:, None], axis=-1)
                    if dd.min(1).max() < 1e-7:
                        entries.append((g["samples"], dd.argmin(1)[None].astype(np.int64), g["mu_tor"]))
                        slots.append(("stored", i))
                    else:
                        stat["v3_unmatched"] += 1
        alpha = RB._alpha_many(entries, cfg) if entries else np.zeros((0, 14))
        stat["lp"] += len(alpha)
        ofs = 0
        for (kind, obj), (_s, combos, _m) in zip(slots, entries):
            a = alpha[ofs:ofs + len(combos)]
            ofs += len(combos)
            if kind == "part":
                obj["ma"] = a.min(1)
            else:
                stat["v3_n"] += 1
                stat["v3_same"] += int(abs(float(a.min()) - float(rec["min_alpha"][obj])) < 1e-6)
        for g in ctxs:
            rows = []
            for i in by_gid[g["gid"]]:
                row = {c: rec[c][i] for c in BASE_COLS if c != "anchor_pos"}
                pos = rec["anchor_pos"][i]
                row["pos_a"], row["pos_b"] = pos.copy(), pos.copy()
                f = new_fields()
                if int(rec["n_anchors"][i]) == N:
                    stat["n3"] += 1
                    ps = str(rec["partition"][i])
                    p = g["parts"].get(ps)
                    if p is None:
                        stat["n3_nopart"] += 1
                    else:
                        f = record_fields(p, g["samples"], pos[:N])
                        if len(p["combos"]) == 0:
                            stat["n3_nocombo"] += 1
                        elif not f["c128_ok"]:
                            stat["n3_zero"] += 1
                        if f["c128_ok"]:
                            stat["n3_ok"] += 1
                            row["pos_a"][:N] = g["samples"].points[f["c128_a_idx"][:N]]
                            row["pos_b"][:N] = g["samples"].points[f["c128_b_idx"][:N]]
                            stat["snap_feasible"] += int(np.isfinite(f["c128_snap_alpha"]))
                            stat["snap_same"] += int(np.array_equal(f["c128_a_idx"], f["c128_b_idx"]))
                            if np.isfinite(f["c128_snap_alpha"]):
                                stat["snap_ratio"].append(f["c128_snap_alpha"] / f["c128_alpha"])
                        if e003 is not None:
                            e = e003.get((g["gid"], ps, round(float(rec["pad_rel"][i]), 4)), {}).get(RULE)
                            if e is not None:
                                stat["v2_n"] += 1
                                ea = e["alpha"]
                                if ea is None:
                                    same = len(p["combos"]) == 0
                                else:
                                    same = len(p["combos"]) > 0 and abs(ea - f["c128_alpha"]) < 1e-6
                                    if same and ea > ALPHA_MIN:
                                        stat["v2_best_n"] += 1
                                        stat["v2_best_same"] += int(np.array_equal(e["best"], f["c128_a_idx"][:N]))
                                stat["v2_same"] += int(same)
                row.update(f)
                rows.append(row)
            writer.add_grasp(rows)
        done = min(c0 + CHUNK, len(gids))
        el = time.time() - t0
        proj = (spent + el / done * len(gids)) / 3600          # 이미 쓴 시간 + 이 단계 전체 예상
        if (c0 // CHUNK) % 20 == 0 or done == len(gids):
            print(f"  {label}: 파지 {done}/{len(gids)}, 선형계획 누적 {stat['lp']:,}회, 경과 {el / 60:.1f}분, "
                  f"생성 전체 예상 {proj:.2f}시간 (이미 쓴 {spent / 60:.0f}분 포함)", flush=True)
        if t_budget is not None:
            if done >= 1000 and not checked:
                checked = True
                print(f"[중단 조건 확인] 학습 파지 {done}개 시점의 생성 전체 예상 {proj:.2f}시간 (상한 {t_budget}시간)", flush=True)
                if proj > t_budget:
                    writer.flush()
                    print("중단 조건: 예상 시간이 상한을 넘습니다 — 멈춥니다.", flush=True)
                    sys.exit(3)
            if (spent + el) / 3600 >= t_budget:
                writer.flush()
                stat["capped"] = done
                print(f"시간 상한 {t_budget}시간에 닿았습니다 — 학습 파지 {done}/{len(gids)}개까지 만들고 끝냅니다.", flush=True)
                return
    writer.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-select", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--spent-min", type=float, default=0.0)
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    out_a, out_b, out_e = (DATA / f"search_records_c128a{sfx}", DATA / f"search_records_c128b{sfx}",
                           DATA / f"search_records_c128_eval{sfx}")
    t0 = time.time()
    cfg = SearchConfig()

    rec = load_base()
    by_gid = defaultdict(list)
    has_n3 = set()
    for i, (g, n) in enumerate(zip(rec["grasp_id"], rec["n_anchors"])):
        by_gid[g].append(i)
        if n == N:
            has_n3.add(g)
    obj_of = {}
    for line in (DATA / "pilot_manifest.tsv").read_text().splitlines()[1:]:
        _t, ds, oid, rel, ridx, _s, _n = line.split("\t")
        obj_of[f"{rel}#{ridx}"] = (ds, oid)
    print(f"원본 레코드 {len(rec['grasp_id']):,}행, 파지 {len(by_gid):,}개, 팔 3대 레코드가 있는 파지 {len(has_n3):,}개 "
          f"(읽기 {time.time() - t0:.0f}s)", flush=True)

    ev = [g for g, _, _ in G.load_grasps()]
    pool_val = sorted(g for g in has_n3 if is_val(obj_of[g][1]) and g not in set(ev))
    pool_tr = sorted(g for g in has_n3 if not is_val(obj_of[g][1]))
    if args.smoke:
        ev, n_tr, n_sel = ev[:8], 40, 4
    else:
        n_tr, n_sel = args.n_train, args.n_select
    sel, tr = spaced(pool_val, n_sel), spaced(pool_tr, n_tr)
    # ── 검산 1: 물체 분리 ──
    o_tr = {obj_of[g][1] for g in tr}
    o_va = {obj_of[g][1] for g in sel} | {obj_of[g][1] for g in ev}
    ok1 = (not (o_tr & o_va) and all(is_val(obj_of[g][1]) for g in ev + sel)
           and not any(is_val(obj_of[g][1]) for g in tr))
    print(f"[검산 1] 학습 파지 {len(tr):,}개(물체 {len(o_tr):,}종) / 모델 선택용 {len(sel)}개 / 평가용 {len(ev)}개 "
          f"(검증 쪽 물체 {len(o_va)}종), 겹치는 물체 {len(o_tr & o_va)}종 → {'통과' if ok1 else '불통과'}", flush=True)
    if not ok1:
        sys.exit(2)

    mk = lambda: dict(lp=0, n_fail=0, n3=0, n3_ok=0, n3_nopart=0, n3_nocombo=0, n3_zero=0, snap_feasible=0,
                      snap_same=0, snap_ratio=[], v2_n=0, v2_same=0, v2_best_n=0, v2_best_same=0,
                      v3_n=0, v3_same=0, v3_unmatched=0, centers_of={}, bad_obj=[])
    e003 = pickle.load(open(P.OUT_PKL, "rb"))["out128"]["recs"]

    done_g, k0 = set(), 0
    if args.resume:
        for p in sorted(out_a.glob("shard_*.npz")):
            assert (out_b / p.name).exists(), f"B 폴더에 같은 묶음이 없음: {p.name}"
            with np.load(p, allow_pickle=False) as z:
                done_g |= set(z["grasp_id"].tolist())
            k0 += 1
        print(f"[이어 하기] 이미 만든 묶음 {k0}개, 파지 {len(done_g):,}개는 건너뜁니다. 앞선 실행에서 쓴 시간 {args.spent_min:.0f}분", flush=True)
    skip_eval = args.resume and any(out_e.glob("shard_*.npz"))

    # ── 평가용 480개 (먼저: 검산 2·3) ──
    st_e = mk()
    if skip_eval:
        print("[이어 하기] 평가용 레코드가 이미 있어 건너뜁니다 (검산 2·3 은 앞선 실행에서 통과).", flush=True)
        st_e["v2_n"] = st_e["v2_same"] = st_e["v3_n"] = st_e["v3_same"] = 1
    else:
        run(ev, rec, by_gid, obj_of, Writer({"a": out_e}), cfg, st_e, "평가용", e003=e003, verify_stored=True)
    r2 = st_e["v2_same"] / max(st_e["v2_n"], 1)
    r3 = st_e["v3_same"] / max(st_e["v3_n"] + st_e["v3_unmatched"], 1)
    print(f"[검산 2] 새 정답 α 가 EXP-003 의 규칙 {RULE} 결과와 같은 레코드 {st_e['v2_same']}/{st_e['v2_n']} = {100 * r2:.2f}% "
          f"(그중 정답 점까지 같은 레코드 {st_e['v2_best_same']}/{st_e['v2_best_n']})")
    print(f"[검산 3] 저장된 정답의 α 를 다시 계산해 저장값과 같은 레코드 {st_e['v3_same']}/{st_e['v3_n'] + st_e['v3_unmatched']} "
          f"= {100 * r3:.2f}%")
    ok = r2 >= VERIFY_MIN and r3 >= VERIFY_MIN and not st_e["bad_obj"]
    print(f"[검산 2·3 판정] {'통과' if ok else '불통과'}", flush=True)
    if not ok:
        sys.exit(2)

    # ── 모델 선택용 + 학습용 ──
    st_t = mk()
    w = Writer({"a": out_a, "b": out_b}, start=k0)
    sel_rem = [g for g in sel if g not in done_g]
    perm = np.random.default_rng(0).permutation(len(tr))      # 고정된 무작위 순서 (처리 순서만 바꾼다)
    tr_rem = [tr[i] for i in perm if tr[i] not in done_g]
    run(sel_rem, rec, by_gid, obj_of, w, cfg, st_t, "모델 선택용")
    run(tr_rem, rec, by_gid, obj_of, w, cfg, st_t, "학습용", t_budget=None if args.smoke else TIME_CAP_H,
        spent=args.spent_min * 60 + (time.time() - t0))
    bad = st_e["bad_obj"] + st_t["bad_obj"]
    print(f"[검산 4] 조각 캐시의 중심 좌표가 표면점과 다른 물체 {len(bad)}종 → {'통과' if not bad else '불통과'}")

    # 묶음 파일 전체를 다시 읽어 센 값 (이어 하기를 해도 전체 집계가 맞도록)
    for name, d in (("평가용 폴더 전체", out_e), ("학습·선택용 폴더 전체", out_a)):
        g_all, n3, okn, feas, same = set(), 0, 0, 0, 0
        for p in sorted(d.glob("shard_*.npz")):
            with np.load(p, allow_pickle=False) as z:
                m3 = z["n_anchors"] == N
                ok_ = z["c128_ok"] & m3
                g_all |= set(z["grasp_id"].tolist())
                n3 += int(m3.sum()); okn += int(ok_.sum())
                feas += int(np.isfinite(z["c128_snap_alpha"][ok_]).sum())
                same += int((z["c128_a_idx"][ok_] == z["c128_b_idx"][ok_]).all(1).sum())
        print(f"[{name}] 파지 {len(g_all):,}개, 팔 3대 레코드 {n3:,}개 중 과제로 쓰는 레코드 {okn:,}개 ({100 * okn / max(n3, 1):.1f}%) | "
              f"옮긴 저장 정답(B)이 놓을 수 있는 조합 {100 * feas / max(okn, 1):.1f}%, B 가 A 와 같은 조합 {100 * same / max(okn, 1):.1f}%")
    summ = {}
    for name, st in (("평가용(이번 실행)", st_e), ("학습·선택용(이번 실행)", st_t)):
        sr = np.array(st["snap_ratio"]) if st["snap_ratio"] else np.array([np.nan])
        s = dict(n3=st["n3"], n3_ok=st["n3_ok"], n3_nopart=st["n3_nopart"], n3_nocombo=st["n3_nocombo"],
                 n3_zero=st["n3_zero"], snap_feasible=st["snap_feasible"], snap_same=st["snap_same"],
                 snap_ratio_median=float(np.median(sr)), lp=st["lp"], n_fail=st["n_fail"])
        summ[name] = s
        ok_n = max(st["n3_ok"], 1)
        print(f"[{name}] 팔 3대 레코드 {st['n3']:,}개 중 과제로 쓰는 레코드 {st['n3_ok']:,}개 "
              f"(뺀 것: 조합 없음 {st['n3_nocombo']:,}, 최선 α 가 0 {st['n3_zero']:,}, 파티션 불일치 {st['n3_nopart']:,}) | "
              f"옮긴 저장 정답(B)이 놓을 수 있는 조합인 레코드 {100 * st['snap_feasible'] / ok_n:.1f}%, "
              f"B 가 A 와 같은 조합인 레코드 {100 * st['snap_same'] / ok_n:.1f}%, B 의 α ÷ A 의 α 중앙값 {s['snap_ratio_median']:.3f} | "
              f"선형계획 {st['lp']:,}회, 불러오기 실패 {st['n_fail']}")
    (ROOT / f"scratchpad/gen_c128_records_summary{sfx}.json").write_text(
        json.dumps(dict(summary=summ, n_train=len(tr), n_select=len(sel), n_eval=len(ev),
                        rule=RULE, alpha_min=ALPHA_MIN, seconds=time.time() - t0), ensure_ascii=False, indent=1))
    print(f"저장 {out_a.name}, {out_b.name}, {out_e.name} — 총 {(time.time() - t0) / 60:.1f}분", flush=True)
    if bad:
        sys.exit(2)


if __name__ == "__main__":
    main()
