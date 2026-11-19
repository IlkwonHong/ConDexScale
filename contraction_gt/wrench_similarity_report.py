"""렌치 공간 닮음 — 판정 가·나 + 집단 그림 (계획서 §5-5, 기준치 §1-5).

입력: data/contraction_gt_300k/wrench_similarity/{records.npz, grasps.npz, realized_contacts.pkl(선택)}
출력: md_claude/gt_viz/wrench_similarity/report.md + PNG 그림

판정 가: 같은 파지 × 같은 팔 개수 안에서 α(min_alpha) ±20% 로 맞춘 합격·불합격 쌍 →
        불합격 쪽 "잃어버린 양"이 더 큰 비율 (동률 0.5). 성립 ≥ 0.60 (보조 AUC ≥ 0.60).
판정 나: 합격 기록 ≥3 개인 파지 × 팔 개수에서 닮음 순위(−잃어버린 양)와 α 순위의 스피어만 상관 중앙값.
        성립 < 0.5.
β ∈ {0.1, 0.2} × 요약 {lost_frac, mean_short} 네 조합 모두 성립이어야 "성립", 일부면 "약한 성립".

실행: ./venv/bin/python -m contraction_gt.wrench_similarity_report
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

IN = Path("data/contraction_gt_300k/wrench_similarity")
OUT = Path("md_claude/gt_viz/wrench_similarity")
BETAS = ("b10", "b20")
METRICS = ("lost_frac", "mean_short")
# 그림 색 (dataviz 기본 팔레트 1~3번 슬롯, 검증기 통과) — 판정: 합격/불합격/정착 실패, 렌치 공간: 사람/계획/실제
C_VERDICT = {"pass": "#2a78d6", "fail": "#eb6834", "fail_settle": "#1baf7a"}
K_VERDICT = {"pass": "합격", "fail": "불합격", "fail_settle": "정착 실패"}
C_SPACE = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
TEXT1, TEXT2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"


def auc(x_fail, x_pass):
    """Mann–Whitney AUC = P(불합격 점수 > 합격 점수): 무작위 불합격 하나와 합격 하나를 뽑았을 때
    불합격 쪽 점수가 더 큰 확률. 0.5 = 구분 못 함, 1 = 완벽 구분, 0.5 미만 = 반대 방향."""
    x_fail, x_pass = np.asarray(x_fail, float), np.asarray(x_pass, float)
    x_fail, x_pass = x_fail[~np.isnan(x_fail)], x_pass[~np.isnan(x_pass)]
    r = rankdata(np.concatenate([x_fail, x_pass]))
    u = r[: len(x_fail)].sum() - len(x_fail) * (len(x_fail) + 1) / 2
    return float(u / (len(x_fail) * len(x_pass)))


def paired_test(rec, metric_key):
    """같은 (파지, N) 안에서 불합격 기록마다 α 가 ±20% 안에서 가장 가까운 합격 기록을 짝지어 비교."""
    gidx, n, v, a = rec["grasp_idx"], rec["n_anchors"], rec["verdict"], rec["min_alpha"].astype(float)
    val = rec[metric_key].astype(float)
    groups = defaultdict(lambda: ([], []))
    for i in range(len(v)):
        if v[i] == "pass":
            groups[(gidx[i], n[i])][0].append(i)
        elif v[i] == "fail":
            groups[(gidx[i], n[i])][1].append(i)
    wins = ties = n_pairs = 0
    diffs, dalpha = [], []
    for (g, N), (ps, fs) in groups.items():
        if not ps or not fs:
            continue
        pa = a[ps]
        for f in fs:
            rel = np.abs(pa - a[f]) / np.maximum(pa, 1e-9)
            ok = np.nonzero(rel <= 0.2)[0]
            if len(ok) == 0:
                continue
            p = ps[ok[np.argmin(rel[ok])]]
            if np.isnan(val[f]) or np.isnan(val[p]):
                continue
            n_pairs += 1
            if val[f] > val[p]:
                wins += 1
            elif val[f] == val[p]:
                ties += 1
            diffs.append(val[f] - val[p]); dalpha.append(a[f] - a[p])
    frac = (wins + 0.5 * ties) / max(n_pairs, 1)
    se = np.sqrt(frac * (1 - frac) / max(n_pairs, 1))
    return dict(n_pairs=n_pairs, win_frac=frac, ci95=(frac - 1.96 * se, frac + 1.96 * se),
                mean_diff=float(np.mean(diffs)) if diffs else float("nan"),
                median_diff=float(np.median(diffs)) if diffs else float("nan"),
                mean_dalpha=float(np.mean(dalpha)) if dalpha else float("nan"), diffs=np.array(diffs))


def within_grasp_rank(rec, metric_key, min_pass=3):
    """합격 기록 ≥ min_pass 인 (파지, N) 마다 스피어만(−잃어버린 양, α)."""
    gidx, n, v, a = rec["grasp_idx"], rec["n_anchors"], rec["verdict"], rec["min_alpha"].astype(float)
    val = rec[metric_key].astype(float)
    groups = defaultdict(list)
    for i in np.nonzero(v == "pass")[0]:
        groups[(gidx[i], n[i])].append(i)
    rhos = []
    for ids in groups.values():
        if len(ids) < min_pass:
            continue
        x, y = -val[ids], a[ids]
        if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
            rhos.append(0.0)
            continue
        rhos.append(float(spearmanr(x, y, nan_policy="omit").statistic))
    rhos = np.array(rhos)
    return dict(n_groups=len(rhos), median=float(np.nanmedian(rhos)), mean=float(np.nanmean(rhos)),
                frac_below_05=float(np.mean(rhos < 0.5)), frac_abs_below_03=float(np.mean(np.abs(rhos) < 0.3)),
                rhos=rhos)


def realized_analysis(rec, gr):
    """C(실제 접촉) 표본: 계획↔실제 거리·각도, 그리고 A→B, A→C, B→C 잃어버린 양."""
    p = IN / "realized_contacts.pkl"
    if not p.exists():
        return None
    import torch
    from .wrench_similarity_eval import MU, N_EDGES, N_DIRS, SEED
    from .wrench_support import sample_directions, cone_edge_wrenches, support_max, similarity
    res = pickle.load(open(p, "rb"))
    idx_to_row = {int(i): k for k, i in enumerate(rec["idx"])}
    dirs = sample_directions(N_DIRS, seed=SEED)
    out = defaultdict(list)
    for r in res:
        k = idx_to_row.get(int(r["idx"]))
        if k is None:
            continue
        g = rec["grasp_idx"][k]
        com = torch.as_tensor(gr["com"][g], dtype=torch.float64)[None]
        robj = float(gr["r_obj"][g])
        def sig(pos, nrm):
            if len(pos) == 0:
                return torch.zeros(1, N_DIRS, dtype=torch.float64)
            W = cone_edge_wrenches(torch.as_tensor(np.asarray(pos), dtype=torch.float64)[None],
                                   torch.as_tensor(np.asarray(nrm), dtype=torch.float64)[None], com, robj,
                                   mu=MU, n_edges=N_EDGES)
            return support_max(W, dirs)
        sA = torch.as_tensor(gr["sig_h23"][g], dtype=torch.float64)[None]
        sB = sig(r["planned_pos"], r["planned_nrm"])
        sC = sig([c["pos_obj"] for c in r["contacts"]], [c["normal_obj"] for c in r["contacts"]])
        v = r["verdict"]
        out["verdict"].append(v)
        out["settled"].append(r["settled"])
        out["n_missing_tip"].append(int((r["tip_ncontact"] == 0).sum()))
        out["tip_dist_rel"].append(np.nanmedian(r["tip_dist"] / r["tip_radius"]) if np.isfinite(r["tip_dist"]).any() else np.nan)
        out["tip_dist_max_rel"].append(np.nanmax(r["tip_dist"] / r["tip_radius"]) if np.isfinite(r["tip_dist"]).any() else np.nan)
        out["tip_angle_max"].append(np.nanmax(r["tip_angle_deg"]) if np.isfinite(r["tip_angle_deg"]).any() else np.nan)
        for name, (h, rr) in {"AB": (sA, sB), "AC": (sA, sC), "BC": (sB, sC)}.items():
            s = similarity(h, rr, beta=0.1)
            out[f"lost_{name}"].append(float(s["lost_frac"][0]))
            out[f"short_{name}"].append(float(s["mean_short"][0]))
        out["sigC_max"].append(float(sC.max()))
        out["sigB_max"].append(float(sB.max()))
    return {k: np.array(v) for k, v in out.items()}


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as _fm
    _fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
    _fam = _fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf").get_name()
    plt.rcParams.update({"font.family": [_fam, "DejaVu Sans"],
                         "axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                         "ytick.color": TEXT2, "text.color": TEXT1, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    OUT.mkdir(parents=True, exist_ok=True)
    rec = dict(np.load(IN / "records.npz", allow_pickle=True))
    gr = dict(np.load(IN / "grasps.npz", allow_pickle=True))
    meta = json.load(open(IN / "meta.json"))
    v = rec["verdict"]
    L = []
    L.append(f"# 렌치 공간 닮음 — 판정 결과 (자동 생성)\n")
    L.append(f"기록 {meta['n_records']:,}개 / 파지 {meta['n_grasps']:,}개, μ={meta['mu']}, 모서리 {meta['n_edges']}, 방향 {meta['n_dirs']}.\n")
    L.append("잃어버린 양 = 사람 손(마디 23개, 또는 부위 6개)이 밀 수 있는 방향 중 로봇(계획 앵커)이 (1−β) 배 미만으로 미는 방향의 비율(lost_frac)과, 그 방향들의 평균 부족분(mean_short).\n")

    # ---- 집단 분포 + AUC
    L.append("## 1. 집단 분포 (중앙값) 와 구분력 AUC\n")
    L.append("| 사람 쪽 | β | 요약 | 합격 | 불합격 | 정착 실패 | AUC (불합격이 더 잃음) | AUC (정착 실패가 더 잃음) | N=3 AUC | N=4 AUC |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    aucs = {}
    for hs in ("h23", "h6"):
        for b in BETAS:
            for mt in METRICS:
                key = f"{hs}__{mt}_{b}"; x = rec[key].astype(float)
                med = {k: float(np.nanmedian(x[v == k])) for k in K_VERDICT}
                a_pf = auc(x[v == "fail"], x[v == "pass"]); a_ps = auc(x[v == "fail_settle"], x[v == "pass"])
                a_n = {N: auc(x[(v == "fail") & (rec["n_anchors"] == N)], x[(v == "pass") & (rec["n_anchors"] == N)]) for N in (3, 4)}
                aucs[key] = a_pf
                L.append(f"| {hs} | {b[1]}.{b[2]} | {mt} | {med['pass']:.3f} | {med['fail']:.3f} | {med['fail_settle']:.3f} | "
                         f"**{a_pf:.3f}** | {a_ps:.3f} | {a_n[3]:.3f} | {a_n[4]:.3f} |")
    xa = rec["min_alpha"].astype(float)
    L.append(f"| (참고) α | — | −min_alpha (불합격이 더 작은 α) | {np.median(xa[v=='pass']):.3f} | {np.median(xa[v=='fail']):.3f} | "
             f"{np.median(xa[v=='fail_settle']):.3f} | {auc(-xa[v=='fail'], -xa[v=='pass']):.3f} | {auc(-xa[v=='fail_settle'], -xa[v=='pass']):.3f} | | |\n")

    # ---- 판정 가
    L.append("## 2. 판정 가 — α 를 맞춘 같은 파지 쌍에서 불합격 쪽이 더 많이 잃은 비율 (성립 기준 ≥ 0.60, 보조 AUC ≥ 0.60)\n")
    L.append("| 사람 쪽 | β | 요약 | 쌍 수 | 불합격이 더 잃은 비율 (95% 구간) | 차이 중앙값 (불합격−합격) | α 차이 평균 | 판정 |")
    L.append("|---|---|---|---|---|---|---|---|")
    ga_pass = {}
    paired_cache = {}
    for hs in ("h23", "h6"):
        for b in BETAS:
            for mt in METRICS:
                key = f"{hs}__{mt}_{b}"
                pt = paired_test(rec, key); paired_cache[key] = pt
                ok = pt["win_frac"] >= 0.60 and aucs[key] >= 0.60
                if hs == "h23":
                    ga_pass[(b, mt)] = ok
                L.append(f"| {hs} | {b[1]}.{b[2]} | {mt} | {pt['n_pairs']:,} | **{pt['win_frac']:.3f}** ({pt['ci95'][0]:.3f}~{pt['ci95'][1]:.3f}) | "
                         f"{pt['median_diff']:+.4f} | {pt['mean_dalpha']:+.4f} | {'성립' if ok else '불성립'} |")
    n_ok = sum(ga_pass.values())
    verdict_ga = "성립" if n_ok == 4 else ("약한 성립" if n_ok > 0 else "불성립")
    L.append(f"\n**판정 가 (마디 23개 기준, 4조합 중 {n_ok}개 성립): {verdict_ga}**\n")

    # ---- 판정 나
    L.append("## 3. 판정 나 — 합격 배치들 사이에서 닮음 순위 vs α 순위 (스피어만, 성립 기준 중앙값 < 0.5)\n")
    L.append("| 사람 쪽 | β | 요약 | 묶음 수 (합격 ≥3) | 상관 중앙값 | 평균 | 0.5 미만 비율 | 절댓값 0.3 미만 비율 | 판정 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    na_pass = {}
    rank_cache = {}
    for hs in ("h23", "h6"):
        for b in BETAS:
            for mt in METRICS:
                key = f"{hs}__{mt}_{b}"
                wr = within_grasp_rank(rec, key); rank_cache[key] = wr
                ok = wr["median"] < 0.5
                if hs == "h23":
                    na_pass[(b, mt)] = ok
                L.append(f"| {hs} | {b[1]}.{b[2]} | {mt} | {wr['n_groups']:,} | **{wr['median']:.3f}** | {wr['mean']:.3f} | "
                         f"{wr['frac_below_05']:.3f} | {wr['frac_abs_below_03']:.3f} | {'성립' if ok else '불성립'} |")
    n_ok2 = sum(na_pass.values())
    verdict_na = "성립" if n_ok2 == 4 else ("약한 성립" if n_ok2 > 0 else "불성립")
    L.append(f"\n**판정 나 (마디 23개 기준, 4조합 중 {n_ok2}개 성립): {verdict_na}**\n")

    # ---- 분해: 23마디 → 6부위 → N앵커
    L.append("## 4. 분해 — 렌치 공간이 어느 단계에서 줄어드는가 (β=0.1, lost_frac 중앙값)\n")
    l_23_6 = gr["h23_vs_h6__lost_frac_b10"].astype(float)
    L.append(f"- 마디 23개 → 부위 6개 (파지 수준): {np.nanmedian(l_23_6):.3f}")
    for k in K_VERDICT:
        m = v == k
        L.append(f"- 부위 6개 → 앵커 ({K_VERDICT[k]}): {np.nanmedian(rec['h6__lost_frac_b10'][m]):.3f} ; 마디 23개 → 앵커: {np.nanmedian(rec['h23__lost_frac_b10'][m]):.3f}")
    L.append(f"- 팔 개수별 (마디 23개 → 앵커, 합격): N=3 {np.nanmedian(rec['h23__lost_frac_b10'][(v=='pass')&(rec['n_anchors']==3)]):.3f}, "
             f"N=4 {np.nanmedian(rec['h23__lost_frac_b10'][(v=='pass')&(rec['n_anchors']==4)]):.3f}\n")

    # ---- C 표본
    ra = realized_analysis(rec, gr)
    L.append("## 5. 실제 접촉(C) 표본 — 계획 대 실제\n")
    if ra is None:
        L.append("(realized_contacts.pkl 없음 — 표본 시뮬 후 다시 실행)\n")
    else:
        L.append("| 판정 | 표본 | 정착 비율 | 접촉 없는 팁 수 평균 | 계획↔실제 거리 / 팁 반지름 (팁별 최대의 중앙값) | 법선 각도 최대 (중앙값, °) | 잃음 A→B | 잃음 A→C | 잃음 B→C |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for k in K_VERDICT:
            m = ra["verdict"] == k
            if m.sum() == 0:
                continue
            L.append(f"| {K_VERDICT[k]} | {m.sum()} | {ra['settled'][m].mean():.2f} | {ra['n_missing_tip'][m].mean():.2f} | "
                     f"{fmt(np.nanmedian(ra['tip_dist_max_rel'][m]), 2)} | {fmt(np.nanmedian(ra['tip_angle_max'][m]), 1)} | "
                     f"{np.nanmedian(ra['lost_AB'][m]):.3f} | {np.nanmedian(ra['lost_AC'][m]):.3f} | **{np.nanmedian(ra['lost_BC'][m]):.3f}** |")
        s = (ra["verdict"] == "pass") | (ra["verdict"] == "fail")
        L.append(f"\n- B→C 잃음으로 합격·불합격을 가르는 AUC: {auc(ra['lost_BC'][ra['verdict']=='fail'], ra['lost_BC'][ra['verdict']=='pass']):.3f} ; "
                 f"계획↔실제 거리(최대) {auc(ra['tip_dist_max_rel'][ra['verdict']=='fail'], ra['tip_dist_max_rel'][ra['verdict']=='pass']):.3f} ; "
                 f"A→C 잃음 {auc(ra['lost_AC'][ra['verdict']=='fail'], ra['lost_AC'][ra['verdict']=='pass']):.3f}\n")

    # ---- 그림
    key = "h23__lost_frac_b10"
    # 그림 1: 판정별 분포
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    for k in ("pass", "fail", "fail_settle"):
        x = rec[key][v == k]
        ax.hist(x, bins=bins, histtype="step", linewidth=2, color=C_VERDICT[k], density=True,
                label=f"{K_VERDICT[k]} (n={int((v==k).sum()):,}, 중앙값 {np.nanmedian(x):.2f})")
    ax.set_xlabel("잃어버린 방향의 비율 (사람 손 마디 23개 대비, β=0.1)"); ax.set_ylabel("밀도")
    ax.set_title("계획 앵커가 사람 손 대비 잃어버린 방향 비율 — 시뮬 판정별", loc="left", color=TEXT1)
    ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig(OUT / "fig1_lost_frac_by_verdict.png", dpi=150); plt.close(fig)
    # 그림 2: 쌍 차이
    pt = paired_cache[key]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pt["diffs"], bins=np.linspace(-0.6, 0.6, 61), color=C_VERDICT["fail"], alpha=0.85, linewidth=0)
    ax.axvline(0, color=TEXT2, linewidth=1)
    ax.set_xlabel("잃어버린 비율 차이 (불합격 − 합격), 같은 파지·같은 팔 개수·α ±20% 쌍")
    ax.set_ylabel("쌍 수")
    ax.set_title(f"판정 가 — 불합격 쪽이 더 잃은 쌍 {pt['win_frac']*100:.1f}% (쌍 {pt['n_pairs']:,}개)", loc="left", color=TEXT1)
    fig.tight_layout(); fig.savefig(OUT / "fig2_paired_diff.png", dpi=150); plt.close(fig)
    # 그림 3: 닮음 vs α 산점 (표본)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in ("pass", "fail"):
        ids = np.nonzero(v == k)[0]; ids = rng.choice(ids, min(6000, len(ids)), replace=False)
        ax.scatter(rec["min_alpha"][ids], rec[key][ids], s=4, alpha=0.25, color=C_VERDICT[k], linewidths=0, label=K_VERDICT[k])
    ax.set_xlabel("α (가장 약한 방향의 내하중 배율)"); ax.set_ylabel("잃어버린 방향의 비율")
    ax.set_title("닮음과 α 는 다른 축인가 — 기록 표본 12,000개", loc="left", color=TEXT1)
    ax.legend(markerscale=4); fig.tight_layout(); fig.savefig(OUT / "fig3_lost_vs_alpha.png", dpi=150); plt.close(fig)
    # 그림 4: 파지 안 순위 상관
    wr = rank_cache[key]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(wr["rhos"], bins=np.linspace(-1, 1, 41), color=C_VERDICT["pass"], alpha=0.85, linewidth=0)
    ax.axvline(wr["median"], color=TEXT1, linewidth=1.5)
    ax.text(wr["median"] + 0.03, ax.get_ylim()[1] * 0.9, f"중앙값 {wr['median']:.2f}", color=TEXT1)
    ax.set_xlabel("스피어만 상관 (−잃어버린 비율, α) — 합격 배치 3개 이상인 파지마다")
    ax.set_ylabel("파지 수"); ax.set_title(f"판정 나 — 파지 안에서 닮음 순위와 α 순위의 일치도 ({wr['n_groups']:,} 묶음)", loc="left", color=TEXT1)
    fig.tight_layout(); fig.savefig(OUT / "fig4_within_grasp_spearman.png", dpi=150); plt.close(fig)
    # 그림 5: 계획 대 실제
    if ra is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for k in ("pass", "fail", "fail_settle"):
            m = ra["verdict"] == k
            x = ra["tip_dist_max_rel"][m]; x = x[np.isfinite(x)]
            axes[0].hist(np.clip(x, 0, 3), bins=np.linspace(0, 3, 31), histtype="step", linewidth=2, color=C_VERDICT[k],
                         label=f"{K_VERDICT[k]} (n={m.sum()})")
            y = ra["lost_BC"][m]
            axes[1].hist(y, bins=np.linspace(0, 1, 26), histtype="step", linewidth=2, color=C_VERDICT[k],
                         label=f"{K_VERDICT[k]} 중앙값 {np.nanmedian(y):.2f}")
        axes[0].set_xlabel("계획 앵커 ↔ 실제 접촉점 거리 (팁 반지름 배, 팁별 최대; 3 이상은 3에 모음)"); axes[0].set_ylabel("기록 수")
        axes[0].set_title("계획 자리와 실제 자리의 어긋남", loc="left", color=TEXT1); axes[0].legend()
        axes[1].set_xlabel("계획(B) 대비 실제(C)가 잃은 방향의 비율 (β=0.1)"); axes[1].set_ylabel("기록 수")
        axes[1].set_title("계획 렌치 공간 → 실제 렌치 공간의 손실", loc="left", color=TEXT1); axes[1].legend()
        fig.tight_layout(); fig.savefig(OUT / "fig5_planned_vs_realized.png", dpi=150); plt.close(fig)

    L.append("## 6. 그림\n")
    for f in sorted(OUT.glob("fig*.png")):
        L.append(f"- `{f.name}`")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
