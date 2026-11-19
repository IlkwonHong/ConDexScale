"""조건부 net2 데이터셋 (2026-10-17) — Stage2Dataset 을 상속, 기존 파일은 그대로.

Stage2Dataset 의 입력(몸통·점 특징·패치 섞기)은 그대로 쓰고, 히트맵 과제만 조건부 형식으로 바꾼다:
  - 그룹 순서 규칙 (학습·추론 공통, canonical_order): 엄지 든 그룹 먼저, 나머지는 손가락 순위
    (검지 0 → 중지 1 → 약지 2 → 소지 3 → 손바닥 4)의 가장 앞 손가락 기준.
  - 정답 조합 최대 MAX_K 벌 (records_format="multi" 면 mode_idx, "single" 이면 기존 레코드 1벌).
  - 작은 흔들림: 정답 한 벌을 골라, 앞 자리(마지막 그룹 제외) 각각을 확률 noise_p 로
    가장 가까운 표면점 6개 중 하나로 바꾼 앞 자리 (학습 때만).
  - 먼 첫 자리: multi 레코드의 첫 자리 조건 표(cond_idx)에서 정답 첫 자리와 다른 줄 하나 (학습 때만).

추가 출력 (모두 고정 크기, -1 패딩):
  cond_ug [6]        부위 → 그룹 (규칙 순서), -1 = 없음
  cond_n             그룹 수
  cond_mask [4,2048] 그룹별 후보 마스크 (Stage2Dataset 과 같은 식, 학습 때 정답 조합 점은 강제 포함)
  cond_tgt [K,4]     정답 조합들의 표면점 인덱스, cond_k = 벌 수
  cond_noise [4], cond_noise_on [4] (bool), cond_noise_m   흔들린 앞 자리와 어느 벌에서 왔는지
  cond_far [4], cond_far_on (bool)                          먼 첫 자리 조합
  cond_pos [2048,3], cond_nrm [2048,3]                      점 위치(무게중심 기준 / r_obj)·안쪽 법선
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .stage2_dataset import Stage2Dataset, UNITS, RHO_REL, MARGIN_REL, ROOT
from .stage1_dataset import N_UNIT, _LINK_UNIT, _LINK_NAMES

MAX_K = 3
_RANK = {"index": 0, "middle": 1, "ring": 2, "little": 3, "palm": 4, "thumb": -1}


def canonical_order(blocks):
    """그룹(부위 이름 목록)들의 순서 → 규칙 순서의 인덱스 목록. 엄지 그룹 먼저, 나머지는 손가락 순위."""
    def key(i):
        b = blocks[i]
        return (0 if "thumb" in b else 1, min(_RANK[u] for u in b))
    return sorted(range(len(blocks)), key=key)


class Stage2CondDataset(Stage2Dataset):
    def __init__(self, *args, records_format="single", noise_p=0.3, noise_knn=6, max_k=MAX_K,
                 cand_from_records=False, only_n=None, **kw):
        # cand_from_records (2026-11-14, EXP-004): 후보 마스크를 접촉에서 다시 계산하지 않고 레코드의
        #   c128_cand [행, 4, K] (표면점 인덱스, -1 패딩) 에서 읽는다. c128_ok 가 거짓인 레코드는 과제에서 뺀다.
        # only_n: 그 팔 수의 레코드만 조건부 과제로 쓴다. 둘 다 기본값이면 기존 동작 그대로.
        kw.setdefault("patch_interp", True)
        super().__init__(*args, **kw)
        self.records_format = records_format
        self.cand_from_records = cand_from_records
        self.only_n = only_n
        self.max_k = max_k                    # 정답 벌 수 상한 (1 = 최선만, 2×2 비교용)
        self.noise_p = noise_p
        self.noise_knn = noise_knn

        # 파지별 조건부 과제: (blocks 규칙 순서, n, combos_idx [K,n] or None, anchor_pos [n,3], far_idx [F,n])
        rec_dir = Path(kw.get("records_dir", ROOT / "data/contraction_gt_300k/search_records"))
        gid_set = {g for g, _ in self.samples}
        self.cond_tasks = defaultdict(list)
        for p in sorted(rec_dir.glob("shard_*.npz")):
            with np.load(p, allow_pickle=False) as z:
                multi = records_format == "multi"
                cols = [z["grasp_id"], z["n_anchors"], z["partition"], z["pad_rel"], z["anchor_pos"]]
                if multi:
                    cols += [z["mode_idx"], z["cond_idx"]]
                if cand_from_records:
                    cols += [z["c128_cand"], z["c128_ok"]]
                for row in zip(*cols):
                    g, n, ps, pr, ap = row[:5]
                    if g not in gid_set:
                        continue
                    n = int(n)
                    if n == 2 and abs(pr - 0.10) > 1e-6:
                        continue                       # 팁 반경 중복 제거 (Stage2Dataset 과 같음)
                    if only_n is not None and n != only_n:
                        continue
                    if cand_from_records and not row[-1]:
                        continue                       # 후보 조합이 없거나 최선 α 가 0 인 레코드
                    blocks = json.loads(ps)
                    order = canonical_order(blocks)
                    blocks_o = [blocks[i] for i in order]
                    if multi:
                        mi, ci = row[5], row[6]
                        combos = mi[(mi[:, 0] >= 0)][:, :n][:, order].astype(np.int64)
                        far = ci[(ci[:, 0] >= 0)][:, :n][:, order].astype(np.int64)
                        # 먼 첫 자리: 정답 첫 자리와 다른 줄만
                        far = far[~np.isin(far[:, 0], combos[:, 0])]
                    else:
                        combos, far = None, np.zeros((0, n), np.int64)
                    cand = row[-2][:n][order].astype(np.int64) if cand_from_records else None
                    self.cond_tasks[g].append((blocks_o, n, combos, ap[:n][order].astype(np.float32), far, cand))
        self.samples = [s for s in self.samples if len(self.cond_tasks[s[0]])]

    def _group_masks(self, con, ridx, ug, n_g, pts_m, r_obj, fallback):
        """Stage2Dataset 과 같은 식의 그룹 마스크 [4,P]."""
        lp, lc, la = (con["link_contact_point"][ridx], con["link_contact"][ridx],
                      con["link_area"][ridx])
        mask = np.zeros((4, len(pts_m)), bool)
        for g in range(n_g):
            members = [u for u in range(N_UNIT) if ug[u] == g]
            li = [j for j in range(len(_LINK_NAMES)) if _LINK_UNIT[j] in members and lc[j]]
            if li:
                w = la[li] / max(la[li].sum(), 1e-12)
                center = (lp[li] * w[:, None]).sum(0)
                spread = float(np.linalg.norm(lp[li] - center, axis=1).max())
            else:
                center, spread = fallback[g], 0.0
            radius = max(spread, RHO_REL * r_obj) + MARGIN_REL * r_obj
            mask[g] = np.linalg.norm(pts_m - center, axis=1) <= radius
        return mask

    def __getitem__(self, i):
        out = super().__getitem__(i)                  # 입력 + (안 쓰는) 기존 히트맵 과제
        gid, e = self.samples[i]
        rel, ridx = gid.rsplit("#", 1)
        ridx = int(ridx)
        r_obj, ts = e["r_obj"], e["total_scale"]
        obj = self._objects.get(self.obj_of[gid])
        con = self._contacts.get(rel)
        pts_m = obj["pts"] * ts

        tasks = self.cond_tasks[gid]
        rng = np.random.default_rng(None if self.augment else self.seed * 31 + i)
        blocks, n_g, combos, ap, far, cand = tasks[int(rng.integers(len(tasks)))]
        if combos is None:                            # single: 기존 레코드 위치 → 최근접 표면점
            combos = np.linalg.norm(pts_m[None, :, :] - ap[:, None, :], axis=-1).argmin(1)[None]
        combos = combos[:self.max_k]
        ug = np.full(N_UNIT, -1, np.int64)
        for g, blk in enumerate(blocks):
            for u in blk:
                ug[UNITS.index(u)] = g

        if cand is not None:                          # 레코드에 저장된 후보 (정답 생성과 같은 집합)
            mask = np.zeros((4, len(pts_m)), bool)
            for g in range(n_g):
                mask[g, cand[g][cand[g] >= 0]] = True
        else:
            mask = self._group_masks(con, ridx, ug, n_g, pts_m, r_obj, pts_m[combos[0]])
        for c in combos:                              # 정답 조합 점은 항상 후보에 포함
            mask[np.arange(n_g), c] = True

        tgt = np.full((MAX_K, 4), -1, np.int64)
        tgt[:len(combos), :n_g] = combos

        noise = np.full(4, -1, np.int64)
        noise_on = np.zeros(4, bool)
        noise_m = 0
        if self.augment and self.noise_p > 0:
            noise_m = int(rng.integers(len(combos)))
            noise[:n_g] = combos[noise_m]
            for k in range(n_g - 1):                  # 마지막 그룹은 앞 자리로 쓰이지 않음
                if rng.random() < self.noise_p:
                    d = np.linalg.norm(pts_m - pts_m[combos[noise_m][k]], axis=1)
                    noise[k] = int(np.argsort(d)[1:1 + self.noise_knn][rng.integers(self.noise_knn)])
                    noise_on[k] = True

        far_c = np.full(4, -1, np.int64)
        far_on = False
        if self.augment and len(far):
            far_c[:n_g] = far[int(rng.integers(len(far)))]
            far_on = True
            mask[np.arange(n_g), far_c[:n_g]] = True

        scale = ts / r_obj
        out["cond_ug"] = torch.from_numpy(ug)
        out["cond_n"] = torch.tensor(n_g, dtype=torch.long)
        out["cond_mask"] = torch.from_numpy(mask)
        out["cond_tgt"] = torch.from_numpy(tgt)
        out["cond_k"] = torch.tensor(len(combos), dtype=torch.long)
        out["cond_noise"] = torch.from_numpy(noise)
        out["cond_noise_on"] = torch.from_numpy(noise_on)
        out["cond_noise_m"] = torch.tensor(noise_m, dtype=torch.long)
        out["cond_far"] = torch.from_numpy(far_c)
        out["cond_far_on"] = torch.tensor(far_on)
        out["cond_pos"] = torch.from_numpy(((obj["pts"] - obj["com"]) * scale).astype(np.float32))
        out["cond_nrm"] = torch.from_numpy(obj["nrm"].astype(np.float32))
        return out
