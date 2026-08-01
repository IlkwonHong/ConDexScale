"""Stage 2 데이터셋 — Stage 1 샘플(파지)에 히트맵 과제를 얹는다.

샘플 = 파지 (Stage1Dataset 그대로) + 그 파지의 레코드(파티션) 1개 추첨:
  hm_unit_group [6]      추첨된 파티션의 부위→그룹 (-1 = 없음)
  hm_n_groups   int      그룹 수 (2~4)
  hm_target     [4]      그룹별 GT 앵커의 표면점 인덱스 (-1 패딩)
  hm_mask       [4,2048] 그룹별 후보 마스크 (참값 중심 + 적응형 반경 + 마진)
  point_feat    [2048,6] 점 로컬 특징 (소속 패치 중심 상대좌표/r_obj 3 + 안쪽 법선 3)
  point_patch   [2048]   점→패치 소속 (브리지 gather 용)
  hm_soft       [4,2048] (가우시안 모드만) soft 정답표

규약: 마스크·정답은 참값(깨끗한 GT 접촉)으로 계산 — 입력 특징 교란과 독립 (명세 3항).
N=2 레코드는 pad_rel=0.10 만 사용 (팁 반경 3종이 같은 앵커 공유 — 중복 제거).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .stage1_dataset import (Stage1Dataset, N_UNIT, _LINK_UNIT, _LINK_NAMES)

ROOT = Path(__file__).resolve().parents[1]
UNITS = ["index", "middle", "ring", "little", "thumb", "palm"]
RHO_REL, MARGIN_REL = 0.15, 0.05      # GT 탐색 반경 규칙 + 여유 마진 (명세 1.3)


class Stage2Dataset(Stage1Dataset):
    def __init__(self, *args, hm_target="onehot", gauss_sigma_rel=0.04, **kw):
        super().__init__(*args, **kw)
        self.hm_target = hm_target
        self.gauss_sigma_rel = gauss_sigma_rel

        # 파지별 히트맵 과제 목록: (unit_group[6], n_groups, anchor_pos[n,3])
        rec_dir = Path(kw.get("records_dir",
                              ROOT / "data/contraction_gt_300k/search_records"))
        gid_set = {g for g, _ in self.samples}
        self.hm_tasks = defaultdict(list)
        for p in sorted(rec_dir.glob("shard_*.npz")):
            with np.load(p, allow_pickle=True) as z:
                for g, n, ps, pr, ap in zip(z["grasp_id"], z["n_anchors"],
                                            z["partition"], z["pad_rel"],
                                            z["anchor_pos"]):
                    if g not in gid_set:
                        continue
                    n = int(n)
                    if n == 2 and abs(pr - 0.10) > 1e-6:
                        continue                       # 팁 반경 중복 제거
                    ug = np.full(N_UNIT, -1, np.int8)
                    for gi, blk in enumerate(json.loads(ps)):
                        for u in blk:
                            ug[UNITS.index(u)] = gi
                    self.hm_tasks[g].append((ug, n, ap[:n].astype(np.float32)))
        self.samples = [s for s in self.samples if len(self.hm_tasks[s[0]])]

        # 접촉 캐시에 링크 필드 추가 로드 (그룹 중심·퍼짐 계산용)
        base_load = self._contacts.loader_fn
        def load_more(rel):
            d = base_load(rel)
            with np.load(self.contact_root / rel.replace(".npy", ".npz")) as z:
                d["link_contact_point"] = z["link_contact_point"].copy()
                d["link_contact"] = z["link_contact"].copy()
                d["link_area"] = z["link_contact_area"].copy()
            return d
        self._contacts.loader_fn = load_more
        self._contacts._store.clear()

    def __getitem__(self, i):
        out = super().__getitem__(i)
        gid, e = self.samples[i]
        rel, ridx = gid.rsplit("#", 1)
        ridx = int(ridx)
        r_obj, ts = e["r_obj"], e["total_scale"]
        obj = self._objects.get(self.obj_of[gid])
        con = self._contacts.get(rel)
        pts_m = obj["pts"] * ts                                    # [2048,3] 미터

        # 점 로컬 특징 + 소속 패치 (브리지 ③·①)
        pid = obj["point_patch_id"] if "point_patch_id" in obj else None
        # _load_object 는 pid 를 저장 안 함 → 캐시에 추가
        if "pid" not in obj:
            with np.load(Path(ROOT / "data/contraction_net" /
                              (f"patch_cache_{self.n_patch}" if self.n_patch != 128
                               else "patch_cache")) /
                         self.obj_of[gid][0] / f"{self.obj_of[gid][1]}.npz") as z:
                obj["pid"] = z["point_patch_id"].copy()
        pid = obj["pid"]
        scale = ts / r_obj
        point_feat = np.concatenate(
            [(obj["pts"] - obj["centers"][pid]) * scale, obj["nrm"]], 1).astype(np.float32)

        # 파티션(레코드) 추첨 — 학습은 무작위, 평가는 결정적
        tasks = self.hm_tasks[gid]
        rng = np.random.default_rng(None if self.augment else self.seed * 31 + i)
        ug, n_g, anchors = tasks[int(rng.integers(len(tasks)))]

        # 그룹 중심·반경 (참값 접촉에서 — GT 탐색과 같은 식) + 마스크 + 정답 인덱스
        lp = con["link_contact_point"][ridx]
        lc = con["link_contact"][ridx]
        la = con["link_area"][ridx]
        hm_mask = np.zeros((4, len(pts_m)), bool)
        hm_target = np.full(4, -1, np.int64)
        soft = np.zeros((4, len(pts_m)), np.float32) if self.hm_target == "gauss" else None
        for g in range(n_g):
            members = [u for u in range(N_UNIT) if ug[u] == g]
            li = [j for j in range(len(_LINK_NAMES))
                  if _LINK_UNIT[j] in members and lc[j]]
            if li:
                w = la[li] / max(la[li].sum(), 1e-12)
                center = (lp[li] * w[:, None]).sum(0)
                spread = float(np.linalg.norm(lp[li] - center, axis=1).max())
            else:                                   # 폴백: 앵커 자체 중심
                center, spread = anchors[g], 0.0
            radius = max(spread, RHO_REL * r_obj) + MARGIN_REL * r_obj
            d = np.linalg.norm(pts_m - center, axis=1)
            hm_mask[g] = d <= radius
            t = int(np.linalg.norm(pts_m - anchors[g], axis=1).argmin())
            hm_mask[g, t] = True                    # 정답은 항상 후보에 포함
            hm_target[g] = t
            if soft is not None:
                dd = np.linalg.norm(pts_m - pts_m[t], axis=1)
                s = np.exp(-dd ** 2 / (2 * (self.gauss_sigma_rel * r_obj) ** 2))
                s[~hm_mask[g]] = 0.0
                soft[g] = s / max(s.sum(), 1e-12)

        out["hm_unit_group"] = torch.from_numpy(ug.astype(np.int64))
        out["hm_n_groups"] = torch.tensor(n_g, dtype=torch.long)
        out["hm_target"] = torch.from_numpy(hm_target)
        out["hm_mask"] = torch.from_numpy(hm_mask)
        out["point_feat"] = torch.from_numpy(point_feat)
        out["point_patch"] = torch.from_numpy(pid.astype(np.int64))
        if soft is not None:
            out["hm_soft"] = torch.from_numpy(soft)
        return out
