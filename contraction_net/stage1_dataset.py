"""Stage 1 학습 데이터셋 — 샘플 1개 = 파지 1개 (명세: md_claude/plans/stage1_topology_net.md).

출력 (파지당):
  patch_feat   [128, 7]        패치 중심(/r_obj)·안쪽 법선·곡률
  patch_local  [128, K, 6]     패치 소속 표면점 (중심 상대/r_obj, 법선), K=16
  patch_local_mask [128, K]
  unit_feat    [6, 10]         접촉 위치(투영/r_obj) 3 + 법선 3 + 마디비율 3 + mask 1
  rel_type     [135, 135] int8 관계 유형 (0:패치인접 1:패치비인접 2:부위접촉 3:부위비접촉
                               4:부위-부위 5:CLS 6:self)
  log_robj     [1]
  target_n3/n4 [C] float32     α·pass 비례 soft target (합=1 또는 전부 0=스킵)
  valid_n3/n4  [C] bool        유효 클래스 마스크 (참값 접촉 기준)
  reg_target   [3], reg_mask [3]   N=2/3/4 최고 min-α 의 log1p
"""

import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import _FileCache, UNITS
from .partition_vocab import PartitionVocab, canon

ROOT = Path(__file__).resolve().parents[1]
N_PATCH, N_UNIT, N_NODE = 128, 6, 135
K_LOCAL = 16
REL_PATCH_ADJ, REL_PATCH_NON, REL_UNIT_CON, REL_UNIT_NON = 0, 1, 2, 3
REL_UNIT_UNIT, REL_CLS, REL_SELF = 4, 5, 6

_META = json.loads((ROOT / "data/dexonomy/contact_cache/meta.json").read_text())
_LINK_NAMES, _LINK_UNIT = _META["link_names"], _META["link_finger_id"]


def _phalanx_bin(name):
    if name.endswith("distal"):
        return 0
    if name.endswith("middle"):
        return 1
    return 2


_LINK_BIN = np.array([_phalanx_bin(n) for n in _LINK_NAMES])


class Stage1Dataset(Dataset):
    def __init__(
        self,
        records_dir=ROOT / "data/contraction_gt_300k/search_records",
        contact_root=ROOT / "data/dexonomy/contact_cache",
        object_cache_root=ROOT / "data/contraction_gt_pilot/object_cache",
        patch_cache_root=ROOT / "data/contraction_net/patch_cache",
        patch_contacts_dir=ROOT / "data/contraction_net/patch_contacts",
        vocab_path=None,
        manifest_path=None,
        split=None,
        val_fraction=0.05,
        augment=False,
        sigma_max=0.08,
        contact_geom=True,     # ablation 스위치: 접촉 위치·법선 특징
        n_patch=N_PATCH,       # 패치 해상도 실험용 (캐시·접촉 경로와 일치해야 함)
        com_feat=False,        # 무게중심 상대 특징 (패치 7→11차원) — 2026-08-02 카드
        feat_norm=False,       # 특징 표준화 (평균0·분산1, feat_stats.json) — 2026-08-04 카드
        seed=0,
    ):
        import hashlib
        records_dir = Path(records_dir)
        self.contact_root = Path(contact_root)
        self.augment = augment
        self.sigma_max = sigma_max
        self.sigma_fixed = None      # 평가용: 노이즈 곡선에서 σ 고정
        self.n_patch = n_patch
        self.n_node = n_patch + N_UNIT + 1
        self.com_feat = com_feat
        self.fstats = None
        if feat_norm:
            st = json.loads((ROOT / "data/contraction_net/feat_stats.json").read_text())
            self.fstats = {k: (np.array(v["mean"], np.float32),
                               np.array(v["std"], np.float32)) for k, v in st.items()}
        self.contact_geom = contact_geom
        self.seed = seed
        self.vocab = PartitionVocab.load(vocab_path) if vocab_path else PartitionVocab.load()
        C = len(self.vocab)

        # ── 레코드 (필요 열만) + 시뮬 판정 ──
        cols = {k: [] for k in ["grasp_id", "n_anchors", "partition", "pad_rel",
                                "min_alpha", "r_obj", "total_scale", "thumbless"]}
        for p in sorted(records_dir.glob("shard_*.npz")):
            with np.load(p, allow_pickle=True) as z:
                for k in cols:
                    cols[k].append(z[k])
        rec = {k: np.concatenate(v) for k, v in cols.items()}
        n_total = len(rec["grasp_id"])
        verdict = np.full(n_total, "", dtype=object)
        # 판정 파일의 idx 는 원본 search_records 의 행 번호. 다른 레코드 폴더(예: search_records_multi)에는
        # (파지, 파티션, 팁 반경) 키로 옮겨 붙인다 (2026-10-24).
        gate_dir = records_dir.parent
        base_dir = gate_dir / "search_records"
        remap = None
        if records_dir.resolve() != base_dir.resolve() and base_dir.exists():
            bcols = {k: [] for k in ["grasp_id", "partition", "pad_rel"]}
            for p in sorted(base_dir.glob("shard_*.npz")):
                with np.load(p, allow_pickle=True) as z:
                    for k in bcols:
                        bcols[k].append(z[k])
            brec = {k: np.concatenate(v) for k, v in bcols.items()}
            key = lambda r, i: (str(r["grasp_id"][i]), str(r["partition"][i]), round(float(r["pad_rel"][i]), 4))
            remap = {key(rec, i): i for i in range(n_total)}
        for name in ("gate_records.npz", "gate_records_all.npz"):
            gp = gate_dir / name
            if gp.exists():
                with np.load(gp, allow_pickle=True) as z:
                    gi = z["idx"].astype(int)
                    if remap is None:
                        assert np.array_equal(rec["grasp_id"][gi], z["grasp_id"]), name
                        verdict[gi] = z["verdict"]
                    else:
                        assert np.array_equal(brec["grasp_id"][gi], z["grasp_id"]), name
                        for bi, v in zip(gi, z["verdict"]):
                            j = remap.get(key(brec, bi))
                            if j is not None:
                                verdict[j] = v

        # ── 파지 목록 파일: grasp -> (dataset, 물체ID) ──
        if manifest_path is None:
            manifest_path = records_dir.parent / "pilot_manifest.tsv"
        self.obj_of = {}
        for line in Path(manifest_path).read_text().splitlines()[1:]:
            _t, ds, oid, rel, ridx, _s, _n = line.split("\t")
            self.obj_of[f"{rel}#{ridx}"] = (ds, oid)

        # ── 파지 단위 target 사전 계산 (한 패스) ──
        by_g = defaultdict(list)
        for i in range(n_total):
            by_g[rec["grasp_id"][i]].append(i)
        self.samples = []      # (gid, meta dict)
        for gid, idxs in by_g.items():
            r_obj = float(rec["r_obj"][idxs[0]])
            ts = float(rec["total_scale"][idxs[0]])
            tgt = {3: {}, 4: {}}
            valid = {3: set(), 4: set()}
            pas = {3: {}, 4: {}}     # 합격 확률 헤드 라벨: 1=pass, 0=fail/침하/α≤0.02, -1=판정불가
            reg = np.full(3, np.nan)
            for i in idxs:
                n = int(rec["n_anchors"][i])
                a = float(rec["min_alpha"][i])
                if n == 2:
                    if abs(rec["pad_rel"][i] - 0.10) < 1e-6:
                        reg[0] = max(np.nan_to_num(reg[0]), a) if not np.isnan(reg[0]) else a
                    continue
                cid = self.vocab.id_of(json.loads(rec["partition"][i]))
                valid[n].add(cid)
                v = verdict[i]
                w = a if (v in ("pass", "unstable", "")) else 0.0
                tgt[n][cid] = w
                pas[n][cid] = 1.0 if v == "pass" else (-1.0 if v == "unstable" else 0.0)
                k = n - 2
                reg[k] = a if np.isnan(reg[k]) else max(reg[k], a)
            entry = {"r_obj": r_obj, "total_scale": ts,
                     "reg": np.log1p(np.nan_to_num(reg, nan=0.0)).astype(np.float32),
                     "reg_mask": ~np.isnan(reg)}
            for n in (3, 4):
                ids = np.array(sorted(valid[n]), dtype=np.int64)
                w = np.array([tgt[n].get(c, 0.0) for c in ids], dtype=np.float64)
                s = w.sum()
                entry[f"ids{n}"] = ids
                entry[f"p{n}"] = (w / s).astype(np.float32) if s > 0 else np.zeros(len(ids), np.float32)
                entry[f"pass{n}"] = np.array([pas[n].get(c, -1.0) for c in ids], np.float32)
            self.samples.append((gid, entry))
        self.samples.sort()

        # ── 분리 (물체 단위 해시 — 레코드 로더와 같은 규칙) ──
        if split is not None:
            assert split in ("train", "val")
            def is_val(gid):
                oid = self.obj_of[gid][1]
                h = int(hashlib.md5(f"{seed}:{oid}".encode()).hexdigest()[:8], 16)
                return (h % 10_000) < val_fraction * 10_000
            self.samples = [s for s in self.samples
                            if is_val(s[0]) == (split == "val")]

        # ── 실측 패치 접촉: grasp -> (unit, patch, area) ──
        self.patch_con = {}
        for p in sorted(Path(patch_contacts_dir).glob("part_*.npz")):
            with np.load(p, allow_pickle=True) as z:
                g, u, pa = z["grasp_id"], z["unit"], z["patch"]
                order = np.argsort(g, kind="stable")
                g, u, pa = g[order], u[order], pa[order]
                cut = np.nonzero(np.r_[True, g[1:] != g[:-1]])[0]
                for a, b in zip(cut, np.r_[cut[1:], len(g)]):
                    self.patch_con[g[a]] = (u[a:b].copy(), pa[a:b].copy())

        # 캐시 상한을 보수적으로 — 워커 수 × 캐시가 스왑을 밀어내지 않게
        self._objects = _FileCache(
            lambda k: self._load_object(object_cache_root, patch_cache_root, k), 1536)
        self._contacts = _FileCache(self._load_contact, 1536)

    # ── 파일 로더 ─────────────────────────────────────────────────────────
    def _load_object(self, obj_root, patch_root, ds_oid):
        ds, oid = ds_oid
        with np.load(Path(obj_root) / ds / f"{oid}.npz") as z:
            pts, nrm = z["points"].astype(np.float32), z["normals"].astype(np.float32)
            com = z["com"].astype(np.float32)
        with np.load(Path(patch_root) / ds / f"{oid}.npz") as z:
            centers, cn = z["centers"], z["normals"]
            curv, pid = z["curvature"], z["point_patch_id"]
            es, ed = z["edge_src"], z["edge_dst"]
        rng = np.random.default_rng(0)
        local = np.zeros((self.n_patch, K_LOCAL, 6), np.float32)
        lmask = np.zeros((self.n_patch, K_LOCAL), bool)
        for p in range(self.n_patch):
            m = np.nonzero(pid == p)[0]
            if len(m) > K_LOCAL:
                m = rng.choice(m, K_LOCAL, replace=False)
            local[p, :len(m), :3] = pts[m] - centers[p]
            local[p, :len(m), 3:] = nrm[m]
            lmask[p, :len(m)] = True
        return dict(pts=pts, nrm=nrm, com=com, centers=centers, cnormals=cn, curv=curv,
                    edge_src=es, edge_dst=ed, local=local, lmask=lmask)

    def _load_contact(self, rel):
        with np.load(self.contact_root / rel.replace(".npy", ".npz")) as z:
            return {k: z[k].copy() for k in
                    ("finger_patch_centroid", "finger_contact", "link_contact_area")}

    # ── Dataset ───────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        gid, e = self.samples[i]
        rel, ridx = gid.rsplit("#", 1)
        ridx = int(ridx)
        r_obj, ts = e["r_obj"], e["total_scale"]
        obj = self._objects.get(self.obj_of[gid])
        con = self._contacts.get(rel)
        rng = np.random.default_rng(None if self.augment else self.seed * 7919 + i)
        if not self.augment:
            sigma = 0.0
        elif self.sigma_fixed is not None:
            sigma = self.sigma_fixed
        else:
            sigma = rng.uniform(0, self.sigma_max)

        pts_m = obj["pts"] * ts                       # 표면점 (미터)
        scale = ts / r_obj

        # 패치 노드
        parts_ = [obj["centers"] * scale, obj["cnormals"], obj["curv"][:, None]]
        if self.com_feat:
            crel = (obj["centers"] - obj["com"]) * scale     # 무게중심 상대 (LP 기준계)
            parts_ += [crel, np.linalg.norm(crel, axis=1, keepdims=True)]
        patch_feat = np.concatenate(parts_, 1).astype(np.float32)
        local = obj["local"].copy()
        local[:, :, :3] *= scale

        # 접촉 부위: 마디 비율 + 위치·법선 (표면 투영)
        cen = con["finger_patch_centroid"][ridx].astype(np.float64)   # [6,3] 미터
        mask = con["finger_contact"][ridx].copy()
        la = con["link_contact_area"][ridx]
        ratios = np.zeros((N_UNIT, 3), np.float32)
        for li in range(len(_LINK_NAMES)):
            ratios[_LINK_UNIT[li], _LINK_BIN[li]] += la[li]
        s = ratios.sum(1, keepdims=True)
        ratios = np.divide(ratios, s, out=np.zeros_like(ratios), where=s > 0)

        # 원천 교란 (전역 SE3 + 접선 jitter) → 표면 재투영
        delta = np.zeros((N_UNIT, 3))
        if sigma > 0:
            ang = rng.normal(0, sigma * 0.5)
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
            t = rng.normal(0, sigma * 0.5 * r_obj, 3)
            cen_new = cen @ R.T + t
            cen_new += rng.normal(0, sigma * r_obj, (N_UNIT, 3))
            delta = cen_new - cen
            cen = cen_new
        d2 = ((pts_m[None] - cen[:, None]) ** 2).sum(-1)              # [6,2048]
        nn = d2.argmin(1)
        proj = pts_m[nn]
        pnrm = obj["nrm"][nn]

        unit_feat = np.zeros((N_UNIT, 10), np.float32)
        if self.contact_geom:
            unit_feat[:, :3] = proj / r_obj
            unit_feat[:, 3:6] = pnrm
        unit_feat[:, 6:9] = ratios
        unit_feat[:, 9] = mask

        if sigma > 0:
            ratios_n = np.clip(ratios + rng.normal(0, sigma * 0.3, ratios.shape), 0, None)
            s = ratios_n.sum(1, keepdims=True)
            unit_feat[:, 6:9] = np.divide(ratios_n, s, out=np.zeros_like(ratios_n), where=s > 0)

        # 관계 유형 행렬
        NP, NN = self.n_patch, self.n_node
        rel_t = np.full((NN, NN), REL_PATCH_NON, np.int8)
        rel_t[NP:NP + N_UNIT, :NP] = REL_UNIT_NON
        rel_t[:NP, NP:NP + N_UNIT] = REL_UNIT_NON
        rel_t[NP:NP + N_UNIT, NP:NP + N_UNIT] = REL_UNIT_UNIT
        rel_t[NN - 1, :] = REL_CLS
        rel_t[:, NN - 1] = REL_CLS
        rel_t[obj["edge_src"], obj["edge_dst"]] = REL_PATCH_ADJ
        rel_t[obj["edge_dst"], obj["edge_src"]] = REL_PATCH_ADJ
        pu, pp = self.patch_con.get(gid, (np.array([], np.int8), np.array([], np.int16)))
        if sigma > 0 and len(pu):
            # 엣지 재유도: 깨끗한 접촉 패치 집합을 부위별 교란 벡터만큼 평행이동
            centers_m = obj["centers"] * ts
            moved = centers_m[pp] + delta[pu.astype(int)]
            pp = ((centers_m[None] - moved[:, None]) ** 2).sum(-1).argmin(1).astype(np.int16)
        for u, p in zip(pu, pp):
            rel_t[NP + int(u), int(p)] = REL_UNIT_CON
            rel_t[int(p), NP + int(u)] = REL_UNIT_CON

        # 접촉 누락 재현: 낮은 확률로 최소 면적 접촉 부위 특징 끄기 (손실 마스크는 참값)
        if sigma > 0 and rng.random() < sigma * 0.3 and mask.sum() > 2:
            areas = np.array([la[np.array(_LINK_UNIT) == u].sum() for u in range(N_UNIT)])
            areas[~mask.astype(bool)] = np.inf
            u = int(areas.argmin())
            unit_feat[u] = 0.0
            rel_t[NP + u, :NP] = REL_UNIT_NON
            rel_t[:NP, NP + u] = REL_UNIT_NON
        np.fill_diagonal(rel_t, REL_SELF)

        if self.fstats is not None:      # 표준화 (증강 이후·반환 직전)
            m, sd = self.fstats["patch_feat"]
            patch_feat = (patch_feat - m) / sd
            m, sd = self.fstats["patch_local"]
            local = (local - m) / sd
            m, sd = self.fstats["unit_feat"]
            unit_feat = (unit_feat - m) / sd
        C = len(self.vocab)
        out = {
            "patch_feat": torch.from_numpy(patch_feat),
            "patch_local": torch.from_numpy(local),
            "patch_local_mask": torch.from_numpy(obj["lmask"]),
            "unit_feat": torch.from_numpy(unit_feat),
            "rel_type": torch.from_numpy(rel_t),   # int8 — 전송량 절감, GPU 에서 변환
            "log_robj": torch.tensor([np.log(r_obj / 0.1)], dtype=torch.float32),
            "reg_target": torch.from_numpy(e["reg"]),
            "reg_mask": torch.from_numpy(e["reg_mask"]),
            "grasp_id": gid,
        }
        for n in (3, 4):
            t = torch.zeros(C)
            v = torch.zeros(C, dtype=torch.bool)
            pt = torch.full((C,), -1.0)
            ids = e[f"ids{n}"]
            if len(ids):
                v[ids] = True
                t[ids] = torch.from_numpy(e[f"p{n}"])
                pt[ids] = torch.from_numpy(e[f"pass{n}"])
            out[f"target_n{n}"] = t
            out[f"valid_n{n}"] = v
            out[f"pass_n{n}"] = pt      # 1/0 = 합격/불합격, -1 = 라벨 없음
        return out
