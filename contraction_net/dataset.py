"""Contraction GT 데이터셋 로더 (dexonomy 단독).

레코드(search_records/shard_*.npz) 1개 = 파지 1개 × 파티션 1개 × 패드반경 1개.
grasp_id 로 세 소스를 join 한다 (핸드오프 문서 §4):
  1. 접촉 테이블  data/dexonomy/contact_cache/<상대경로>.npz  → 조건 입력 (유닛 6개)
  2. 물체 캐시    <object_cache_root>/<dataset>/<물체ID>.npz   → 표면점 2048 + 안쪽 법선
  3. (선택) 원본 qpos  succ_collect/<상대경로> 의 grasp_qpos[i] (29차원)

좌표·부호 규약 (전부 파지 좌표계, 미터):
  - 물체는 원점, 회전 없음. 단위 메쉬 × total_scale = 실제 크기.
  - 법선은 출력 전부 "물체 안쪽" 방향으로 통일한다. 접촉 테이블 원본만
    바깥 방향이므로 여기서 부호를 반전한다 (함정 목록 1항).
  - NaN 패딩(anchor/prior_dist)은 0 + anchor_mask 로 변환해 내보낸다.

수록 임계·등급은 미확정이라 전부 생성자 파라미터다 (지침 3·5항).
"""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

UNITS = ["index", "middle", "ring", "little", "thumb", "palm"]  # contact_cache/meta.json 순서
MAX_ANCHORS = 4
ALPHA_DIRS = 14


class _FileCache:
    """파일 단위 FIFO 캐시 (DataLoader worker 별로 독립)."""

    def __init__(self, loader_fn, max_entries):
        self.loader_fn = loader_fn
        self.max_entries = max_entries
        self._store = OrderedDict()

    def get(self, key):
        if key not in self._store:
            self._store[key] = self.loader_fn(key)
            if len(self._store) > self.max_entries:
                self._store.popitem(last=False)
        return self._store[key]


class ContractionGTDataset(Dataset):
    """search_records shard 들을 메모리에 올리고, 접촉·물체·qpos 를 lazy join 한다.

    Args:
        records_dir: shard_*.npz 가 있는 폴더 (예: data/contraction_gt_dev/search_records)
        manifest_path: 파지 목록 파일. None 이면 records_dir 부모의 pilot_manifest.tsv
        contact_root: 접촉 테이블 루트 (data/dexonomy/contact_cache)
        object_cache_root: 물체 캐시 루트 (data/contraction_gt_pilot/object_cache)
        succ_root: 원본 파지 루트. load_qpos=True 일 때만 필요
        min_alpha: 이 값 미만 레코드 제외 (None = 필터 없음; 수록 임계는 미확정 — 지침 5항)
        n_anchors: 예 (2,3,4). None = 전부
        pad_rel: 허용 pad_rel 집합. None = 전부
        n2_trust_alpha: N=2 신뢰 등급 임계 (지침 3항, 기본 0.05)
        gate_records: 도장 결과 npz 경로. None 이면 records_dir 부모의
            gate_records.npz 를 자동 탐지 (없으면 도장 없이 동작)
        split / val_fraction: 물체ID 해시로 train/val 분리 (같은 물체는 항상 같은 쪽)
        n_points: 물체 표면점 수 (2048 이하, 앞에서 자름이 아니라 균일 부분샘플)
        load_qpos: True 면 qpos [29] 포함
    """

    def __init__(
        self,
        records_dir,
        manifest_path=None,
        contact_root="data/dexonomy/contact_cache",
        object_cache_root="data/contraction_gt_pilot/object_cache",
        succ_root="data/dexonomy/Dexonomy_GRASP_shadow/succ_collect",
        min_alpha=None,
        n_anchors=None,
        pad_rel=None,
        n2_trust_alpha=0.05,
        gate_records=None,
        split=None,
        val_fraction=0.05,
        n_points=2048,
        load_qpos=False,
        cache_contacts=4096,
        cache_objects=8192,
        seed=0,
    ):
        records_dir = Path(records_dir)
        self.contact_root = Path(contact_root)
        self.object_cache_root = Path(object_cache_root)
        self.succ_root = Path(succ_root)
        self.n_points = n_points
        self.load_qpos = load_qpos
        self.n2_trust_alpha = n2_trust_alpha
        self.seed = seed

        # ── shard 병합 (레코드 배열만; shard 메타 스칼라 n_grasp/n_fail/times 제외) ──
        shards = sorted(records_dir.glob("shard_*.npz"))
        if not shards:
            raise FileNotFoundError(f"{records_dir} 에 shard_*.npz 없음")
        cols = {}
        for p in shards:
            with np.load(p, allow_pickle=True) as z:
                for k in z.files:
                    if z[k].ndim == 0:
                        continue
                    cols.setdefault(k, []).append(z[k])
        self.rec = {k: np.concatenate(v) for k, v in cols.items()}
        n_total = len(self.rec["grasp_id"])

        # ── 필터 (전부 파라미터 — 임계 미확정) ──
        keep = np.ones(n_total, dtype=bool)
        if min_alpha is not None:
            keep &= self.rec["min_alpha"] >= min_alpha
        if n_anchors is not None:
            keep &= np.isin(self.rec["n_anchors"], list(n_anchors))
        if pad_rel is not None:
            ok = np.zeros(n_total, dtype=bool)
            for v in pad_rel:
                ok |= np.isclose(self.rec["pad_rel"], v)
            keep &= ok

        # ── 파지 목록 파일로 grasp_id → (dataset, 물체ID) join ──
        if manifest_path is None:
            manifest_path = records_dir.parent / "pilot_manifest.tsv"
        self.obj_of = {}
        for line in Path(manifest_path).read_text().splitlines()[1:]:
            _t, ds, oid, rel, ridx, _s, _n = line.split("\t")
            self.obj_of[f"{rel}#{ridx}"] = (ds, oid)
        missing = [g for g in np.unique(self.rec["grasp_id"][keep])
                   if g not in self.obj_of]
        if missing:
            raise KeyError(f"파지 목록에 없는 grasp_id {len(missing)}개 (예: {missing[0]})")

        # ── train/val 분리: 물체ID 해시 (레코드·파지가 아니라 물체 단위 → 누수 방지) ──
        if split is not None:
            assert split in ("train", "val")
            def is_val(gid):
                oid = self.obj_of[gid][1]
                h = int(hashlib.md5(f"{seed}:{oid}".encode()).hexdigest()[:8], 16)
                return (h % 10_000) < val_fraction * 10_000
            val_mask = np.array([is_val(g) for g in self.rec["grasp_id"]])
            keep &= val_mask if split == "val" else ~val_mask

        self.idx = np.nonzero(keep)[0]

        # ── gate verdict join ──
        # gate_records.npz 의 idx 는 "search_records 를 사전순 glob 으로 이어붙인
        # 행 번호" (gate_runner._load_records 와 이 로더가 같은 정렬을 씀).
        # 파일 구성이 달라졌으면 어긋나므로 grasp_id·n_anchors 전수 대조로 검증한다.
        if gate_records is None:
            default = records_dir.parent / "gate_records.npz"
            gate_records = default if default.exists() else None
        self.gate_verdict = None
        if gate_records is not None and Path(gate_records).exists():
            with np.load(gate_records, allow_pickle=True) as z:
                gi = z["idx"].astype(int)
                if not (np.array_equal(self.rec["grasp_id"][gi], z["grasp_id"])
                        and np.array_equal(self.rec["n_anchors"][gi], z["n_anchors"])):
                    raise ValueError(
                        f"{gate_records}: idx 가 현재 조각 파일 순서와 안 맞음 — "
                        "search_records 구성이 도장 시점과 달라진 것")
                self.gate_verdict = np.full(n_total, "", dtype=object)
                self.gate_verdict[gi] = z["verdict"]

        # ── taxonomy 타입 사전 ──
        self.type_vocab = sorted(set(self.rec["type"].tolist()))
        self._type_idx = {t: i for i, t in enumerate(self.type_vocab)}

        self._contacts = _FileCache(self._load_contact_file, cache_contacts)
        self._objects = _FileCache(self._load_object_file, cache_objects)
        self._qpos_files = _FileCache(self._load_qpos_file, cache_contacts)

    # ── 파일 로더들 ─────────────────────────────────────────────────────────
    def _load_contact_file(self, rel):
        with np.load(self.contact_root / rel.replace(".npy", ".npz")) as z:
            return {k: z[k].copy() for k in
                    ("finger_patch_centroid", "finger_contact_normal",
                     "finger_contact", "finger_contact_area")}

    def _load_object_file(self, ds_oid):
        ds, oid = ds_oid
        with np.load(self.object_cache_root / ds / f"{oid}.npz") as z:
            return {"points": z["points"].copy(), "normals": z["normals"].copy()}

    def _load_qpos_file(self, rel):
        d = np.load(self.succ_root / rel, allow_pickle=True).item()
        return d["grasp_qpos"].copy()

    # ── Dataset 프로토콜 ────────────────────────────────────────────────────
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.idx[i]
        rec = self.rec
        gid = rec["grasp_id"][r]
        rel, ridx = gid.rsplit("#", 1)
        ridx = int(ridx)
        n = int(rec["n_anchors"][r])
        total_scale = float(rec["total_scale"][r])

        # 물체: 단위 메쉬 표면점 × total_scale, 법선은 이미 안쪽
        obj = self._objects.get(self.obj_of[gid])
        pts, nrm = obj["points"], obj["normals"]
        if self.n_points < len(pts):
            sel = np.random.default_rng(self.seed * 1_000_003 + r).choice(
                len(pts), self.n_points, replace=False)
            pts, nrm = pts[sel], nrm[sel]
        obj_points = torch.from_numpy(pts * total_scale).float()
        obj_normals = torch.from_numpy(nrm.copy()).float()

        # 접촉 조건: 유닛 6개. 법선은 원본이 바깥 방향 → 안쪽으로 반전
        con = self._contacts.get(rel)
        contact_pos = torch.from_numpy(con["finger_patch_centroid"][ridx].copy()).float()
        contact_normal = torch.from_numpy(-con["finger_contact_normal"][ridx]).float()
        contact_mask = torch.from_numpy(con["finger_contact"][ridx].copy())
        contact_area = torch.from_numpy(con["finger_contact_area"][ridx].copy()).float()

        # 파티션 → 유닛별 그룹 번호 [6] (그룹 g 의 유닛들 → 앵커 g; 없으면 -1)
        partition = json.loads(rec["partition"][r])
        unit_group = torch.full((len(UNITS),), -1, dtype=torch.long)
        for g, block in enumerate(partition):
            for u in block:
                unit_group[UNITS.index(u)] = g

        # 라벨: NaN 패딩 → 0 + 마스크
        anchor_pos = rec["anchor_pos"][r].copy()
        anchor_normal = rec["anchor_normal"][r].copy()
        prior_dist = rec["prior_dist"][r].copy()
        anchor_mask = torch.zeros(MAX_ANCHORS, dtype=torch.bool)
        anchor_mask[:n] = True
        anchor_pos[n:] = 0.0
        anchor_normal[n:] = 0.0
        prior_dist[n:] = 0.0

        # 등급 (지침 3항): 1 = 물리 신뢰(N=2 & α 임계 이상, 또는 gate pass),
        #                  0 = 탐색만 됨 (도장 안 된 레코드 포함).
        verdict = self.gate_verdict[r] if self.gate_verdict is not None else ""
        if verdict:
            trusted = verdict == "pass"
        else:
            trusted = n == 2 and rec["min_alpha"][r] > self.n2_trust_alpha
        out = {
            "obj_points": obj_points,
            "obj_normals": obj_normals,
            "contact_pos": contact_pos,
            "contact_normal": contact_normal,
            "contact_mask": contact_mask,
            "contact_area": contact_area,
            "unit_group": unit_group,
            "anchor_pos": torch.from_numpy(anchor_pos).float(),
            "anchor_normal": torch.from_numpy(anchor_normal).float(),
            "anchor_mask": anchor_mask,
            "n_anchors": torch.tensor(n, dtype=torch.long),
            "alpha": torch.from_numpy(rec["alpha"][r].copy()).float(),
            "alpha_mu03": torch.from_numpy(rec["alpha_mu03"][r].copy()).float(),
            "alpha_mu10": torch.from_numpy(rec["alpha_mu10"][r].copy()).float(),
            "min_alpha": torch.tensor(rec["min_alpha"][r], dtype=torch.float32),
            "weak_dir": torch.tensor(rec["weak_dir"][r], dtype=torch.long),
            "prior_dist": torch.from_numpy(prior_dist).float(),
            "pad_rel": torch.tensor(rec["pad_rel"][r], dtype=torch.float32),
            "r_obj": torch.tensor(rec["r_obj"][r], dtype=torch.float32),
            "total_scale": torch.tensor(total_scale, dtype=torch.float32),
            "thumbless": torch.tensor(bool(rec["thumbless"][r])),
            "trusted": torch.tensor(bool(trusted)),
            "type_idx": torch.tensor(self._type_idx[rec["type"][r]], dtype=torch.long),
            "grasp_id": gid,
        }
        if self.load_qpos:
            out["qpos"] = torch.from_numpy(
                self._qpos_files.get(rel)[ridx].copy()).float()
        return out
