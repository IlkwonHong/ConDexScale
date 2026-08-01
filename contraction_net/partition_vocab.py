"""파티션 클래스 사전 — Stage 1 분류의 출력 공간 (N≥3, 그릴링 질문 1·3 확정).

클래스 = 파티션의 정규형: 블록(부위 이름 정렬 tuple)들을 정렬한 tuple.
예: (("index", "middle"), ("little", "palm", "ring"), ("thumb",))

사전은 1차분 관측치로 구축하되, 열거 규칙(contraction_gt.units)과의 일치를 검증한다.
추론 시 유효 클래스는 접촉 부위 집합 + 고정 앵커 부위에서 같은 규칙으로 열거한다.

빌드: ./venv/bin/python -m contraction_net.partition_vocab   → data/contraction_net/partition_vocab.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = ROOT / "data" / "contraction_net" / "partition_vocab.json"

sys.path.insert(0, str(ROOT))
from contraction_gt.units import _adjacency, _connected, _set_partitions  # noqa: E402


def canon(partition):
    """파티션(블록 리스트) -> 정규형 tuple."""
    return tuple(sorted(tuple(sorted(b)) for b in partition))


def canon_key(partition):
    """정규형 -> json 키 문자열."""
    return "|".join(",".join(b) for b in canon(partition))


def enumerate_valid(units_present, anchor0, n_anchors):
    """접촉 부위 집합 + 고정 앵커 부위 -> 유효 파티션(정규형) 목록.

    contraction_gt.units.enumerate_partitions 와 같은 규칙 (앵커0 단독 고정,
    나머지를 인접 그래프 위의 연결 그룹 n-1개로 분할).
    """
    units = list(units_present)
    if len(units) < n_anchors or anchor0 not in units:
        return []
    rest = [u for u in units if u != anchor0]
    adj = _adjacency(rest)
    out, seen = [], set()
    for part in _set_partitions(rest, n_anchors - 1):
        if all(_connected(b, adj) for b in part):
            key = frozenset(frozenset(b) for b in part)
            if key not in seen:
                seen.add(key)
                out.append(canon([[anchor0]] + [list(b) for b in part]))
    return out


class PartitionVocab:
    def __init__(self, classes):
        self.classes = [tuple(tuple(b) for b in c) for c in classes]
        self.index = {c: i for i, c in enumerate(self.classes)}
        self.n_of = np.array([len(c) for c in self.classes])   # 클래스별 앵커 수

    def __len__(self):
        return len(self.classes)

    def id_of(self, partition):
        return self.index[canon(partition)]

    @classmethod
    def load(cls, path=VOCAB_PATH):
        data = json.loads(Path(path).read_text())
        return cls([tuple(tuple(b) for b in c) for c in data["classes"]])

    def save(self, path=VOCAB_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(
            {"classes": [[list(b) for b in c] for c in self.classes]},
            ensure_ascii=False))


def build():
    """1차분 관측 파티션에서 N≥3 클래스 사전 구축 + 열거 규칙 일치 검증."""
    import collections
    rec_dir = ROOT / "data" / "contraction_gt_300k" / "search_records"
    seen = set()
    per_grasp = collections.defaultdict(set)     # 검증용: grasp -> (units, anchor0, N) 별 관측
    obs_by_key = collections.defaultdict(set)
    for p in sorted(rec_dir.glob("shard_*.npz")):
        with np.load(p, allow_pickle=True) as z:
            m = z["n_anchors"] >= 3
            for g, ps in zip(z["grasp_id"][m], z["partition"][m]):
                part = json.loads(ps)
                c = canon(part)
                seen.add(c)
                units = tuple(sorted(u for b in part for u in b))
                anchor0 = part[0][0]
                obs_by_key[(units, anchor0, len(part))].add(c)
    vocab = PartitionVocab(sorted(seen))
    vocab.save()
    print(f"클래스 {len(vocab)}개 (N=3: {(vocab.n_of==3).sum()}, N=4: {(vocab.n_of==4).sum()})")

    # 검증: 관측 == 열거 (접촉 부위 집합·앵커0·N 이 같으면 유효 집합이 같아야 함)
    n_mismatch = 0
    for (units, anchor0, n), obs in obs_by_key.items():
        enum = set(enumerate_valid(list(units), anchor0, n))
        if obs != enum:
            n_mismatch += 1
            if n_mismatch <= 3:
                print(f"  불일치: units={units} anchor0={anchor0} N={n} "
                      f"관측 {len(obs)} vs 열거 {len(enum)}")
    print(f"열거 규칙 일치 검증: 조합 {len(obs_by_key)}개 중 불일치 {n_mismatch}")
    assert n_mismatch == 0
    return vocab


if __name__ == "__main__":
    build()
