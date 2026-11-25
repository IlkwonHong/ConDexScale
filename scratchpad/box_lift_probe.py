"""ARCTIC 박스 들기 키프레임 — 측면-만 앵커 배치의 14방향 α 판별 테스트.

질문: 사람 양손이 측면만 잡은 접촉에서 수축한 앵커가 중력·전도 토크를 버티는가?
방법: 양손 동시 파지+들림 키프레임(00090/00300/00630, 손-물체 거리 스캔으로 선정)의
      손끝 접촉 → 파이프라인 규약(그룹 중심+적응 반경 마스크) 안에서 좌표 상승
      탐색으로 LP min-α 최대화. 네트워크는 안 씀 — "마스크 제약 하 물리 상한" 측정.
좌표 검증: JSON 접촉(물체중심 원점·세계 축) → R@(canonical−centroid) 패치와
      접촉 손 기준 0.2~1.8cm 일치 확인(00450 오른손). 세계 축이므로 중력 = 힘 -z 행.
비교축: N=2/3/4 × 패드 0.10r/0.05r × (마스크 제한 vs 자유 배치) + 6월 기하 앵커.
실행: ./venv/bin/python scratchpad/box_lift_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contraction_gt.surface import sample_surface
from contraction_gt.search import _batched_alpha, mu_tor_from_pad

MU_T = 0.6
KEYFRAMES = [3, 10, 21]            # 파일 인덱스 → 프레임 00090 / 00300 / 00630
TOUCH_CM = 3.0                     # 이보다 멀면 비접촉 손가락으로 제외
RHO_REL, MARGIN_REL = 0.15, 0.05   # stage2 마스크 규약
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIR_NAMES = ["F+x", "F-x", "F+y", "F-y", "F+z", "F-z(중력)",
             "T+x", "T-x", "T+y", "T-y", "T+z", "T-z", "T+약축", "T-약축"]
FINGERS = ["thumb", "index", "middle", "ring", "little"]

_mdir = ROOT / "extlib/dexmachina/dexmachina/assets/arctic/box"
MESH_CANON = trimesh.util.concatenate([
    trimesh.load(_mdir / "bottom_watertight_tiny_box.obj", force="mesh"),
    trimesh.load(_mdir / "top_watertight_tiny_box.obj", force="mesh")])


def load_frame(fidx):
    """키프레임 하나의 세계 축 정렬 표본 + 접촉 손가락 사전 + 6월 앵커."""
    j = json.load(open(ROOT / "data/results_arctic/s01/box_grab_01/Robotiq/mu0.9"
                       / f"{fidx}_Robotiq_0_mu_0.9.json"))
    g = json.load(open(ROOT / "data/arctic_cache/sample_graphs"
                       / f"{j['file_name']}_graph.json"))
    R = np.array(g["obj_rotation"], dtype=np.float64)
    cen = np.array(g["canonical_centroid"], dtype=np.float64)
    V = (R @ (MESH_CANON.vertices - cen).T).T
    mesh_w = trimesh.Trimesh(vertices=V, faces=MESH_CANON.faces, process=False)
    smp = sample_surface(mesh_w, 2048)

    contacts, dropped = {}, []
    for side, key in [("L", "human_finger_contacts_left"), ("R", "human_finger_contacts")]:
        for f in FINGERS:
            c = np.array(j[key][f], dtype=np.float64)
            d = np.linalg.norm(smp.points - c, axis=1).min()
            if d * 100 <= TOUCH_CM:
                contacts[f"{side}_{f}"] = c
            else:
                dropped.append(f"{side}_{f}({d*100:.0f}cm)")
    anchors_jun = np.array(j["human_anchor"], dtype=np.float64)
    print(f"[{j['file_name'][-5:]}] 접촉 {len(contacts)}/10"
          + (f", 제외: {', '.join(dropped)}" if dropped else "")
          + f", 6월 앵커 {len(anchors_jun)}개, r_obj={smp.r_obj:.3f}m")
    return smp, contacts, anchors_jun


def alpha_of(smp, idx, pad_rel):
    mu_tor = mu_tor_from_pad(pad_rel * smp.r_obj, MU_T, 0.5)
    return _batched_alpha(smp, np.asarray(idx), MU_T, np.full(len(idx), mu_tor), DEV)


def coordinate_ascent(smp, masks, init, pad_rel, rounds=2):
    cur = list(init)
    for _ in range(rounds):
        for gi, cand in enumerate(masks):
            combos = np.tile(cur, (len(cand), 1))
            combos[:, gi] = cand
            a = alpha_of(smp, combos, pad_rel).min(1)
            cur[gi] = int(cand[int(np.nanargmax(a))])
    return cur


def masks_for(smp, groups, contacts, free=False):
    masks, init = [], []
    for members in groups:
        pts = np.stack([contacts[m] for m in members])
        center = pts.mean(0)                       # 면적 정보 없음 → 균일 가중
        spread = np.linalg.norm(pts - center, axis=1).max() if len(pts) > 1 else 0.0
        radius = max(spread, RHO_REL * smp.r_obj) + MARGIN_REL * smp.r_obj
        d = np.linalg.norm(smp.points - center, axis=1)
        cand = np.arange(2048) if free else np.where(d <= radius)[0]
        if len(cand) == 0:
            cand = np.argsort(d)[:8]
        masks.append(cand)
        init.append(int(np.argmin(d)))
    return masks, init


def report(tag, smp, idx, pad_rel):
    a = alpha_of(smp, [idx], pad_rel)[0]
    worst = int(np.nanargmin(a))
    print(f"  {tag:36s} min-α={np.nanmin(a):.3f}  중력α={a[5]:.3f}  "
          f"전도α={np.nanmin(a[6:10]):.3f}  약축α={np.nanmin(a[12:14]):.3f}  "
          f"최약={DIR_NAMES[worst]}")
    return a


def partitions_of(contacts):
    """접촉 유효 손가락으로 파티션 구성 (엄지 없으면 그 손 분할 변형 생략)."""
    L = [k for k in contacts if k.startswith("L_")]
    R = [k for k in contacts if k.startswith("R_")]
    parts = {}
    if L and R:
        parts["N=2  [좌손|우손]"] = [L, R]
        if "R_thumb" in R and len(R) > 1:
            parts["N=3a [좌손|우엄지|우나머지]"] = [L, ["R_thumb"],
                                                    [f for f in R if f != "R_thumb"]]
        if "L_thumb" in L and len(L) > 1:
            parts["N=3b [우손|좌엄지|좌나머지]"] = [R, ["L_thumb"],
                                                    [f for f in L if f != "L_thumb"]]
        if "N=3a [좌손|우엄지|우나머지]" in parts and "N=3b [우손|좌엄지|좌나머지]" in parts:
            parts["N=4  [좌엄|좌4|우엄|우4]"] = [["L_thumb"],
                                                  [f for f in L if f != "L_thumb"],
                                                  ["R_thumb"],
                                                  [f for f in R if f != "R_thumb"]]
    return parts


def main():
    for fidx in KEYFRAMES:
        print("=" * 78)
        smp, contacts, anchors_jun = load_frame(fidx)
        parts = partitions_of(contacts)

        print("-- 마스크 제한 탐색 (파이프라인 규약) --")
        for pad in (0.10, 0.05):
            for name, groups in parts.items():
                masks, init = masks_for(smp, groups, contacts)
                best = coordinate_ascent(smp, masks, init, pad)
                report(f"{name} pad={pad:.2f}", smp, best, pad)

        print("-- 자유 배치 탐색 (지지 삽입 상한, pad=0.10) --")
        for name in parts:
            if not (name.startswith("N=3a") or name.startswith("N=4")):
                continue
            groups = parts[name]
            masks_m, init = masks_for(smp, groups, contacts)
            seed = coordinate_ascent(smp, masks_m, init, 0.10)
            masks_f, _ = masks_for(smp, groups, contacts, free=True)
            best = coordinate_ascent(smp, masks_f, seed, 0.10)
            report(f"{name} 자유", smp, best, 0.10)
            zs = ", ".join(f"{p[2]:+.3f}" for p in smp.points[best])
            print(f"     앵커 z: {zs}  (박스 z {smp.points[:,2].min():+.3f}~{smp.points[:,2].max():+.3f})")

        print("-- 6월 기하 수축 앵커 (그때 방식, 최적화 없음) --")
        d = np.linalg.norm(anchors_jun[:, None] - smp.points[None], axis=-1)
        idx = d.min(1).argsort()[:len(anchors_jun)]
        snap = d.argmin(1)
        print(f"  표면 스냅 거리: {[round(x*100,1) for x in d.min(1)]} cm")
        report(f"6월 앵커 {len(anchors_jun)}점 pad=0.10", smp, snap, 0.10)


if __name__ == "__main__":
    main()
