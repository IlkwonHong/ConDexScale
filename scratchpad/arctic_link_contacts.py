"""ARCTIC 전체 손(정점 778개) 마디별 접촉 추출 — 네트워크 입력용.

6월 측정기(손끝 구역 16정점만)를 확장: 손 전체 정점을 MANO 스키닝 가중치로
마디에 배정 → 마디별 접촉 면적(정점 수) + 손바닥 접촉 + 부위별 접촉 정점 위치.
문턱값은 Dexonomy 학습 접촉 테이블과 동일한 3mm 기본 (민감도로 5/8mm 병행).

자체 검증 3종:
  ① 물체 중심이 6월 JSON의 obj_center와 일치하는가
  ② 손끝 구역 정점(알려진 정답)이 "끝마디 칸"에 배정되는가
  ③ 부위 접촉 유무가 6월 손끝 판정과 대체로 일치하는가
실행: ./venv/bin/python scratchpad/arctic_link_contacts.py
"""
import json
import sys
import os.path as osp
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import measure_arctic_contact_events as M
from manopth.manolayer import ManoLayer

SEQ = sys.argv[1] if len(sys.argv) > 1 else "box_grab_01"
FRAMES = [int(a) for a in sys.argv[2:]] or [90, 300, 630]
SUBJ = "s01"
THRESHOLDS = [0.003, 0.005, 0.008]          # m — 3mm = 학습 테이블 기준
UNITS = ["index", "middle", "ring", "little", "thumb", "palm"]
OUT = ROOT / "scratchpad" / f"arctic_link_contacts_{SEQ}.npz"
from arctic_scene import json_map
JMAP = json_map(SEQ, SUBJ)


def vertex_link_map(layer):
    """정점 778 → (부위 0..5, 마디칸 0..2). 손끝 구역 정점으로 손가락 정체 검증."""
    W = layer.th_weights.numpy() if hasattr(layer, "th_weights") else \
        layer.smpl_data["weights"]
    dom = np.asarray(W).argmax(1)                       # 정점별 지배 관절 (0=손목)
    triples = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15)]
    finger_of = {}
    for fi, name in enumerate(M.FINGER_NAMES):
        zone = M.ZONE_IDX[fi]                           # 알려진 손끝 정점 16개
        j = np.bincount(dom[zone], minlength=16).argmax()
        for ti, tr in enumerate(triples):
            if j in tr:
                finger_of[ti] = fi
    assert len(finger_of) == 5, f"손가락-관절 대응 실패: {finger_of}"
    unit = np.full(778, 5, np.int8)                     # 기본 = 손바닥
    binv = np.full(778, 2, np.int8)                     # 손바닥 = 몸쪽 칸
    for ti, tr in enumerate(triples):
        fi = finger_of[ti]
        for k, j in enumerate(tr):                      # 몸쪽→끝 순서
            unit[dom == j] = fi
            binv[dom == j] = 2 - k                      # 끝마디=0, 가운데=1, 몸쪽=2
    # 검증 ②: 손끝 구역 정점이 (자기 손가락, 끝마디 칸)에 배정되는 비율
    ok = np.mean([
        (unit[M.ZONE_IDX[fi]] == fi).mean() * (binv[M.ZONE_IDX[fi]] == 0).mean()
        for fi in range(5)])
    print(f"  정점-마디 배정 검증: 손끝 구역의 (손가락,끝마디) 적중 {ok:.1%}")
    for u in range(6):
        print(f"    {UNITS[u]:6s}: 정점 {(unit == u).sum():3d}개 "
              f"(칸별 {[int(((unit == u) & (binv == b)).sum()) for b in (0, 1, 2)]})")
    return unit, binv


def full_verts(mano_data, side, layer, frames):
    rot = torch.from_numpy(np.asarray(mano_data[side]["rot"], np.float32))[frames]
    pose = torch.from_numpy(np.asarray(mano_data[side]["pose"], np.float32))[frames]
    trans = torch.from_numpy(np.asarray(mano_data[side]["trans"], np.float32))[frames]
    shape = torch.from_numpy(np.asarray(mano_data[side]["shape"], np.float32)).view(1, 10)
    F = rot.shape[0]
    verts, *_ = layer(
        th_pose_coeffs=torch.cat([rot.view(F, 3), pose.view(F, 45)], 1),
        th_betas=shape.expand(F, -1), th_trans=torch.zeros(F, 3), root_palm=False)
    return (verts + trans.unsqueeze(1)).numpy()          # (F, 778, 3) 세계 좌표


def main():
    root = M._DEFAULT_ARCTIC_ROOT
    mano_root = osp.join(M.REPO_ROOT, "network", "manopth", "mano", "models")
    layers = {s: ManoLayer(ncomps=45, side=s, mano_root=mano_root,
                           use_pca=False, flat_hand_mean=False)
              for s in ("right", "left")}
    print("정점-마디 배정 (오른손):")
    unit_r, bin_r = vertex_link_map(layers["right"])
    print("정점-마디 배정 (왼손):")
    unit_l, bin_l = vertex_link_map(layers["left"])

    mano = np.load(osp.join(root, "data", "raw_seqs", SUBJ, SEQ + ".mano.npy"),
                   allow_pickle=True).item()
    objd = np.load(osp.join(root, "data", "raw_seqs", SUBJ, SEQ + ".object.npy"),
                   allow_pickle=True)
    verts_m, parts = M.load_object_templates(root)[SEQ.split("_")[0]]
    ov = M.object_verts_world(verts_m, parts, objd[FRAMES], "cpu").numpy()  # (F,V,3)

    hv = {"right": full_verts(mano, "right", layers["right"], FRAMES),
          "left": full_verts(mano, "left", layers["left"], FRAMES)}
    vmap = {"right": (unit_r, bin_r), "left": (unit_l, bin_l)}

    save = {}
    for f_i, fr in enumerate(FRAMES):
        oc = ov[f_i].mean(0)
        j = json.load(open(JMAP[fr][0]))
        print(f"\n=== 프레임 {fr:05d} ===  검증①: 물체중심 vs 6월 JSON "
              f"{np.linalg.norm(oc - np.array(j['obj_center'])) * 100:.2f}cm")
        for side in ("right", "left"):
            uu, bb = vmap[side]
            d = np.linalg.norm(hv[side][f_i][:, None] - ov[f_i][None], axis=-1).min(1)
            jc = j["human_finger_contacts" if side == "right"
                   else "human_finger_contacts_left"]
            for thr in THRESHOLDS:
                c = d < thr
                cnt = [[int((c & (uu == u) & (bb == b)).sum()) for b in (0, 1, 2)]
                       for u in range(6)]
                touched = [UNITS[u] for u in range(6) if sum(cnt[u])]
                print(f"  {side:5s} {thr*1000:.0f}mm: 접촉 부위 {touched}")
            # 계층 문턱: 부위별로 5mm 접촉을 기본, 없으면 8mm 로 구제 (엄지 누락 대책)
            c5, c8 = d < 0.005, d < 0.008
            cnt = np.zeros((6, 3), np.float32)
            keep = np.zeros(len(d), bool)
            src = []
            for u in range(6):
                cu = c5 & (uu == u)
                if not cu.any():
                    cu = c8 & (uu == u)
                    if cu.any():
                        src.append(f"{UNITS[u]}(8mm)")
                keep |= cu
                for b in (0, 1, 2):
                    cnt[u, b] = (cu & (bb == b)).sum()
            s = cnt.sum(1, keepdims=True)
            save[f"{fr}_{side}_ratios"] = np.divide(
                cnt, s, out=np.zeros_like(cnt), where=s > 0)
            save[f"{fr}_{side}_mask"] = cnt.sum(1) > 0
            save[f"{fr}_{side}_cpos"] = (hv[side][f_i][keep] - oc).astype(np.float32)
            save[f"{fr}_{side}_cunit"] = uu[keep]
            print(f"        저장(계층): {[UNITS[u] for u in range(6) if cnt[u].sum()]}"
                  + (f" — 8mm 구제: {', '.join(src)}" if src else ""))
            print(f"        검증③ 6월 손끝판정(1.5cm): {sorted(jc.keys())}")
        save[f"{fr}_obj_center"] = oc
    np.savez_compressed(OUT, **save)
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
