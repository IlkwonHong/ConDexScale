"""박스를 Stage 1+2 네트워크 입력 형식으로 조립 (손별 배치 dict 생성).

검증 2종 (박스 조립 전에 실행):
  A. 패치 생성 포팅 검증 — Dexonomy 물체 1개를 내 코드로 재생성해 저장 캐시와 대조
  B. 특징 조립 검증 — Dexonomy 검증 파지 1개를 내 조립기로 만들어
     실제 Stage2Dataset 출력과 필드별 대조 (augment=False 기준)

박스 경로: 키프레임(00090/00300/00630)별 세계 정렬 메시 → 표면 2048점 + 패치 128
  + arctic_link_contacts npz(5mm)의 부위 마스크·마디비율·접촉정점 → 손별 배치 dict
출력: scratchpad/box_net_batches.pt
실행: ./venv/bin/python scratchpad/box_net_input.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratchpad"))

from contraction_gt.surface import sample_surface
from contraction_net.partition_vocab import enumerate_valid
from contraction_net.stage1_dataset import N_UNIT
from contraction_net.stage2_dataset import Stage2Dataset, UNITS

K_LOCAL, N_PATCH, KNN = 16, 128, 6
SEQ = sys.argv[1] if len(sys.argv) > 1 else "box_grab_01"
FRAMES = [int(a) for a in sys.argv[2:]] or [90, 300, 630]
CON_NPZ = ROOT / f"scratchpad/arctic_link_contacts_{SEQ}.npz"
OUT = ROOT / f"scratchpad/net_batches_{SEQ}.pt"
from arctic_scene import scene
FSTATS = json.loads((ROOT / "data/contraction_net/feat_stats.json").read_text())
REL_PATCH_ADJ, REL_PATCH_NON, REL_UNIT_CON, REL_UNIT_NON = 0, 1, 2, 3
REL_UNIT_UNIT, REL_CLS, REL_SELF = 4, 5, 6


def build_patch(pts, nrm):
    """build_patch_cache.build_one 포팅 (동일 논리·결정적 FPS)."""
    sel = [0]
    d = np.linalg.norm(pts - pts[0], axis=1)
    for _ in range(N_PATCH - 1):
        i = int(d.argmax())
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    idx = np.array(sel)
    centers = pts[idx]
    pid = ((pts[:, None] - centers[None]) ** 2).sum(-1).argmin(1).astype(np.int16)
    normals, curv = np.zeros((N_PATCH, 3)), np.zeros(N_PATCH)
    for p in range(N_PATCH):
        m = pid == p
        mean = nrm[m].mean(0) if m.any() else nrm[idx[p]]
        n = np.linalg.norm(mean)
        normals[p] = mean / max(n, 1e-9)
        curv[p] = 1.0 - n
    cd = ((centers[:, None] - centers[None]) ** 2).sum(-1)
    np.fill_diagonal(cd, np.inf)
    nn = cd.argsort(1)[:, :KNN]
    es = np.repeat(np.arange(N_PATCH), KNN).astype(np.int16)
    ed = nn.reshape(-1).astype(np.int16)
    return dict(centers=centers.astype(np.float32), cnormals=normals.astype(np.float32),
                curv=curv.astype(np.float32), pid=pid, edge_src=es, edge_dst=ed)


def build_local(pts, nrm, centers, pid):
    rng = np.random.default_rng(0)
    local = np.zeros((N_PATCH, K_LOCAL, 6), np.float32)
    lmask = np.zeros((N_PATCH, K_LOCAL), bool)
    for p in range(N_PATCH):
        m = np.nonzero(pid == p)[0]
        if len(m) > K_LOCAL:
            m = rng.choice(m, K_LOCAL, replace=False)
        local[p, :len(m), :3] = pts[m] - centers[p]
        local[p, :len(m), 3:] = nrm[m]
        lmask[p, :len(m)] = True
    return local, lmask


def anchor0_of(umask, cpos, cunit, pts, nrm):
    """엄지 접촉 시 엄지, 아니면 대립 부위(units.opposition_unit 의 D2 규칙:
    나머지 부위들의 개수 가중 평균 법선과 가장 마주보는 부위)."""
    if umask[UNITS.index("thumb")]:
        return "thumb"
    present = [u for u in range(N_UNIT) if umask[u]]
    nn = ((cpos[:, None] - pts[None]) ** 2).sum(-1).argmin(1)
    un = {u: nrm[nn[cunit == u]].mean(0) for u in present}
    best, score = None, np.inf
    for u in present:
        others = [v for v in present if v != u]
        w = np.array([(cunit == v).sum() for v in others], np.float64)
        mean = sum(wi * un[v] for wi, v in zip(w, others)) / max(w.sum(), 1e-9)
        s = float(np.dot(un[u], mean))
        if s < score:
            best, score = u, s
    return UNITS[best]


def assemble(pts, nrm, com, patch, local, lmask, r_obj, ts,
             ratios, umask, pu, pp, vocab, anchor0):
    """Stage2Dataset.__getitem__ 의 특징 수식 재현 (augment=0, contact_geom=False,
    com_feat=True, feat_norm=True)."""
    scale = ts / r_obj
    parts_ = [patch["centers"] * scale, patch["cnormals"], patch["curv"][:, None]]
    crel = (patch["centers"] - com) * scale
    parts_ += [crel, np.linalg.norm(crel, axis=1, keepdims=True)]
    patch_feat = np.concatenate(parts_, 1).astype(np.float32)
    loc = local.copy()
    loc[:, :, :3] *= scale

    unit_feat = np.zeros((N_UNIT, 10), np.float32)
    unit_feat[:, 6:9] = ratios
    unit_feat[:, 9] = umask

    NN = N_PATCH + N_UNIT + 1
    rel = np.full((NN, NN), REL_PATCH_NON, np.int8)
    rel[N_PATCH:N_PATCH + N_UNIT, :N_PATCH] = REL_UNIT_NON
    rel[:N_PATCH, N_PATCH:N_PATCH + N_UNIT] = REL_UNIT_NON
    rel[N_PATCH:N_PATCH + N_UNIT, N_PATCH:N_PATCH + N_UNIT] = REL_UNIT_UNIT
    rel[NN - 1, :] = REL_CLS
    rel[:, NN - 1] = REL_CLS
    rel[patch["edge_src"], patch["edge_dst"]] = REL_PATCH_ADJ
    rel[patch["edge_dst"], patch["edge_src"]] = REL_PATCH_ADJ
    for u, p in zip(pu, pp):
        rel[N_PATCH + int(u), int(p)] = REL_UNIT_CON
        rel[int(p), N_PATCH + int(u)] = REL_UNIT_CON
    np.fill_diagonal(rel, REL_SELF)

    pid = patch["pid"]
    pparts = [(pts - patch["centers"][pid]) * scale, nrm]
    prel = (pts - com) * scale
    pparts += [prel, np.linalg.norm(prel, axis=1, keepdims=True)]
    point_feat = np.concatenate(pparts, 1).astype(np.float32)

    def std(x, key):
        m, sd = (np.array(FSTATS[key]["mean"], np.float32),
                 np.array(FSTATS[key]["std"], np.float32))
        return (x - m) / sd
    patch_feat = std(patch_feat, "patch_feat")
    loc = std(loc, "patch_local")
    unit_feat = std(unit_feat, "unit_feat")
    point_feat = std(point_feat, "point_feat")

    out = {
        "patch_feat": torch.from_numpy(patch_feat),
        "patch_local": torch.from_numpy(loc),
        "patch_local_mask": torch.from_numpy(lmask),
        "unit_feat": torch.from_numpy(unit_feat),
        "rel_type": torch.from_numpy(rel),
        "log_robj": torch.tensor([np.log(r_obj / 0.1)], dtype=torch.float32),
        "point_feat": torch.from_numpy(point_feat),
        "point_patch": torch.from_numpy(pid.astype(np.int64)),
    }
    units_present = [UNITS[u] for u in range(N_UNIT) if umask[u]]
    for n in (3, 4):
        v = torch.zeros(len(vocab), dtype=torch.bool)
        for part in enumerate_valid(units_present, anchor0, n):
            v[vocab.id_of(part)] = True
        out[f"valid_n{n}"] = v
    return out


def verify(va):
    """검증 A·B — Dexonomy 물체·파지 1개 재현 대조."""
    gid, e = va.samples[0]
    ds, oid = va.obj_of[gid]
    ref = va[0]
    with np.load(ROOT / "data/contraction_gt_pilot/object_cache" / ds / f"{oid}.npz") as z:
        pts, nrm = z["points"].astype(np.float32), z["normals"].astype(np.float32)
        com = z["com"].astype(np.float32)
    patch = build_patch(pts, nrm)
    with np.load(ROOT / "data/contraction_net/patch_cache" / ds / f"{oid}.npz") as z:
        for a, b in [("centers", "centers"), ("point_patch_id", "pid"),
                     ("edge_src", "edge_src")]:
            same = np.allclose(z[a], patch[b])
            print(f"  검증A {a:16s}: {'일치' if same else '불일치!!'}")
    local, lmask = build_local(pts, nrm, patch["centers"], patch["pid"])

    rel, ridx = gid.rsplit("#", 1)
    con = va._contacts.get(rel)
    la = con["link_contact_area"][int(ridx)]
    umask = con["finger_contact"][int(ridx)].copy()
    from contraction_net.stage1_dataset import _LINK_NAMES, _LINK_UNIT, _LINK_BIN
    ratios = np.zeros((N_UNIT, 3), np.float32)
    for li in range(len(_LINK_NAMES)):
        ratios[_LINK_UNIT[li], _LINK_BIN[li]] += la[li]
    s = ratios.sum(1, keepdims=True)
    ratios = np.divide(ratios, s, out=np.zeros_like(ratios), where=s > 0)
    pu, pp = va.patch_con.get(gid, (np.array([]), np.array([])))
    a0 = "thumb" if umask[UNITS.index("thumb")] else None
    mine = assemble(pts, nrm, com, patch, local, lmask,
                    e["r_obj"], e["total_scale"], ratios, umask, pu, pp, va.vocab, a0)
    for k in ("patch_feat", "patch_local", "patch_local_mask", "unit_feat",
              "rel_type", "log_robj", "point_feat", "point_patch",
              "valid_n3", "valid_n4"):
        a, b = mine[k].numpy(), ref[k].numpy()
        same = a.shape == b.shape and np.allclose(a.astype(np.float64),
                                                 b.astype(np.float64), atol=1e-5)
        mx = np.abs(a.astype(np.float64) - b.astype(np.float64)).max() if a.shape == b.shape else -1
        print(f"  검증B {k:16s}: {'일치' if same else f'불일치!! (최대차 {mx:.2e})'}")


def main():
    va = Stage2Dataset(split="val", contact_geom=False, com_feat=True, feat_norm=True)
    print("=== 검증 (Dexonomy 재현 대조) ===")
    verify(va)

    print(f"\n=== {SEQ} 배치 조립 ===")
    con = np.load(CON_NPZ)
    batches = {}
    for fr in FRAMES:
        smp, _meta = scene(SEQ, fr, sample_surface)
        pts = smp.points.astype(np.float32)
        nrm = smp.normals.astype(np.float32)
        patch = build_patch(pts, nrm)
        local, lmask = build_local(pts, nrm, patch["centers"], patch["pid"])
        for side in ("right", "left"):
            ratios = con[f"{fr}_{side}_ratios"]
            umask = con[f"{fr}_{side}_mask"].astype(np.float32)
            cpos = con[f"{fr}_{side}_cpos"]        # 접촉 정점 (물체중심 원점, 세계 축)
            cunit = con[f"{fr}_{side}_cunit"]
            dchk = np.linalg.norm(cpos[:, None] - pts[None], axis=-1).min(1)
            print(f"    좌표 사슬 검증 {fr:05d} {side}: 접촉정점→표면 "
                  f"평균 {dchk.mean()*1000:.1f}mm 최대 {dchk.max()*1000:.1f}mm")
            near = ((cpos[:, None] - patch["centers"][None]) ** 2).sum(-1).argmin(1)
            pairs = sorted(set(zip(cunit.tolist(), near.tolist())))
            pu = np.array([u for u, _ in pairs], np.int8)
            pp = np.array([p for _, p in pairs], np.int16)
            a0 = anchor0_of(umask, cpos, cunit, pts, nrm)
            b = assemble(pts, nrm, smp.com.astype(np.float32), patch, local, lmask,
                         smp.r_obj, 1.0, ratios, umask, pu, pp, va.vocab, a0)
            b["_anchor0"] = a0
            b["_cpos"], b["_cunit"] = torch.from_numpy(cpos), torch.from_numpy(cunit)
            b["_r_obj"] = smp.r_obj
            batches[f"{fr}_{side}"] = b
            print(f"  {fr:05d} {side:5s}: 접촉부위 {[UNITS[u] for u in range(6) if umask[u]]}, "
                  f"부위-패치 연결 {len(pairs)}개, 유효 파티션 n3={int(b['valid_n3'].sum())} "
                  f"n4={int(b['valid_n4'].sum())}")
    torch.save(batches, OUT)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
