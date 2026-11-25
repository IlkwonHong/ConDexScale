"""ARCTIC 시퀀스 공용 장면 헬퍼 — 키프레임의 세계 정렬 표면 표본 생성.

메시 출처: ARCTIC 원본 템플릿 (mm→m, parts.json 가동부).
관절: 가동부(parts==0)를 canonical −Z 축으로 arti 각도만큼 회전 (6월 검증 규칙).
프레임: 6월 JSON 파일 번호가 시퀀스별 연속이라 file_name 으로 대응표를 만든다.
"""
import json
import os
import os.path as osp
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ARCTIC = ROOT.parent / "ext_proj/arctic/unpack/arctic_data"


def json_map(seq, subj="s01"):
    """프레임 번호 → (6월 JSON 경로, 그래프 JSON 경로)."""
    d = ROOT / f"data/results_arctic/{subj}/{seq}/Robotiq/mu0.9"
    out = {}
    for f in os.listdir(d):
        j = json.load(open(d / f))
        fr = int(j["file_name"][-5:])
        out[fr] = (d / f, ROOT / "data/arctic_cache/sample_graphs"
                   / f"{j['file_name']}_graph.json")
    return out


def load_template(obj):
    mesh = trimesh.load_mesh(ARCTIC / "data/meta/object_vtemplates" / obj / "mesh.obj",
                             process=False)
    parts = np.array(json.load(open(ARCTIC / "data/meta/object_vtemplates" / obj
                                    / "parts.json")), dtype=np.int32)
    return np.array(mesh.vertices, np.float64) / 1000.0, np.array(mesh.faces), parts


def scene(seq, frame, sample_fn, n=2048):
    """키프레임의 세계 정렬(물체중심 원점) 표면 표본 + 메타."""
    obj = seq.split("_")[0]
    jp, gp = json_map(seq)[frame]
    j, g = json.load(open(jp)), json.load(open(gp))
    R = np.array(g["obj_rotation"], np.float64)
    cen = np.array(g["canonical_centroid"], np.float64)
    theta = float(j["obj_arti_angle"])
    V, F, parts = load_template(obj)
    V = V.copy()
    top = parts == 0
    if abs(theta) > 1e-8:
        Ra = Rotation.from_rotvec([0, 0, -theta]).as_matrix()
        V[top] = V[top] @ Ra.T
    mesh_w = trimesh.Trimesh(vertices=(R @ (V - cen).T).T, faces=F, process=False)
    smp = sample_fn(mesh_w, n)
    return smp, dict(j=j, g=g, R=R, cen=cen, theta=theta)
