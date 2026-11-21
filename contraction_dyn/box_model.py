"""ARCTIC 박스 물체 모형 — URDF 관성(질량·질량중심·관성 텐서)을 canonical 좌표로 제공.

좌표계: canonical (관절각 반영 전 정점 좌표, 경첩 = 원점·−z 축). rho_eval(2026-08-30)에서
URDF 관성 원점 = canonical 좌표임을 대조 확인. 뚜껑(top)은 canonical 에서 rotvec [0,0,−θ]
로 회전 (canonical_smp / MJCF hinge axis "0 0 -1" 과 동일).

제공:
    BoxModel.mass_com(theta)      → (총질량, 합성 질량중심 [3])
    BoxModel.inertia(theta)       → 합성 질량중심 기준 관성 텐서 [3,3] (canonical)
    BoxModel.inertial_str(link)   → MJCF <inertial .../> 문자열 (play_contact_mj 와 동일 규약)
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "extlib/dexmachina/dexmachina/assets/arctic/box/decomp/box_decomp.urdf"


class BoxModel:
    def __init__(self, path=URDF):
        root = ET.parse(path).getroot()
        self.links = {lk.get("name"): lk for lk in root.findall("link")}
        self.m, self.c, self.I = {}, {}, {}
        for nm, lk in self.links.items():
            ine = lk.find("inertial")
            org = ine.find("origin")
            self.c[nm] = np.array([float(x) for x in org.get("xyz").split()])
            rpy = [float(v) for v in org.get("rpy").split()]
            R = Rotation.from_euler("xyz", rpy).as_matrix()
            self.m[nm] = float(ine.find("mass").get("value"))
            g = {k: float(ine.find("inertia").get(k))
                 for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
            Im = np.array([[g["ixx"], g["ixy"], g["ixz"]],
                           [g["ixy"], g["iyy"], g["iyz"]],
                           [g["ixz"], g["iyz"], g["izz"]]])
            self.I[nm] = R @ Im @ R.T                 # 링크 프레임, 링크 질량중심 기준
        self.mass = sum(self.m.values())

    @staticmethod
    def _Rtop(theta):
        return Rotation.from_rotvec([0, 0, -theta]).as_matrix() if abs(theta) > 1e-8 \
            else np.eye(3)

    def mass_com(self, theta):
        Rt = self._Rtop(theta)
        c = (self.m["bottom"] * self.c["bottom"] + self.m["top"] * (Rt @ self.c["top"])) / self.mass
        return self.mass, c

    def inertia(self, theta):
        """합성 질량중심 기준 관성 텐서 (canonical 프레임)."""
        Rt = self._Rtop(theta)
        _, c = self.mass_com(theta)
        parts = [("bottom", np.eye(3)), ("top", Rt)]
        I = np.zeros((3, 3))
        for nm, R in parts:
            ck = R @ self.c[nm]
            d = ck - c
            I += R @ self.I[nm] @ R.T + self.m[nm] * (d @ d * np.eye(3) - np.outer(d, d))
        return I

    def inertial_str(self, link):
        ine = self.links[link].find("inertial")
        pos = ine.find("origin").get("xyz")
        Ib = self.I[link]
        return (f'<inertial pos="{pos}" mass="{self.m[link]}" fullinertia='
                f'"{Ib[0,0]} {Ib[1,1]} {Ib[2,2]} {Ib[0,1]} {Ib[0,2]} {Ib[1,2]}"/>')


if __name__ == "__main__":
    bm = BoxModel()
    m, c = bm.mass_com(0.0)
    print(f"질량 {m:.4f}kg  질량중심 {np.round(c, 4)}  관성 대각 {np.round(np.diag(bm.inertia(0.0)), 5)}")
