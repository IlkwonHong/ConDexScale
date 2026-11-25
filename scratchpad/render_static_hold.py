"""ρ 정적 파지(공중 유지) 시험의 시뮬레이션 영상 렌더링.

static_hold_test 와 물리 동일하되 좌표만 등가 변환: 박스를 실측 최악 자세
(R_worst)로 기울이고 중력을 세계 −z 로 둔다 (시험은 박스 정자세 + 기울인
중력 — 두 표현은 등가). 영상에는 "기울어진 박스가 팁 6개만으로 공중에서
버티는" 장면이 그대로 나온다.

구간마다: 정착(유령 홀드) → 유령 해제 → 2초 자유 유지. 상단에 구간·ρ·판정
자막. 팁 색: 왼손 주황 / 오른손 파랑 (시각화 규약).

실행: ./venv/bin/python scratchpad/render_static_hold.py box_use_02 [--placement tuned]
출력: md_claude/dexmachina/static_hold_rho_{seq}.mp4
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contraction_dyn.static_hold_test import (equilibrium_forces, build_model,  # noqa: E402
                                              quat_angle, Q0, KP, TIP_R, TIP_M,
                                              G, DT, SETTLE1, SETTLE2, HOLD_T,
                                              PASS_POS, PASS_ROT)

FPS, W, H = 30, 800, 600
FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


def render_segment(row, plc, cam_azim=150):
    arms = sorted(plc["arms"])
    pos_c = np.array([plc["arms"][a]["pos"] for a in arms])
    nrm_c = np.array([plc["arms"][a]["nrm_in"] for a in arms])
    cen, com = np.array(row["cen"]), np.array(row["com_ref"])
    Rw = np.array(row["R_frames"][plc["worst_frame_i"]])
    quat = Rotation.from_matrix(Rw).as_quat()[[3, 0, 1, 2]]     # wxyz

    # 평형해는 canonical 에서 풀고 세계로 회전 (시험과 동일한 해)
    g_dir_c = Rw.T @ np.array([0, 0, -1.0])
    w_req_c = np.concatenate([-row["mass"] * G * g_dir_c, np.zeros(3)])
    f_c, resid = equilibrium_forces(pos_c - com, nrm_c, w_req_c,
                                    row["mu_t"], row["r_obj"], row["f_cap"])
    f_w = f_c @ Rw.T
    nrm_w = nrm_c @ Rw.T
    anchors_w = (pos_c + cen) @ Rw.T                            # 몸체좌표→세계 (Q0 기준)

    # 카메라: MjvCamera.lookat 이 이 빌드에서 설정 불가 → MJCF 고정 카메라를
    # mocap 몸체(rig)에 달아, 매 프레임 rig 회전으로 박스 주위를 공전시킨다.
    look = Q0 + Rw @ cen
    el, dist = np.deg2rad(-18), 0.8
    campos = dist * np.array([np.cos(el), 0.0, -np.sin(el)])   # rig 기준 (az=0)
    f = -campos / np.linalg.norm(campos)
    r_ = np.cross(f, [0, 0, 1.0])
    r_ /= np.linalg.norm(r_)
    u = np.cross(r_, f)
    cam_xml = (
        f'<body name="camrig" mocap="true" pos="{look[0]} {look[1]} {look[2]}">'
        f'<camera name="viewcam" pos="{campos[0]} {campos[1]} {campos[2]}" '
        f'xyaxes="{r_[0]} {r_[1]} {r_[2]} {u[0]} {u[1]} {u[2]}"/></body>')
    model = build_model(row["theta"], list(zip(pos_c + cen, nrm_c)), cen,
                        row["mu_t"], extra=cam_xml)
    BID_B = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_bottom")
    TIP_BIDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tip{a}")
                for a in range(len(arms))]
    # 팁 색 (왼손 주황 / 오른손 파랑)
    for a, arm in enumerate(arms):
        gid = model.body_geomadr[TIP_BIDS[a]]
        model.geom_rgba[gid] = ([0.93, 0.49, 0.19, 1] if arm.startswith("left")
                                else [0.18, 0.46, 0.71, 1])
    d = mujoco.MjData(model)
    d.qpos[3:7] = quat                                           # 박스를 실측 자세로

    # 레이캐스트 표면 정렬 (세계 좌표)
    mujoco.mj_forward(model, d)
    grp = np.zeros(6, np.uint8)
    grp[0] = 1
    gid = np.zeros(1, np.int32)
    touch = []
    for a in range(len(arms)):
        aw = Q0 + anchors_w[a]
        hit = None
        for back in (0.05, 0.03, 0.015, 0.008):
            o = aw - nrm_w[a] * back
            dist = mujoco.mj_ray(model, d, o, nrm_w[a], grp, 1, -1, gid)
            if dist >= 0 and gid[0] >= 0:
                h = o + nrm_w[a] * dist
                if np.linalg.norm(h - aw) < 0.04:
                    hit = h
                    break
        touch.append((hit if hit is not None else aw) - nrm_w[a] * TIP_R)
    for a in range(len(arms)):
        d.qpos[7 + 3 * a: 10 + 3 * a] = touch[a]
        d.ctrl[3 * a: 3 * a + 3] = touch[a] + f_w[a] / KP
        d.xfrc_applied[TIP_BIDS[a], 2] = TIP_M * G

    def hold_wrench(gain):
        perr = Q0 - d.qpos[:3]
        R_cur = Rotation.from_quat(d.qpos[3:7][[1, 2, 3, 0]]).as_matrix()
        rvec = Rotation.from_matrix(R_cur @ Rw.T).as_rotvec()
        d.xfrc_applied[BID_B, :3] = gain * (600.0 * perr - 30.0 * d.qvel[:3])
        d.xfrc_applied[BID_B, 3:] = gain * (-25.0 * rvec - 1.5 * d.qvel[3:6])

    rend = mujoco.Renderer(model, H, W)
    frames, rec_every = [], max(1, int(1 / (FPS * DT)))
    font_b = ImageFont.truetype(FONT, 26)
    font_s = ImageFont.truetype(FONT, 20)

    orbit = [np.deg2rad(cam_azim)]                 # 공전각 (스냅마다 전진)

    def snap(phase, extra=""):
        orbit[0] += np.deg2rad(30.0) / FPS         # 30°/초 공전
        a = orbit[0]
        d.mocap_quat[0] = [np.cos(a / 2), 0, 0, np.sin(a / 2)]
        rend.update_scene(d, camera="viewcam")
        im = Image.fromarray(rend.render())
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W, 78], fill=(0, 0, 0, 160))
        dr.text((14, 8), f"box_use_02 구간 {row['seg']} · {row['hand']} · "
                f"팔 {len(arms)} · ρ={plc['rho_worst']:+.3f} "
                f"({plc['rho_worst']*row['f_cap']:+.1f}N)", font=font_b,
                fill=(255, 255, 255))
        dr.text((14, 46), phase + extra, font=font_s, fill=(255, 230, 120))
        frames.append(np.asarray(im))

    # 1) 정착 (마지막 0.5초만 녹화)
    n1 = int(SETTLE1 / DT)
    for i in range(n1):
        hold_wrench(1.0)
        mujoco.mj_step(model, d)
        if i > n1 - int(0.5 / DT) and i % rec_every == 0:
            snap("정착: 유령 홀드 + LP 평형해 사전 인가")
    # 2) 유령 해제
    for i in range(int(SETTLE2 / DT)):
        hold_wrench(max(0.0, 1.0 - i * DT / 0.75))
        mujoco.mj_step(model, d)
        if i % rec_every == 0:
            snap(f"유령 홀드 해제 중 ({max(0.0, 1.0 - i*DT/0.75)*100:3.0f}%)")
    d.xfrc_applied[BID_B, :] = 0
    # 3) 자유 유지
    p0, q0 = d.xpos[BID_B].copy(), d.qpos[3:7].copy()
    max_p, max_r = 0.0, 0.0
    for i in range(int(HOLD_T / DT)):
        mujoco.mj_step(model, d)
        max_p = max(max_p, float(np.linalg.norm(d.xpos[BID_B] - p0)))
        max_r = max(max_r, quat_angle(d.qpos[3:7], q0))
        if i % rec_every == 0:
            snap("자유 유지 (외력 없음 — 접촉력만)",
                 f"  드리프트 {max_p*100:.2f}cm / {np.degrees(max_r):.1f}°")
    ok = max_p <= PASS_POS and max_r <= PASS_ROT
    verdict = ("통과" if ok else "실패") + \
        f" (최대 {max_p*100:.2f}cm / {np.degrees(max_r):.1f}°)"
    for _ in range(FPS):                                        # 판정 1초 정지컷
        frames.append(frames[-1])
    im = Image.fromarray(frames[-1].copy())
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, H - 56, W, H], fill=(0, 90, 0) if ok else (120, 0, 0))
    dr.text((14, H - 46), f"판정: {verdict}", font=font_b, fill=(255, 255, 255))
    frames.extend([np.asarray(im)] * FPS)
    del rend
    return frames, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", nargs="?", default="box_use_02")
    ap.add_argument("--subj", default="s01")
    ap.add_argument("--placement", default="tuned", choices=["tuned", "baseline"])
    args = ap.parse_args()
    cap = json.load(open(ROOT / "data/contraction_dyn/static_capacity"
                         / f"{args.subj}__{args.seq}.json"))
    all_frames, n_ok, n = [], 0, 0
    for row in cap["rows"]:
        plc = row.get(args.placement)
        if not plc:
            continue
        fr, ok = render_segment(row, plc)
        all_frames.extend(fr)
        n += 1
        n_ok += ok
        print(f"seg{row['seg']:2d} {'통과' if ok else '실패'} "
              f"(누적 {len(all_frames)}프레임)", flush=True)
    import imageio
    out = ROOT / "md_claude/dexmachina" / \
        f"static_hold_rho_{args.seq}_{args.placement}.mp4"
    imageio.mimsave(out, all_frames, fps=FPS)
    print(f"저장: {out} ({len(all_frames)/FPS:.0f}s, {n_ok}/{n} 통과)")


if __name__ == "__main__":
    main()
