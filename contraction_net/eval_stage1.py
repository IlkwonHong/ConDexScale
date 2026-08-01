"""Stage 1 실용 지표 평가 — 필터 유/무 비교.

지표 (검증셋, N=3/4 각각):
  top1        top-1 이 "합격 중 최고 α" 와 정확 일치
  quality     고른 파티션 품질 / 최적 품질 (중앙값·평균)
  fail_pick   물리 불합격(또는 α=0) 파티션을 고른 비율
  top3        top-3 안에 최적 존재
디코드 2종: raw = argmax / filtered = 합격 확률 ≥ τ 인 클래스 안에서 argmax
            (전부 τ 미만이면 합격 확률 최대 클래스로 폴백)

실행: ./venv/bin/python -m contraction_net.eval_stage1 <ckpt.pt> [--tau 0.5] [--no-contact-geom]
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .stage1_dataset import Stage1Dataset
from .stage1_model import Stage1Net, Stage1NetV2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--no-contact-geom", action="store_true")
    ap.add_argument("--arch", default="v1", choices=["v1", "v2"])
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    va = Stage1Dataset(split="val", contact_geom=not args.no_contact_geom)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = (Stage1NetV2(va.vocab) if args.arch == "v2"
             else Stage1Net(len(va.vocab))).to(dev)
    missing, _ = model.load_state_dict(ck["model"], strict=False)
    has_pass = args.arch == "v1" and not any(k.startswith("pass_head") for k in missing)
    model.eval()
    print(f"{args.ckpt} (epoch {ck.get('epoch')}), 합격 헤드 {'있음' if has_pass else '없음(v1)'}")

    modes = ["raw"] + (["filtered"] if has_pass else [])
    stats = {(m, n): {"r": [], "top3": 0, "fail": 0, "top1": 0, "cnt": 0}
             for m in modes for n in (3, 4)}
    dl = DataLoader(va, batch_size=256, num_workers=4)
    with torch.no_grad():
        for b in dl:
            bb = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            logits, _, pl = model(bb)
            for n in (3, 4):
                t, v = bb[f"target_n{n}"], bb[f"valid_n{n}"]
                rows = t.sum(-1) > 0
                if not rows.any():
                    continue
                lg = logits[rows].masked_fill(~v[rows], float("-inf"))
                tr = t[rows]
                best = tr.argmax(-1)
                for mode in modes:
                    if mode == "raw":
                        cand = lg
                    else:
                        ok = (torch.sigmoid(pl[rows]) >= args.tau) & v[rows]
                        cand = lg.masked_fill(~ok, float("-inf"))
                        none = ~ok.any(-1)                       # 전부 필터되면 폴백
                        if none.any():
                            cand[none] = lg[none]
                    pred = cand.argmax(-1)
                    ratio = tr.gather(1, pred[:, None])[:, 0] / tr.max(-1).values
                    k = min(3, cand.shape[-1])
                    hit3 = (cand.topk(k, -1).indices == best[:, None]).any(-1)
                    s = stats[(mode, n)]
                    s["r"] += ratio.cpu().tolist()
                    s["top1"] += int((pred == best).sum())
                    s["top3"] += int(hit3.sum())
                    s["fail"] += int((ratio == 0).sum())
                    s["cnt"] += int(rows.sum())

    for n in (3, 4):
        print(f"\nN={n} (파지 {stats[(modes[0], n)]['cnt']:,}):")
        for mode in modes:
            s = stats[(mode, n)]
            r = np.array(s["r"])
            print(f"  [{mode:8s}] top1 {s['top1']/s['cnt']:.1%} | 품질비 중앙값 {np.median(r):.2f} "
                  f"평균 {r.mean():.2f} | 불합격 선택 {s['fail']/s['cnt']:.1%} | "
                  f"top3 적중 {s['top3']/s['cnt']:.1%}")


if __name__ == "__main__":
    main()
