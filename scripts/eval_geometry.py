#!/usr/bin/env python3
"""Eval depth + normal heads on synthetic val (GT available). Depth: AbsRel, RMSE, delta<1.25.
Normals: mean angular error (deg). On valid pixels only (depth>0 / |normal|>0)."""
import argparse, csv, sys, glob
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_multitask_geometry import GeoModel, letterbox_map, FeatureHook, Decoder
import __main__
__main__.FeatureHook = FeatureHook; __main__.GeoModel = GeoModel; __main__.Decoder = Decoder
AUX = ROOT / "data/synthetic_yolo/aux"
CKPT = ROOT / "output/synthetic/searched_geo/weights/last.pt"


def load_model(device):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    trained = ck["model"].float()
    m = GeoModel(str(ROOT.parent.parent / "lightweight_detector_search/results/architectures/v2_cib_cib_scd_psa_w25d34.yaml"))
    m.load_state_dict(trained.state_dict(), strict=False)
    return m.eval().to(device)


@torch.no_grad()
def main(n=60, device=None, out=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = load_model(device)
    imgs = sorted(glob.glob(str(ROOT / "data/synthetic_yolo/images/val/*.jpg")))[:n]
    absrel, rmse, d125, ang = [], [], [], []
    rows = []
    for ip in imgs:
        stem = Path(ip).stem
        im = cv2.imread(ip)
        lb = letterbox_map(im, 640, cv2.INTER_LINEAR)
        t = torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].to(device)
        m(t)  # populate hook
        f = m._f["p3"]
        dp = F.softplus(m.depth_head.net(f)) if False else m.depth_head(f)  # depth_head has softplus inside
        npd = F.normalize(m.normal_head(f), dim=1)
        # GT (letterboxed same)
        dg = torch.from_numpy(letterbox_map(np.load(AUX / "depth/val" / (stem + ".npy")).astype(np.float32), 640)).to(device)
        ng = torch.from_numpy(letterbox_map(np.load(AUX / "normal/val" / (stem + ".npy")).astype(np.float32), 640)).permute(2, 0, 1).to(device)
        dp = F.interpolate(dp, size=dg.shape, mode="bilinear", align_corners=False)[0, 0]
        npd = F.interpolate(npd, size=dg.shape, mode="bilinear", align_corners=False)[0]
        vd = dg > 0.5
        if vd.sum() > 100:
            p, g = dp[vd].clamp(0.1, 80), dg[vd]
            ar = ((p - g).abs() / g).mean().item()
            er = torch.sqrt(((p - g) ** 2).mean()).item()
            de = (torch.max(p / g, g / p) < 1.25).float().mean().item()
            absrel.append(ar)
            rmse.append(er)
            d125.append(de)
        vn = ng.abs().sum(0) > 0.1
        ae = float("nan")
        if vn.sum() > 100:
            cos = (npd * F.normalize(ng, dim=0)).sum(0)[vn].clamp(-1, 1)
            ae = torch.rad2deg(torch.arccos(cos)).mean().item()
            ang.append(ae)
        rows.append({
            "frame": stem,
            "depth_absrel": ar if vd.sum() > 100 else float("nan"),
            "depth_rmse_m": er if vd.sum() > 100 else float("nan"),
            "depth_delta125": de if vd.sum() > 100 else float("nan"),
            "normal_angular_error_deg": ae,
        })
    summary = {
        "n_depth": len(absrel),
        "depth_absrel": float(np.mean(absrel)),
        "depth_rmse_m": float(np.mean(rmse)),
        "depth_delta125": float(np.mean(d125)),
        "n_normal": len(ang),
        "normal_angular_error_deg": float(np.mean(ang)),
    }
    print(f"DEPTH  (n={summary['n_depth']}):  AbsRel={summary['depth_absrel']:.3f}  RMSE={summary['depth_rmse_m']:.2f}m  delta<1.25={summary['depth_delta125']:.3f}")
    print(f"NORMAL (n={summary['n_normal']}):  mean angular error={summary['normal_angular_error_deg']:.1f} deg")
    if out:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "geometry_frame_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (out / "geometry_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        md = [
            "# Synthetic Geometry Evaluation",
            "",
            f"Depth evaluated on {summary['n_depth']} synthetic validation frames with valid depth targets.",
            f"Normals evaluated on {summary['n_normal']} synthetic validation frames with valid normal targets.",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| Depth AbsRel | {summary['depth_absrel']:.3f} |",
            f"| Depth RMSE m | {summary['depth_rmse_m']:.2f} |",
            f"| Depth delta < 1.25 | {summary['depth_delta125']:.3f} |",
            f"| Normal mean angular error deg | {summary['normal_angular_error_deg']:.1f} |",
            "",
            "Interpretation: these synthetic-only geometry metrics support the paper's claim that privileged "
            "depth and normal supervision can be measured in simulation. They should not be presented as "
            "real-world geometry accuracy because no real depth/normal ground truth is available in the "
            "dashcam deer clips.",
        ]
        (out / "geometry_summary.md").write_text("\n".join(md))
        print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = None if args.device == "auto" else torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    main(args.n, dev, args.out)
