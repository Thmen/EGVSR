#!/usr/bin/env python3
"""Generate a tiny synthetic Vid4-style sequence for smoke-testing EGVSR.

The real Vid4 / ToS3 / Gvt72 datasets are large and distributed via an external
(Baidu) link, so they are not available inside the Cloud Agent VM. This script
fabricates a short, deterministic RGB video with smooth motion, then produces the
matching low-resolution frames using the same BD degradation the models expect
(Gaussian blur with sigma=1.5 followed by x4 down-sampling). That is enough to run
the full test pipeline end to end (inference + PSNR / LPIPS / tOF metrics).

Output layout (matches PairedFolderDataset expectations):

    <root>/GT/<seq>/0001.png ...            # ground-truth HR frames
    <root>/Gaussian4xLR/<seq>/0001.png ...  # LR frames (x4 smaller)
"""
import argparse
import os
import os.path as osp

import cv2
import numpy as np


def make_gt_frame(t, h, w):
    """Build a colorful HR frame at time-step ``t`` with smooth global motion."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Moving diagonal color gradient (gives temporal coherence for tOF).
    shift = 12 * t
    r = (np.sin((xx + shift) / 18.0) * 0.5 + 0.5)
    g = (np.sin((yy + shift) / 22.0 + 1.0) * 0.5 + 0.5)
    b = (np.sin((xx + yy + shift) / 26.0 + 2.0) * 0.5 + 0.5)
    frame = np.stack([r, g, b], axis=-1)

    # A moving bright disc adds high-frequency detail that SR should recover.
    cx = int(w * (0.3 + 0.4 * (t / 8.0)))
    cy = int(h * (0.5 + 0.2 * np.sin(t / 2.0)))
    radius = max(h, w) // 8
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disc = np.clip(1.0 - dist / radius, 0.0, 1.0)[..., None]
    frame = frame * (1 - disc) + disc * np.array([1.0, 0.9, 0.2])

    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def gaussian_kernel(sigma):
    ksize = 1 + 2 * int(sigma * 3.0)
    ax = np.arange(ksize) - ksize // 2
    g1d = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g1d /= g1d.sum()
    return np.outer(g1d, g1d)


def degrade(gt, sigma, scale):
    """BD degradation: Gaussian blur then x4 stride down-sampling (per channel)."""
    kernel = gaussian_kernel(sigma)
    lr = cv2.filter2D(gt, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return lr[::scale, ::scale, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/Vid4",
                        help="dataset root to populate")
    parser.add_argument("--seq", default="demo", help="sequence name")
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    gt_dir = osp.join(args.root, "GT", args.seq)
    lr_dir = osp.join(args.root, "Gaussian4xLR", args.seq)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    for t in range(args.frames):
        gt = make_gt_frame(t, args.height, args.width)
        lr = degrade(gt, args.sigma, args.scale)
        name = f"{t + 1:04d}.png"
        # cv2 writes BGR; store BGR so the RGB round-trip in the loader is correct.
        cv2.imwrite(osp.join(gt_dir, name), gt[..., ::-1])
        cv2.imwrite(osp.join(lr_dir, name), lr[..., ::-1])

    print(f"Wrote {args.frames} frames to:\n  {gt_dir}\n  {lr_dir}")


if __name__ == "__main__":
    main()
