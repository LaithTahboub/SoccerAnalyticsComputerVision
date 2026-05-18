import argparse
import json

import numpy as np

from .homography import fit_homography, smooth_homography_seq, project_points
from .pipeline import data_dir, out_dir

# experiment 1: player localization accuracy vs ground truth.
# three conditions, error in metres on the pitch:
#   ours (gt lines)         our homography from the dataset's pitch lines
#   ours (soccernet inputs) our homography from soccernet's detected lines/boxes
#   soccernet               soccernet's own predicted positions
# the first uses perfect inputs (an upper bound); the other two share
# soccernet's detections, so they isolate the geometry method.

iou_thresh = 0.5


def load_gt(clip):
    labels = json.loads((data_dir / clip / "Labels-GameState.json").read_text())
    images = {im["image_id"]: im for im in labels["images"]}
    image_ids = sorted(images, key=lambda k: images[k]["file_name"])
    w = images[image_ids[0]]["width"]
    h = images[image_ids[0]]["height"]

    lines_per = {}
    gt_per = {}
    for ann in labels["annotations"]:
        iid = ann["image_id"]
        if ann.get("supercategory") == "pitch":
            lines_per[iid] = ann.get("lines", {})
            continue
        bp = ann.get("bbox_pitch")
        b = ann.get("bbox_image")
        attrs = ann.get("attributes") or {}
        if not isinstance(bp, dict) or not isinstance(b, dict):
            continue
        if attrs.get("role") == "ball":
            continue
        gt_per.setdefault(iid, []).append({
            "bbox": (b["x"], b["y"], b["w"], b["h"]),
            "foot": (b["x_center"], b["y"] + b["h"]),
            "pitch": (bp["x_bottom_middle"], bp["y_bottom_middle"]),
        })
    return image_ids, w, h, lines_per, gt_per


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    ux, uy = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ux - ix) * max(0.0, uy - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match(sn_dets, gt_dets):
    # greedy image-space iou matching of soccernet detections to ground truth
    pairs = []
    for si, sd in enumerate(sn_dets):
        for gi, gd in enumerate(gt_dets):
            v = iou(sd["bbox"], gd["bbox"])
            if v >= iou_thresh:
                pairs.append((v, si, gi))
    pairs.sort(reverse=True)
    used_s, used_g, matched = set(), set(), []
    for v, si, gi in pairs:
        if si in used_s or gi in used_g:
            continue
        used_s.add(si)
        used_g.add(gi)
        matched.append((si, gi))
    return matched


def homographies(lines_seq, w, h):
    return smooth_homography_seq([fit_homography(ln, w, h) for ln in lines_seq], w, h)


def evaluate_clip(clip):
    image_ids, w, h, lines_per, gt_per = load_gt(clip)

    # condition 1: our homography from ground truth lines
    g_gt = homographies([lines_per.get(i, {}) for i in image_ids], w, h)
    e_gt = []
    for g, iid in zip(g_gt, image_ids):
        if g is None:
            continue
        for d in gt_per.get(iid, []):
            p = project_points(g, np.array([d["foot"]]))[0]
            if np.all(np.isfinite(p)):
                e_gt.append(np.hypot(p[0] - d["pitch"][0], p[1] - d["pitch"][1]))

    # conditions 2 and 3: soccernet's detected lines and boxes
    e_ours, e_sn = [], []
    sn_path = out_dir / clip / "soccernet.json"
    if sn_path.exists():
        sn = json.loads(sn_path.read_text())
        g_sn = homographies([sn.get(str(i), {}).get("lines", {}) for i in image_ids], w, h)
        for g, iid in zip(g_sn, image_ids):
            if g is None:
                continue
            sn_dets = [d for d in sn.get(str(iid), {}).get("dets", [])
                       if d["role"] != "ball" and d["pitch"] is not None]
            gt_dets = gt_per.get(iid, [])
            for si, gi in match(sn_dets, gt_dets):
                gt_xy = gt_dets[gi]["pitch"]
                p = project_points(g, np.array([sn_dets[si]["foot"]]))[0]
                if not np.all(np.isfinite(p)):
                    continue
                sn_xy = sn_dets[si]["pitch"]
                e_ours.append(np.hypot(p[0] - gt_xy[0], p[1] - gt_xy[1]))
                e_sn.append(np.hypot(sn_xy[0] - gt_xy[0], sn_xy[1] - gt_xy[1]))

    return e_gt, e_ours, e_sn


def median(errors):
    return np.median(errors) if errors else float("nan")


def summary(name, errors):
    e = np.array(errors)
    if len(e) == 0:
        return "%-26s no data" % name
    return ("%-26s N=%-7d median=%5.2f  mean=%6.2f  p90=%6.2f  "
            "<1m=%4.1f%%  <2m=%4.1f%%  <5m=%4.1f%%") % (
        name, len(e), np.median(e), e.mean(), np.percentile(e, 90),
        100 * (e < 1).mean(), 100 * (e < 2).mean(), 100 * (e < 5).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*", help="clip names; default = all with soccernet.json")
    args = ap.parse_args()

    if args.clips:
        clips = args.clips
    else:
        clips = sorted(p.name for p in out_dir.iterdir()
                       if (p / "soccernet.json").exists())

    all_gt, all_ours, all_sn = [], [], []
    rows = []
    for clip in clips:
        e_gt, e_ours, e_sn = evaluate_clip(clip)
        all_gt += e_gt
        all_ours += e_ours
        all_sn += e_sn
        rows.append("%s  gt=%.2f  ours=%.2f  soccernet=%.2f"
                     % (clip, median(e_gt), median(e_ours), median(e_sn)))
        print(rows[-1])

    lines = [
        "",
        "=== experiment 1: player localization error (metres) ===",
        summary("ours (gt lines)", all_gt),
        summary("ours (soccernet inputs)", all_ours),
        summary("soccernet", all_sn),
    ]
    print("\n".join(lines))

    report = out_dir / "experiment1_accuracy.txt"
    with report.open("w") as fp:
        fp.write("per-clip median error (metres)\n")
        fp.write("\n".join(rows) + "\n")
        fp.write("\n".join(lines) + "\n")
    print("wrote", report)


if __name__ == "__main__":
    main()
