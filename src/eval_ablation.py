import argparse
import json

import numpy as np

from .homography import (fit_homography, smooth_homography_seq,
                         project_points, probe_points)
from .pipeline import data_dir, out_dir, build_ball_track, ball_off_pitch

# experiment 2: ablation of the three pipeline components, over all 58 clips.
#   A  circle-degeneracy handling -> fraction of frames with a valid homography
#   B  temporal smoothing         -> per-frame jitter of fixed probe points
#   C  airborne-ball gating       -> off-pitch ball frames and ball error


def load_clip_gt(clip):
    labels = json.loads((data_dir / clip / "Labels-GameState.json").read_text())
    images = {im["image_id"]: im for im in labels["images"]}
    image_ids = sorted(images, key=lambda k: images[k]["file_name"])
    w = images[image_ids[0]]["width"]
    h = images[image_ids[0]]["height"]
    lines_per, ball_per = {}, {}
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
            ball_per[iid] = {
                "foot": (b["x_center"], b["y"] + b["h"]),
                "pitch": (bp["x_bottom_middle"], bp["y_bottom_middle"]),
            }
    return image_ids, w, h, lines_per, ball_per


def jitter(homs, canon):
    # per-frame displacement of the probe points between consecutive frames
    steps = []
    prev = None
    for g in homs:
        cur = project_points(g, canon) if g is not None else None
        if cur is not None and not np.all(np.isfinite(cur)):
            cur = None
        if cur is not None and prev is not None:
            steps.append(np.linalg.norm(cur - prev, axis=1).mean())
        prev = cur
    return steps


def off_pitch(p):
    return p is not None and (abs(p[0]) > ball_off_pitch[0]
                              or abs(p[1]) > ball_off_pitch[1])


def evaluate_clip(clip):
    image_ids, w, h, lines_per, ball_per = load_clip_gt(clip)
    canon = probe_points(w, h)

    raw_full = [fit_homography(lines_per.get(i, {}), w, h) for i in image_ids]
    raw_nc = [fit_homography({k: v for k, v in lines_per.get(i, {}).items()
                              if k != "Circle central"}, w, h)
              for i in image_ids]
    smoothed = smooth_homography_seq(raw_full, w, h)

    # A: circle-degeneracy handling
    a = {
        "frames": len(image_ids),
        "valid_full": sum(g is not None for g in raw_full),
        "valid_nc": sum(g is not None for g in raw_nc),
    }

    # B: temporal smoothing
    b = {"raw": jitter(raw_full, canon), "smoothed": jitter(smoothed, canon)}

    # C: airborne-ball gating
    ball_dets = {i: [{"role": "ball", "foot": ball_per[i]["foot"]}]
                 for i in image_ids if i in ball_per}
    gated = build_ball_track(image_ids, ball_dets, smoothed)
    raw_ball = []
    for g, iid in zip(smoothed, image_ids):
        if iid in ball_per and g is not None:
            p = project_points(g, np.array([ball_per[iid]["foot"]]))[0]
            raw_ball.append(tuple(p) if np.all(np.isfinite(p)) else None)
        else:
            raw_ball.append(None)

    c = {"raw_off": 0, "gated_off": 0, "raw_err": [], "gated_err": []}
    for idx, iid in enumerate(image_ids):
        if iid not in ball_per:
            continue
        gt = ball_per[iid]["pitch"]
        c["raw_off"] += off_pitch(raw_ball[idx])
        c["gated_off"] += off_pitch(gated[idx])
        # accuracy only where the gt ball position is itself on-pitch (sane)
        if abs(gt[0]) <= ball_off_pitch[0] and abs(gt[1]) <= ball_off_pitch[1]:
            if raw_ball[idx] is not None:
                c["raw_err"].append(np.hypot(raw_ball[idx][0] - gt[0],
                                             raw_ball[idx][1] - gt[1]))
            if gated[idx] is not None:
                c["gated_err"].append(np.hypot(gated[idx][0] - gt[0],
                                               gated[idx][1] - gt[1]))
    return a, b, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    args = ap.parse_args()
    clips = args.clips or sorted(p.name for p in out_dir.iterdir()
                                 if (p / "soccernet.json").exists())

    frames = valid_full = valid_nc = 0
    raw_steps, sm_steps = [], []
    raw_off = gated_off = 0
    raw_err, gated_err = [], []
    for clip in clips:
        a, b, c = evaluate_clip(clip)
        frames += a["frames"]
        valid_full += a["valid_full"]
        valid_nc += a["valid_nc"]
        raw_steps += b["raw"]
        sm_steps += b["smoothed"]
        raw_off += c["raw_off"]
        gated_off += c["gated_off"]
        raw_err += c["raw_err"]
        gated_err += c["gated_err"]
        print("done", clip)

    rs, ss = np.array(raw_steps), np.array(sm_steps)
    lines = [
        "",
        "=== experiment 2: ablations (%d clips, %d frames) ===" % (len(clips), frames),
        "",
        "A. circle-degeneracy handling -- frames with a valid homography",
        "   with circle handling : %d / %d  (%.1f%%)" % (
            valid_full, frames, 100 * valid_full / frames),
        "   without (ablated)    : %d / %d  (%.1f%%)" % (
            valid_nc, frames, 100 * valid_nc / frames),
        "   recovered by circle handling: %.1f%% of all frames" % (
            100 * (valid_full - valid_nc) / frames),
        "",
        "B. temporal smoothing -- per-frame probe-point displacement (metres)",
        "   raw fits (ablated)   : mean=%.3f  p99=%.2f  max=%.1f" % (
            rs.mean(), np.percentile(rs, 99), rs.max()),
        "   median-smoothed      : mean=%.3f  p99=%.2f  max=%.1f" % (
            ss.mean(), np.percentile(ss, 99), ss.max()),
        "",
        "C. airborne-ball gating",
        "   off-pitch ball frames: raw projection (ablated)=%d   gated=%d" % (
            raw_off, gated_off),
        "   ball error vs gt on sane-gt frames (metres): raw median=%.2f  gated median=%.2f" % (
            np.median(raw_err), np.median(gated_err)),
        "   ball error mean: raw=%.2f  gated=%.2f" % (
            np.mean(raw_err), np.mean(gated_err)),
    ]
    print("\n".join(lines))

    report = out_dir / "experiment2_ablation.txt"
    report.write_text("\n".join(lines) + "\n")
    print("wrote", report)


if __name__ == "__main__":
    main()
