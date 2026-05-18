import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .analytics import make_analytics, print_analytics
from .homography import fit_homography, project_points, smooth_homography_seq
from .pitch import make_minimap_canvas

root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
out_dir = root / "outputs"

colors = {"left": (0, 90, 230), "right": (230, 140, 0), "ball": (0, 255, 255)}

max_track_step = 2.0
ball_max_speed = 2.0
ball_speed_tol = 1.5
ball_off_pitch = (58.0, 38.0)


def load_clip(name):
    clip = data_dir / name
    labels = json.loads((clip / "Labels-GameState.json").read_text())
    images = {im["image_id"]: im for im in labels["images"]}
    img_dir = clip / labels["info"]["im_dir"]
    pitch_per = {}
    dets_per = {}
    for ann in labels["annotations"]:
        iid = ann["image_id"]
        if ann.get("supercategory") == "pitch":
            pitch_per[iid] = ann.get("lines", {})
        elif "bbox_image" in ann:
            b = ann["bbox_image"]
            attrs = ann.get("attributes") or {}
            dets_per.setdefault(iid, []).append({
                "track_id": ann.get("track_id"),
                "role": attrs.get("role"),
                "team": attrs.get("team"),
                "bbox": (b["x"], b["y"], b["w"], b["h"]),
                "foot": (b["x_center"], b["y"] + b["h"]),
            })
    return labels["info"], img_dir, images, pitch_per, dets_per


def color_of(det):
    if det["role"] == "ball":
        return colors["ball"]
    return colors.get(det["team"], (200, 200, 200))


def draw_broadcast(frame, dets):
    out = frame.copy()
    for d in dets:
        x, y, w, h = d["bbox"]
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color_of(d), 2)
    return out


def draw_minimap(canvas, w2p, dets):
    img = canvas.copy()
    for d in dets:
        if d.get("world_xy") is None:
            continue
        p = w2p(*d["world_xy"])
        r = 6 if d["role"] == "ball" else 8
        cv2.circle(img, p, r, color_of(d), -1)
        cv2.circle(img, p, r, (0, 0, 0), 1)
    return img


def stack(a, b, h=720):
    aw = int(a.shape[1] * h / a.shape[0])
    bw = int(b.shape[1] * h/ b.shape[0])
    return np.hstack([cv2.resize(a, (aw, h)), cv2.resize(b, (bw, h))])


def smooth_tracks(dets, track_xy, beta=0.4):
    # per-track EMA plus a clamp on how far a point may move in one frame
    # the clamp rejects teleports from a bad homography or a bad detection
    for d in dets:
        if d["role"] == "ball":
            continue
        tid, p = d.get("track_id"), d.get("world_xy")
        if tid is None or p is None:
            continue
        if tid in track_xy:
            prev = track_xy[tid]
            p = (beta * p[0] + (1 - beta) * prev[0],
                 beta * p[1] + (1 - beta) * prev[1])
            dx, dy = p[0] - prev[0], p[1] - prev[1]
            step = np.hypot(dx, dy)
            if step > max_track_step:
                scale = max_track_step / step
                p = (prev[0] + dx * scale, prev[1] + dy * scale)
        track_xy[tid] = p
        d["world_xy"] = p


def project_ball(dets, g):
    ball = next((d for d in dets if d["role"] == "ball"), None)
    if ball is None or g is None:
        return None
    p = project_points(g, np.array([ball["foot"]]))[0]
    if not np.all(np.isfinite(p)):
        return None
    return (float(p[0]), float(p[1]))


def build_ball_track(image_ids, dets_per, homographies):
    # a ground homography cannot place an airborne ball: the ball image point
    # can even cross the horizon and project hundreds of metres away. but a
    # real ball never moves faster than ball_max_speed, so any frame that
    # jumps faster than that (or lands off the pitch) is a misprojection.
    # trust the rest and linearly interpolate the ground position across the
    # bad spans (a ball in flight crosses the ground at a near-constant speed)
    raw = [project_ball(dets_per.get(iid, []), g)
           for iid, g in zip(image_ids, homographies)]

    trusted = [False] * len(raw)
    last_i = None
    last_p = None
    for i, p in enumerate(raw):
        if p is None:
            continue
        if abs(p[0]) > ball_off_pitch[0] or abs(p[1]) > ball_off_pitch[1]:
            continue
        if last_i is None:
            trusted[i] = True
            last_i, last_p = i, p
            continue
        budget = (i - last_i) * ball_max_speed + ball_speed_tol
        if np.hypot(p[0] - last_p[0], p[1] - last_p[1]) <= budget:
            trusted[i] = True
            last_i, last_p = i, p

    anchors = [i for i in range(len(raw)) if trusted[i]]
    track = [None] * len(raw)
    if not anchors:
        return track

    for i in range(len(raw)):
        if trusted[i]:
            track[i] = raw[i]
        elif i < anchors[0]:
            track[i] = raw[anchors[0]]
        elif i > anchors[-1]:
            track[i] = raw[anchors[-1]]
        else:
            prev = max(a for a in anchors if a < i)
            nxt = min(a for a in anchors if a > i)
            t = (i - prev) / (nxt - prev)
            ap, an = raw[prev], raw[nxt]
            track[i] = (ap[0] + (an[0] - ap[0]) * t,
                        ap[1] + (an[1] - ap[1]) * t)

    # light EMA to take out residual jitter on the trusted frames
    smoothed = []
    prev = None
    for p in track:
        if prev is None:
            prev = p
        else:
            prev = (0.6 * p[0] + 0.4 * prev[0], 0.6 * p[1] + 0.4 * prev[1])
        smoothed.append(prev)
    return smoothed


def project_dets(dets, g):
    for d in dets:
        d["world_xy"] = None
    if g is not None and dets:
        feet = np.array([d["foot"] for d in dets])
        world = project_points(g, feet)
        for d, p in zip(dets, world):
            if np.all(np.isfinite(p)):
                d["world_xy"] = (float(p[0]), float(p[1]))


def compute_minimap(clip):
    # run the homography pipeline and return, per frame, the detections with
    # their minimap position filled in. no rendering, so it can be reused.
    info, img_dir, images, pitch_per, dets_per = load_clip(clip)
    sample = next(iter(images.values()))
    w, h = sample["width"], sample["height"]
    image_ids = sorted(images, key=lambda k: images[k]["file_name"])

    raw_g = [fit_homography(pitch_per.get(iid, {}), w, h) for iid in image_ids]
    smoothed_g = smooth_homography_seq(raw_g, w, h)
    ball_track = build_ball_track(image_ids, dets_per, smoothed_g)

    track_xy = {}
    frames = []
    for idx, (iid, g) in enumerate(zip(image_ids, smoothed_g)):
        dets = dets_per.get(iid, [])
        project_dets(dets, g)
        smooth_tracks(dets, track_xy)
        for d in dets:
            if d["role"] == "ball":
                d["world_xy"] = ball_track[idx]
        frames.append(dets)
    return info, img_dir, images, image_ids, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    args = ap.parse_args()

    info, img_dir, images, image_ids, frames = compute_minimap(args.clip)
    fps = info.get("frame_rate", 25)
    canvas, w2p = make_minimap_canvas()
    clip_out = out_dir / args.clip
    clip_out.mkdir(parents=True, exist_ok=True)
    video_path = clip_out / "minimap.mp4"

    writer = None
    analytics = make_analytics()
    for iid, dets in zip(image_ids, frames):
        frame = cv2.imread(str(img_dir / images[iid]["file_name"]))
        if frame is None:
            continue

        for a in analytics:
            a.update(dets)

        side = stack(draw_broadcast(frame, dets), draw_minimap(canvas, w2p, dets))
        if writer is None:
            wh, ww = side.shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, wh))
        writer.write(side)

    if writer:
        writer.release()

    print_analytics(analytics)


if __name__ == "__main__":
    main()
