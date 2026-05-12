import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .homography import fit_homography, project_points
from .pitch import make_minimap_canvas

root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
out_dir = root / "outputs"

colors = {"left": (0, 90, 230), "right": (230, 140, 0), "ball": (0, 255, 255)}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    args = ap.parse_args()

    info, img_dir, images, pitch_per, dets_per = load_clip(args.clip)
    sample = next(iter(images.values()))
    w, h = sample["width"], sample["height"]
    fps = info.get("frame_rate", 25)

    canvas, w2p = make_minimap_canvas()
    clip_out = out_dir / args.clip
    clip_out.mkdir(parents=True, exist_ok=True)
    video_path = clip_out / "minimap.mp4"

    image_ids = sorted(images, key=lambda k: images[k]["file_name"])
    writer = None

    for iid in image_ids:
        frame = cv2.imread(str(img_dir / images[iid]["file_name"]))
        if frame is None:
            continue

        g = fit_homography(pitch_per.get(iid, {}), w, h)
        dets = dets_per.get(iid, [])
        for d in dets:
            d["world_xy"] = None
        if g is not None and dets:
            feet = np.array([d["foot"] for d in dets])
            world = project_points(g, feet)
            for d, p in zip(dets, world):
                if np.all(np.isfinite(p)):
                    d["world_xy"] = (float(p[0]), float(p[1]))

        side = stack(draw_broadcast(frame, dets), draw_minimap(canvas, w2p, dets))
        if writer is None:
            wh, ww = side.shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, wh))
        writer.write(side)

    if writer:
        writer.release()


if __name__ == "__main__":
    main()
