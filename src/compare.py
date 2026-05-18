import argparse
import json

import cv2
import numpy as np

from .pipeline import (compute_minimap, draw_minimap, build_ball_track,
                       smooth_tracks, project_dets, out_dir)
from .homography import fit_homography, smooth_homography_seq
from .pitch import make_minimap_canvas

# side by side comparison video for one clip:
#   row of minimaps:  ours (gt inputs) | ours (soccernet inputs) | soccernet
#   the source match video below
# the first two panels both run our homography pipeline; they differ only in
# whether it is fed the ground truth pitch lines/boxes or soccernet's detected
# ones. the third panel is soccernet's own predicted positions. soccernet does
# not place the ball on its minimap, so that panel shows players only.


def label(img, text):
    bar = np.full((30, img.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def soccernet_frames(sn, image_ids):
    # soccernet's own predicted positions; the ball is dropped because the
    # soccernet baseline does not place it on the minimap
    out = []
    for iid in image_ids:
        dets = []
        for d in sn.get(str(iid), {}).get("dets", []):
            if d["role"] == "ball" or d["pitch"] is None:
                continue
            dets.append({"role": d["role"], "team": d["team"],
                         "world_xy": tuple(d["pitch"])})
        out.append(dets)
    return out


def ours_on_soccernet(sn, image_ids, w, h):
    # our homography pipeline fed soccernet's detected lines and boxes
    raw_g = [fit_homography(sn.get(str(iid), {}).get("lines", {}), w, h)
             for iid in image_ids]
    smoothed_g = smooth_homography_seq(raw_g, w, h)

    dets_per = {}
    for iid in image_ids:
        dets = []
        for d in sn.get(str(iid), {}).get("dets", []):
            dets.append({"track_id": d["track_id"], "role": d["role"],
                         "team": d["team"], "foot": tuple(d["foot"])})
        dets_per[iid] = dets

    ball_track = build_ball_track(image_ids, dets_per, smoothed_g)

    track_xy = {}
    frames = []
    for idx, iid in enumerate(image_ids):
        dets = dets_per[iid]
        project_dets(dets, smoothed_g[idx])
        smooth_tracks(dets, track_xy)
        for d in dets:
            if d["role"] == "ball":
                d["world_xy"] = ball_track[idx]
        frames.append(dets)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    args = ap.parse_args()
    clip = args.clip

    # panel 1: our homography pipeline on the ground truth lines and boxes
    info, img_dir, images, image_ids, ours_gt = compute_minimap(clip)
    sample = next(iter(images.values()))
    w, h = sample["width"], sample["height"]
    fps = info.get("frame_rate", 25)

    sn_path = out_dir / clip / "soccernet.json"
    if not sn_path.exists():
        print("no outputs/%s/soccernet.json - run extract_soccernet.py first" % clip)
        return
    sn = json.loads(sn_path.read_text())

    panels = [
        ("ours (gt inputs)", ours_gt),
        ("ours (soccernet inputs)", ours_on_soccernet(sn, image_ids, w, h)),
        ("soccernet", soccernet_frames(sn, image_ids)),
    ]

    canvas, w2p = make_minimap_canvas(width_px=620)
    row_w = canvas.shape[1] * len(panels)
    video_h = int(h * row_w / w)
    out_path = out_dir / clip / "comparison.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (row_w, (canvas.shape[0] + 30) + video_h))

    for i, iid in enumerate(image_ids):
        row = np.hstack([label(draw_minimap(canvas, w2p, src[i]), title)
                         for title, src in panels])
        frame = cv2.imread(str(img_dir / images[iid]["file_name"]))
        if frame is None:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        video = cv2.resize(frame, (row_w, video_h))
        writer.write(np.vstack([row, video]))

    writer.release()
    print("wrote", out_path)


if __name__ == "__main__":
    main()
