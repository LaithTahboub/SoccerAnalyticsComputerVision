import numpy as np

from .pitch import world_lines_straight


def get_pitch_line_points(pitch_lines, img_w, img_h):
    out = {}
    for name, pts in pitch_lines.items():
        if name not in world_lines_straight:
            continue
        out[name] = np.array([(p["x"] * img_w, p["y"] * img_h) for p in pts])
    return out


def fit_homography(pitch_lines, img_w, img_h):
    return None


def project_points(g, pts_img):
    if g is None or len(pts_img) == 0:
        return np.zeros((0, 2))
    p = np.column_stack([pts_img, np.ones(len(pts_img))])
    w = (g @ p.T).T
    w = w / w[:, 2:3]
    return w[:, :2]
