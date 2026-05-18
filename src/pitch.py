import numpy as np
import cv2

pitch_length = 105.0
pitch_width = 68.0
center_circle_r = 9.15

world_lines_straight = {
    "Side line top": ((-52.5, -34), (52.5, -34)),
    "Side line bottom": ((-52.5, 34), (52.5, 34)),
    "Side line left": ((-52.5, -34), (-52.5, 34)),
    "Side line right": ((52.5, -34), (52.5, 34)),
    "Middle line": ((0, -34), (0, 34)),
    "Big rect. left main": ((-36, -20.16), (-36, 20.16)),
    "Big rect. left top": ((-52.5, -20.16), (-36, -20.16)),
    "Big rect. left bottom": ((-52.5, 20.16), (-36, 20.16)),
    "Big rect. right main": ((36, -20.16), (36, 20.16)),
    "Big rect. right top": ((52.5, -20.16), (36, -20.16)),
    "Big rect. right bottom": ((52.5, 20.16), (36, 20.16)),
    "Small rect. left main": ((-47, -9.16), (-47, 9.16)),
    "Small rect. left top": ((-52.5, -9.16), (-47, -9.16)),
    "Small rect. left bottom": ((-52.5, 9.16), (-47, 9.16)),
    "Small rect. right main": ((47, -9.16), (47, 9.16)),
    "Small rect. right top": ((52.5, -9.16), (47, -9.16)),
    "Small rect. right bottom": ((52.5, 9.16), (47, 9.16)),
}


def _is_vertical(endpoints):
    (x1, y1), (x2, y2) = endpoints
    return abs(x1 - x2) < abs(y1 - y2)


# pairs of pitch lines that cross at a finite point (one vertical, one horizontal)
valid_pairs = [
    (v, h)
    for v in world_lines_straight if _is_vertical(world_lines_straight[v])
    for h in world_lines_straight if not _is_vertical(world_lines_straight[h])
]


def make_minimap_canvas(width_px=900, margin_px=40):
    scale = (width_px - 2 * margin_px) / pitch_length
    height_px = int(pitch_width * scale) + 2 * margin_px
    canvas = np.full((height_px, width_px, 3), (34, 139, 34), dtype=np.uint8)

    def w2p(x, y):
        return (int(margin_px + (x + pitch_length / 2) * scale),
                int(margin_px + (y + pitch_width / 2) * scale))

    for p, q in world_lines_straight.values():
        cv2.line(canvas, w2p(*p), w2p(*q), (255, 255, 255), 2)
    cv2.circle(canvas, w2p(0, 0), int(center_circle_r * scale), (255, 255, 255), 2)

    return canvas, w2p
