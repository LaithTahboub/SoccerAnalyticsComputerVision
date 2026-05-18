import numpy as np

# each analytic is a small class with:
#   name           a label for the printout
#   update(dets)   called once per frame with that frame's detections
#   result()       called at the end, returns a string to print
# to add a new analytic: write such a class, then add it to make_analytics().


def players(dets, team=None):
    out = []
    for d in dets:
        if d["role"] == "ball" or d.get("world_xy") is None:
            continue
        if team is not None and d.get("team") != team:
            continue
        out.append(d)
    return out


class Possession:
    name = "ball possession"

    def __init__(self, max_dist=3.0):
        self.max_dist = max_dist
        self.counts = {"left": 0, "right": 0}

    def update(self, dets):
        ball = next((d for d in dets if d["role"] == "ball"), None)
        if ball is None or ball.get("world_xy") is None:
            return
        bx, by = ball["world_xy"]
        owner = None
        best = self.max_dist
        for d in players(dets):
            x, y = d["world_xy"]
            dist = np.hypot(x - bx, y - by)
            if dist < best:
                best = dist
                owner = d
        if owner is not None and owner.get("team") in self.counts:
            self.counts[owner["team"]] += 1

    def result(self):
        total = self.counts["left"] + self.counts["right"]
        if total == 0:
            return "no data"
        left = 100.0 * self.counts["left"] / total
        return "left %.0f%%  right %.0f%%" % (left, 100.0 - left)


class TeamSpeed:
    name = "team average speed"

    def __init__(self, fps=25):
        self.fps = fps
        self.prev = {}
        self.speeds = {"left": [], "right": []}

    def update(self, dets):
        for d in players(dets):
            tid = d.get("track_id")
            team = d.get("team")
            xy = d["world_xy"]
            if tid is not None and tid in self.prev and team in self.speeds:
                px, py = self.prev[tid]
                self.speeds[team].append(np.hypot(xy[0] - px, xy[1] - py) * self.fps)
            if tid is not None:
                self.prev[tid] = xy

    def result(self):
        out = []
        for team in ("left", "right"):
            s = self.speeds[team]
            out.append("%s %.2f m/s" % (team, sum(s) / len(s) if s else 0.0))
        return "  ".join(out)


class Compactness:
    name = "team compactness"

    def __init__(self):
        self.spread = {"left": [], "right": []}

    def update(self, dets):
        for team in ("left", "right"):
            pts = [d["world_xy"] for d in players(dets, team) if d["role"] == "player"]
            if len(pts) < 2:
                continue
            pts = np.array(pts)
            center = pts.mean(axis=0)
            self.spread[team].append(np.hypot(*(pts - center).T).mean())

    def result(self):
        out = []
        for team in ("left", "right"):
            s = self.spread[team]
            out.append("%s %.1f m" % (team, sum(s) / len(s) if s else 0.0))
        return "  ".join(out)


def make_analytics():
    return [Possession(), TeamSpeed(), Compactness()]


def print_analytics(analytics):
    print("--- analytics ---")
    for a in analytics:
        print("%-22s %s" % (a.name, a.result()))
