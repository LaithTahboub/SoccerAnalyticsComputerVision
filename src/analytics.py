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

class Passes:
    name = "passes"

    def __init__(self, max_dist=3.0, rf=5):
        self.max_dist = max_dist
        self.rf = rf
        self.counts = {"left": 0, "right": 0}        
        self.owner = None        
        self.pending = None

    def update(self, dets):
        ball = next((d for d in dets if d["role"] == "ball"), None)
        if ball is None or ball.get("world_xy") is None:
            return
        b_x, b_y = ball["world_xy"]

        closest_player = None
        best = self.max_dist
        for d in players(dets):
            tid = d.get("track_id")
            if tid is None:
                continue
            x, y = d["world_xy"]
            dist = np.hypot(x - b_x, y - b_y)
            if dist < best:
                best = dist
                closest_player = (tid, d.get("team"))

        if closest_player is None:
            self.pending = None
            return
        if self.owner is not None and closest_player[0] == self.owner[0]:
            self.pending = None
            return

        if self.pending is not None and self.pending[0] == closest_player[0]:
            self.pending = (self.pending[0], self.pending[1], self.pending[2] + 1)
        else:
            self.pending = (closest_player[0], closest_player[1], 1)

        if self.pending[2] >= self.rf:
            prev = self.owner
            self.owner = (self.pending[0], self.pending[1])
            self.pending = None

            if prev is not None:
                prev_team_id, prev_team = prev
                new_team_id, new_team = self.owner

                if new_team_id != prev_team_id and new_team == prev_team and new_team in self.counts:
                    self.counts[new_team] += 1

    def result(self):
        return "left %d  right %d" % (self.counts["left"], self.counts["right"])
    

class DistanceTravelled:
    name = "distance covered"

    def __init__(self, max_dist_per_frame=1.0):
        self.max_dist_per_frame = max_dist_per_frame
        self.distances = {"left": 0.0, "right": 0.0}
        self.last_pos = {}

    def update(self, dets):
        for d in players(dets):
            team_id = d.get("track_id")
            team = d.get("team")
            
            if team_id is None or team not in self.distances:
                continue
                
            xy = d["world_xy"]
            if team_id in self.last_pos:
                p_x, p_y = self.last_pos[team_id]
                dist = np.hypot(xy[0] - p_x, xy[1] - p_y)                
                if dist < self.max_dist_per_frame:
                    self.distances[team] += dist
            self.last_pos[team_id] = xy

    def result(self):
        return "left %.1f m  right %.1f m" % (self.distances["left"], self.distances["right"])


class AveragePossessionRate:
    name = "average ball possession"

    def __init__(self, max_dist=3.0, fps=25, required_frames=6):
        self.max_dist = max_dist
        self.fps = fps
        self.required_frames = required_frames
        self.pos_duration = {"left": [], "right": []}
        self.owner_team = None
        self.pending_team = None
        self.owner_possession_frames = 0
        self.pending_possession_frames = 0

    def update(self, dets):
        ball = next((d for d in dets if d["role"] == "ball"), None)
        if ball is None or ball.get("world_xy") is None:
            self.pending_team = None
            if self.owner_team is not None:
                self.owner_possession_frames += 1
            return

        b_x, b_y = ball["world_xy"]
        closest_team = None
        best = self.max_dist
        for d in players(dets):
            x, y = d["world_xy"]
            dist = np.hypot(x - b_x, y - b_y)
            if dist < best:
                best = dist
                closest_team = d.get("team")

        if closest_team is None:
            self.pending_team = None
            if self.owner_team is not None:
                self.owner_possession_frames += 1
            return

        if self.owner_team == closest_team:
            self.pending_team = None
            self.owner_possession_frames += 1
            return
        
        if self.pending_team == closest_team:
            self.pending_possession_frames += 1
        else:
            self.pending_team = closest_team
            self.pending_possession_frames = 1            

        if self.pending_possession_frames >= self.required_frames:
            if self.owner_team is not None and self.owner_possession_frames > 0:
                self.pos_duration[self.owner_team].append(self.owner_possession_frames)
            self.owner_team = self.pending_team
            self.owner_possession_frames = self.pending_possession_frames
            self.pending_team = None
            self.pending_possession_frames = 0
        else:
            if self.owner_team is not None:
                self.owner_possession_frames += 1

    def result(self):
        output = []
        for team in ("left", "right"):
            pos_durations = self.pos_duration[team]
            output.append("%s %.1fs" % (team, sum(pos_durations) / len(pos_durations) / self.fps if pos_durations else 0.0))  
        return "  ".join(output)
    
class AverageBallSpeed:
    name = "average ball speed"

    def __init__(self, fps=25, max_speed=60.0, required_frames=15):
        self.fps = fps
        self.max_speed = max_speed
        self.required_frames = required_frames
        self.speeds = {"left": [], "right": []}
        self.prev_ball_xy = None
        self.owner = None
        self.pending = None
        self.pending_frames = 0
        self.max_dist = 3.0

    def update(self, dets):
        ball = next((d for d in dets if d["role"] == "ball"), None)        
        if ball is not None and ball.get("world_xy") is not None:
            b_x, b_y = ball["world_xy"]
            closest_team = None
            best = self.max_dist
            for d in players(dets):
                x, y = d["world_xy"]
                dist = np.hypot(x - b_x, y - b_y)
                if dist < best:
                    best = dist
                    closest_team = d.get("team")

            if closest_team is not None:
                if self.owner == closest_team:
                    self.pending = None
                    self.pending_frames = 0
                elif self.pending == closest_team:
                    self.pending_frames += 1

                    if self.pending_frames >= self.required_frames:
                        self.owner = self.pending
                        self.pending = None
                        self.pending_frames = 0
                else:
                    self.pending = closest_team
                    self.pending_frames = 1

        if ball is not None and ball.get("world_xy") is not None:
            curr_xy = ball["world_xy"]

            if self.owner is not None and self.prev_ball_xy is not None:
                dist = np.hypot(curr_xy[0] - self.prev_ball_xy[0], curr_xy[1] - self.prev_ball_xy[1])
                speed = dist * self.fps                
                if speed < self.max_speed:
                    self.speeds[self.owner].append(speed)
            self.prev_ball_xy = curr_xy
        else:
            self.prev_ball_xy = None

    def result(self):
        output = []
        for team in ("left", "right"):
            s = self.speeds[team]
            output.append("%s %.2f meters/second" % (team, sum(s) / len(s) if s else 0.0))
        return "  ".join(output)

def make_analytics():
    return [Possession(), TeamSpeed(), Compactness(), Passes(), DistanceTravelled(), AveragePossessionRate(), AverageBallSpeed()]


def print_analytics(analytics):
    print("--- analytics ---")
    for a in analytics:
        print("%-30s %s" % (a.name, a.result()))