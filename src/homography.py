import cv2
import numpy as np

from .pitch import center_circle_r, world_lines_straight


min_system_rows = 8
min_circle_points = 5
min_line_points = 2
ransac_sample_size = 12
ransac_seed = 0
homography_eps = 1e-12
intersection_eps = 1e-9
bad_residual = 1e9
min_world_span_x = 25.0
min_world_span_y = 15.0
max_world_center_x = 70.0
max_world_center_y = 50.0


def make_line_constraint(image_xy, world_line):
    return {
        "kind": "line",
        "image_xy": image_xy,
        "world_line": world_line,
    }


def make_point_constraint(image_xy, world_xy):
    return {
        "kind": "point",
        "image_xy": image_xy,
        "world_xy": world_xy,
    }


def annotation_points_to_image_points(points, img_w, img_h):
    image_points = []
    for point in points:
        image_x = point["x"] * img_w
        image_y = point["y"] * img_h
        image_points.append((image_x, image_y))
    return np.array(image_points, dtype=np.float32)


def named_points(pitch_lines, name, img_w, img_h):
    if name not in pitch_lines:
        return np.zeros((0, 2), dtype=np.float32)
    return annotation_points_to_image_points(pitch_lines[name], img_w, img_h)


def get_pitch_line_points(pitch_lines, img_w, img_h):
    pitch_line_points = {}
    for name, points in pitch_lines.items():
        if name not in world_lines_straight:
            continue
        pitch_line_points[name] = annotation_points_to_image_points(points, img_w, img_h)
    return pitch_line_points


def fit_central_circle(circle_points):
    if len(circle_points) < min_circle_points:
        return None
    return cv2.fitEllipse(circle_points)


def ellipse_landmarks(ellipse):
    center_xy, axes, angle_deg = ellipse
    center_x, center_y = center_xy
    axis_1, axis_2 = axes
    angle_rad = np.deg2rad(angle_deg)

    axis_1_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    axis_2_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    center = np.array([center_x, center_y], dtype=np.float64)

    # longer image axis maps to world x
    if axis_1 >= axis_2:
        major_vector = axis_1_dir * (axis_1 / 2.0)
        minor_vector = axis_2_dir * (axis_2 / 2.0)
    else:
        major_vector = axis_2_dir * (axis_2 / 2.0)
        minor_vector = axis_1_dir * (axis_1 / 2.0)

    major_points = [center - major_vector, center + major_vector]
    minor_points = [center - minor_vector, center + minor_vector]

    major_points = sorted(major_points, key=lambda point: point[0])
    minor_points = sorted(minor_points, key=lambda point: point[1])
    return center, major_points, minor_points


def circle_correspondences(pitch_lines, img_w, img_h):
    # rough point matches from the center circle
    # only needed when straight lines alone are ambiguous
    circle_points = named_points(pitch_lines, "Circle central", img_w, img_h)
    ellipse = fit_central_circle(circle_points)
    if ellipse is None:
        return []

    center, major_points, minor_points = ellipse_landmarks(ellipse)
    radius = center_circle_r

    constraints = []
    constraints.append(make_point_constraint(tuple(center), (0.0, 0.0)))
    constraints.append(make_point_constraint(tuple(major_points[0]), (-radius, 0.0)))
    constraints.append(make_point_constraint(tuple(major_points[1]), (radius, 0.0)))
    constraints.append(make_point_constraint(tuple(minor_points[0]), (0.0, -radius)))
    constraints.append(make_point_constraint(tuple(minor_points[1]), (0.0, radius)))
    return constraints


def fit_line_to_points(points):
    vx, vy, x0, y0 = cv2.fitLine(
        points.astype(np.float32),
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).flatten()

    line_a = vy
    line_b = -vx
    line_c = -vy * x0 + vx * y0
    line_norm = np.hypot(line_a, line_b)
    return line_a / line_norm, line_b / line_norm, line_c / line_norm


def line_ellipse_intersections(line, ellipse):
    line_a, line_b, line_c = line
    center_xy, axes, angle_deg = ellipse
    center_x, center_y = center_xy
    axis_1, axis_2 = axes

    radius_a = axis_1 / 2.0
    radius_b = axis_2 / 2.0
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # plug the rotated ellipse into ax + by + c = 0
    major_term = line_a * radius_a * cos_angle + line_b * radius_a * sin_angle
    minor_term = -line_a * radius_b * sin_angle + line_b * radius_b * cos_angle
    center_term = line_a * center_x + line_b * center_y + line_c

    amplitude = np.hypot(major_term, minor_term)
    if amplitude < intersection_eps:
        return None

    ratio = -center_term / amplitude
    if abs(ratio) > 1.0:
        return None

    base_angle = np.arctan2(minor_term, major_term)
    delta_angle = np.arccos(ratio)

    intersections = []
    for angle in [base_angle - delta_angle, base_angle + delta_angle]:
        image_x = center_x + radius_a * np.cos(angle) * cos_angle
        image_x = image_x - radius_b * np.sin(angle) * sin_angle

        image_y = center_y + radius_a * np.cos(angle) * sin_angle
        image_y = image_y + radius_b * np.sin(angle) * cos_angle

        intersections.append((image_x, image_y))
    return intersections


def circle_line_points(pitch_lines, img_w, img_h):
    # exact point matches from circle and halfway line crossings
    circle_points = named_points(pitch_lines, "Circle central", img_w, img_h)
    middle_line_points = named_points(pitch_lines, "Middle line", img_w, img_h)
    ellipse = fit_central_circle(circle_points)

    if ellipse is None:
        return []
    if len(middle_line_points) < min_line_points:
        return []

    middle_image_line = fit_line_to_points(middle_line_points)
    intersections = line_ellipse_intersections(middle_image_line, ellipse)
    if intersections is None:
        return []

    top_point, bottom_point = sorted(intersections, key=lambda point: point[1])

    constraints = []
    constraints.append(make_point_constraint(tuple(top_point), (0.0, -center_circle_r)))
    constraints.append(make_point_constraint(tuple(bottom_point), (0.0, center_circle_r)))
    return constraints


def world_line(name):
    world_start, world_end = world_lines_straight[name]
    start_h = np.array([world_start[0], world_start[1], 1.0])
    end_h = np.array([world_end[0], world_end[1], 1.0])
    line = np.cross(start_h, end_h)
    line = line / np.hypot(line[0], line[1])
    return tuple(line.tolist())


def collect_constraints(pitch_lines, img_w, img_h):
    constraints = []
    pitch_line_points = get_pitch_line_points(pitch_lines, img_w, img_h)

    for line_name, image_points in pitch_line_points.items():
        target_world_line = world_line(line_name)
        for image_x, image_y in image_points:
            image_xy = (float(image_x), float(image_y))
            constraints.append(make_line_constraint(image_xy, target_world_line))

    exact_circle_points = circle_line_points(pitch_lines, img_w, img_h)
    constraints.extend(exact_circle_points)

    if len(pitch_line_points) < 3:
        rough_circle_points = circle_correspondences(pitch_lines, img_w, img_h)
        constraints.extend(rough_circle_points)

    return constraints


def constraint_rows(constraint, image_scale):
    image_x, image_y = constraint["image_xy"]
    scaled_x = image_x / image_scale
    scaled_y = image_y / image_scale

    # point constraints give 2 rows
    # line constraints give 1 row
    if constraint["kind"] == "line":
        line_a, line_b, line_c = constraint["world_line"]
        return [[
            line_a * scaled_x,
            line_a * scaled_y,
            line_a,
            line_b * scaled_x,
            line_b * scaled_y,
            line_b,
            line_c * scaled_x,
            line_c * scaled_y,
            line_c,
        ]]

    world_x, world_y = constraint["world_xy"]

    row_1 = [
        0.0,
        0.0,
        0.0,
        scaled_x,
        scaled_y,
        1.0,
        -world_y * scaled_x,
        -world_y * scaled_y,
        -world_y,
    ]
    row_2 = [
        -scaled_x,
        -scaled_y,
        -1.0,
        0.0,
        0.0,
        0.0,
        world_x * scaled_x,
        world_x * scaled_y,
        world_x,
    ]
    return [row_1, row_2]


def solve_homography(constraints, image_scale):
    rows = []
    for constraint in constraints:
        rows.extend(constraint_rows(constraint, image_scale))

    if len(rows) < min_system_rows:
        return None

    design_matrix = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(design_matrix)

    # rows were built with scaled image coordinates
    # undo that scaling after the solve
    undo_scale = np.array([
        [1.0 / image_scale, 0.0, 0.0],
        [0.0, 1.0 / image_scale, 0.0],
        [0.0, 0.0, 1.0],
    ])
    homography = vt[-1].reshape(3, 3) @ undo_scale

    if abs(homography[2, 2]) < homography_eps:
        return None
    return homography / homography[2, 2]


def residual(homography, constraint):
    image_x, image_y = constraint["image_xy"]
    world_h = homography @ np.array([image_x, image_y, 1.0])
    if abs(world_h[2]) < homography_eps:
        return bad_residual

    world_x = world_h[0] / world_h[2]
    world_y = world_h[1] / world_h[2]

    if constraint["kind"] == "line":
        line_a, line_b, line_c = constraint["world_line"]
        return abs(line_a * world_x + line_b * world_y + line_c)

    expected_world_x, expected_world_y = constraint["world_xy"]
    return np.hypot(world_x - expected_world_x, world_y - expected_world_y)


def find_inliers(homography, constraints, thresh):
    inliers = []
    for constraint in constraints:
        if residual(homography, constraint) < thresh:
            inliers.append(constraint)
    return inliers


def is_plausible_homography(homography, img_w, img_h):
    # reject collapsed fits or fits far off the pitch
    image_corners = np.array([
        [0.0, 0.0],
        [img_w, 0.0],
        [img_w, img_h],
        [0.0, img_h],
    ])
    world_corners = project_points(homography, image_corners)
    if not np.all(np.isfinite(world_corners)):
        return False

    world_span = world_corners.max(axis=0) - world_corners.min(axis=0)
    if world_span[0] < min_world_span_x:
        return False
    if world_span[1] < min_world_span_y:
        return False

    image_center = np.array([[img_w / 2.0, img_h / 2.0]])
    world_center = project_points(homography, image_center)[0]
    if not np.all(np.isfinite(world_center)):
        return False
    if abs(world_center[0]) > max_world_center_x:
        return False
    if abs(world_center[1]) > max_world_center_y:
        return False

    return True


def fit_homography(pitch_lines, img_w, img_h, iters=200, thresh=2.0):
    image_scale = max(img_w, img_h)
    constraints = collect_constraints(pitch_lines, img_w, img_h)
    if len(constraints) < min_system_rows:
        return None

    rng = np.random.default_rng(ransac_seed)
    sample_size = min(ransac_sample_size, len(constraints))
    best_homography = None
    best_inliers = []

    for _ in range(iters):
        sample_indices = rng.choice(len(constraints), size=sample_size, replace=False)

        sample_constraints = []
        for index in sample_indices:
            sample_constraints.append(constraints[index])

        candidate = solve_homography(sample_constraints, image_scale)
        if candidate is None:
            continue

        candidate_inliers = find_inliers(candidate, constraints, thresh)
        if len(candidate_inliers) > len(best_inliers):
            best_homography = candidate
            best_inliers = candidate_inliers

    if best_homography is None:
        return None
    if len(best_inliers) < min_system_rows:
        return None

    refit = solve_homography(best_inliers, image_scale)
    if refit is None:
        homography = best_homography
    else:
        homography = refit

    if not is_plausible_homography(homography, img_w, img_h):
        return None
    return homography


def project_points(homography, pts_img):
    if homography is None:
        return np.zeros((0, 2))
    if len(pts_img) == 0:
        return np.zeros((0, 2))

    pts = np.asarray(pts_img, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2)


def probe_points(img_w, img_h):
    margin = 0.15
    return np.array([
        [img_w * margin, img_h * margin],
        [img_w * (1.0 - margin), img_h * margin],
        [img_w * (1.0 - margin), img_h * (1.0 - margin)],
        [img_w * margin, img_h * (1.0 - margin)],
    ])


def smooth_homography_seq(raw_homographies, img_w, img_h, window=2):
    # the per-frame fit occasionally spikes to a wildly wrong homography,
    # which shows up as wobble on the minimap. smooth the whole sequence by
    # median filtering a few fixed probe points: the median throws out the
    # single-frame spikes without lagging real camera pans, then each
    # homography is rebuilt from the cleaned probe points
    canon = probe_points(img_w, img_h)

    probes = []
    for homography in raw_homographies:
        if homography is None:
            probes.append(None)
            continue
        world = project_points(homography, canon)
        probes.append(world if np.all(np.isfinite(world)) else None)

    # carry the last good probe forward, then back-fill the start
    last = None
    for i in range(len(probes)):
        if probes[i] is not None:
            last = probes[i]
        else:
            probes[i] = last
    first = None
    for p in probes:
        if p is not None:
            first = p
            break
    if first is None:
        return [None] * len(raw_homographies)
    probes = [first if p is None else p for p in probes]

    smoothed = []
    for i in range(len(probes)):
        lo = max(0, i - window)
        hi = min(len(probes), i + window + 1)
        world = np.median(np.stack(probes[lo:hi]), axis=0)
        homography, _ = cv2.findHomography(canon, world)
        if homography is None:
            smoothed.append(None)
        else:
            smoothed.append(homography / homography[2, 2])
    return smoothed
