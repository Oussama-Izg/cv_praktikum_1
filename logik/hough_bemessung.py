from itertools import combinations, product

import cv2
import numpy as np

from .models import DimensionResult, LineDetection


def _unit_vector(angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32)


def _line_frame(angle_deg, offset):
    direction = _unit_vector(angle_deg)
    return direction, _unit_vector(angle_deg + 90) * offset


def _as_points(segment):
    x1, y1, x2, y2 = segment
    return np.array([[x1, y1], [x2, y2]], dtype=np.float32)


def _rounded_segment(start, end):
    return tuple(map(int, np.round(np.concatenate((start, end)))))


def angle_distance(first_angle, second_angle):
    difference = abs(first_angle - second_angle)
    return min(difference, 180 - difference)


def line_candidate_from_points(points):
    x1, y1, x2, y2 = map(int, points)
    angle = ((np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 90) % 180) - 90
    midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    offset = midpoint @ _unit_vector(angle + 90)
    return (np.hypot(x2 - x1, y2 - y1), angle, offset, (x1, y1, x2, y2))


def find_hough_line_candidates(edges, min_line_length_ratio, threshold, max_line_gap_ratio):
    height, width = edges.shape[:2]
    diagonal = np.hypot(width, height)
    min_line_length = int(diagonal * min_line_length_ratio)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=int(diagonal * max_line_gap_ratio),
    )
    candidates = []
    if lines is not None:
        for points in lines[:, 0]:
            candidate = line_candidate_from_points(points)
            if candidate[0] >= min_line_length:
                candidates.append(candidate)

    return lines, sorted(candidates, reverse=True, key=lambda item: item[0])


def _point_to_segment_distance(point, start, end):
    segment = end - start
    length_sq = segment @ segment
    if length_sq == 0:
        return np.linalg.norm(point - start)
    t = np.clip(((point - start) @ segment) / length_sq, 0, 1)
    return np.linalg.norm(point - (start + t * segment))


def _orientation(a, b, c):
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])


def _segments_intersect(first_points, second_points):
    a, b = _as_points(first_points)
    c, d = _as_points(second_points)
    return (
        _orientation(a, b, c) * _orientation(a, b, d) <= 0
        and _orientation(c, d, a) * _orientation(c, d, b) <= 0
    )


def segment_distance(first_points, second_points):
    if _segments_intersect(first_points, second_points):
        return 0.0

    a, b = _as_points(first_points)
    c, d = _as_points(second_points)
    return min(
        _point_to_segment_distance(point, start, end)
        for point, start, end in ((a, c, d), (b, c, d), (c, a, b), (d, a, b))
    )


def _projection_interval(line, angle, offset):
    direction, line_point = _line_frame(angle, offset)
    return sorted((_as_points(line[3]) - line_point) @ direction)


def _overlap_ratio(first_interval, second_interval):
    first_start, first_end = first_interval
    second_start, second_end = second_interval
    overlap = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    return overlap / max(min(first_end - first_start, second_end - second_start), 1.0)


def _lines_can_merge(
    first,
    second,
    angle_tolerance_deg,
    distance_tolerance_px,
    strict_distance_tolerance_px,
    max_overlap_ratio,
):
    if angle_distance(first[1], second[1]) > angle_tolerance_deg:
        return False

    offset_distance = abs(first[2] - second[2])
    if offset_distance <= strict_distance_tolerance_px:
        return True
    if offset_distance > distance_tolerance_px:
        return False

    return (
        _overlap_ratio(
            _projection_interval(first, first[1], first[2]),
            _projection_interval(second, first[1], first[2]),
        )
        <= max_overlap_ratio
    )


def _merge_line_group(group, angle_tolerance_deg, distance_tolerance_px):
    base_line = max(group, key=lambda item: item[0])
    _, angle, offset, _ = base_line
    direction, line_point = _line_frame(angle, offset)
    matched = [
        line
        for line in group
        if angle_distance(angle, line[1]) <= angle_tolerance_deg
        and abs(offset - line[2]) <= distance_tolerance_px
    ]
    projections = np.concatenate([(_as_points(line[3]) - line_point) @ direction for line in matched])
    segment = _rounded_segment(
        line_point + direction * projections.min(),
        line_point + direction * projections.max(),
    )
    return (np.hypot(segment[2] - segment[0], segment[3] - segment[1]), angle, offset, segment)


def merge_collinear_line_candidates(
    line_candidates,
    angle_tolerance_deg,
    distance_tolerance_px,
    strict_distance_tolerance_px=8,
    max_overlap_ratio=0.25,
):
    groups = []
    for candidate in line_candidates:
        for group in groups:
            if any(
                _lines_can_merge(
                    candidate,
                    line,
                    angle_tolerance_deg,
                    distance_tolerance_px,
                    strict_distance_tolerance_px,
                    max_overlap_ratio,
                )
                for line in group
            ):
                group.append(candidate)
                break
        else:
            groups.append([candidate])

    merged = [_merge_line_group(group, angle_tolerance_deg, distance_tolerance_px) for group in groups]
    return sorted(merged, reverse=True, key=lambda item: item[0])


def line_intersection(first_points, second_points):
    x1, y1, x2, y2 = map(float, first_points)
    x3, y3, x4, y4 = map(float, second_points)
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None

    first_det = x1 * y2 - y1 * x2
    second_det = x3 * y4 - y3 * x4
    return np.array(
        [
            (first_det * (x3 - x4) - (x1 - x2) * second_det) / denominator,
            (first_det * (y3 - y4) - (y1 - y2) * second_det) / denominator,
        ],
        dtype=np.float32,
    )


def _extend_to_point(line, point):
    _, angle, offset, points = line
    endpoints = _as_points(points)
    index = 0 if np.linalg.norm(point - endpoints[0]) < np.linalg.norm(point - endpoints[1]) else 1
    old_endpoint = endpoints[index].copy()
    endpoints[index] = point
    segment = _rounded_segment(endpoints[0], endpoints[1])
    extension = _rounded_segment(old_endpoint, point)
    return (np.hypot(segment[2] - segment[0], segment[3] - segment[1]), angle, offset, segment), extension


def _extend_pair_to_intersection(pair):
    if len(pair) < 2 or _segments_intersect(pair[0][3], pair[1][3]):
        return list(pair), []

    intersection = line_intersection(pair[0][3], pair[1][3])
    if intersection is None:
        return list(pair), []

    extended = [_extend_to_point(line, intersection) for line in pair]
    return [edge for edge, _ in extended], [extension for _, extension in extended]


def _accept_extensions(pair, extended_edges, extension_segments, max_length_ratio):
    edges = []
    extensions = []
    for original, edge, extension in zip(pair, extended_edges, extension_segments):
        if edge[0] <= original[0] * max_length_ratio:
            edges.append(edge)
            extensions.append(extension)
        else:
            edges.append(original)
    return edges, extensions


def select_right_angle_edges(line_candidates, right_angle_tolerance_deg, pair_count, max_length_ratio):
    scored_pairs = []
    for first, second in combinations(line_candidates, 2):
        error = abs(angle_distance(first[1], second[1]) - 90)
        if error <= right_angle_tolerance_deg:
            score = (-(first[0] + second[0]), error, segment_distance(first[3], second[3]))
            scored_pairs.append((score, (first, second)))

    if not scored_pairs:
        return [], [], [], []

    selected_pairs = [pair for _, pair in sorted(scored_pairs, key=lambda item: item[0])[:pair_count]]
    best_error = min(abs(angle_distance(first[1], second[1]) - 90) for first, second in selected_pairs)
    accepted_edges = []
    accepted_extensions = []
    longest_edges = []
    longest_extensions = []
    longest_score = None

    for pair in selected_pairs:
        extended_edges, extension_segments = _extend_pair_to_intersection(pair)
        edges, extensions = _accept_extensions(pair, extended_edges, extension_segments, max_length_ratio)
        accepted_edges.extend(edges)
        accepted_extensions.extend(extensions)

        pair_error = abs(angle_distance(pair[0][1], pair[1][1]) - 90)
        pair_score = max((edge[0] for edge in extended_edges), default=0.0)
        if pair_error <= best_error + 0.5 and (longest_score is None or pair_score > longest_score):
            longest_score = pair_score
            longest_edges = extended_edges
            longest_extensions = extension_segments

    return accepted_edges, longest_edges, accepted_extensions, longest_extensions


def perspective_corrected_vector_length(dx, dy, selected_circle):
    raw_length = float(np.hypot(dx, dy))
    if selected_circle is None or selected_circle.ellipse is None:
        return raw_length

    _, axes, angle = selected_circle.ellipse
    axis_a, axis_b = axes
    major_axis = max(axis_a, axis_b)
    minor_axis = min(axis_a, axis_b)
    if major_axis <= 0 or minor_axis <= 0:
        return raw_length

    major_direction = _unit_vector(angle if axis_a >= axis_b else angle + 90)
    minor_direction = np.array([-major_direction[1], major_direction[0]], dtype=np.float32)
    vector = np.array([dx, dy], dtype=np.float32)
    return float(np.hypot(vector @ major_direction, (vector @ minor_direction) / (minor_axis / major_axis)))


def _corrected_line_length(line, selected_circle):
    length, _, _, (x1, y1, x2, y2) = line
    if selected_circle is None:
        return length
    return perspective_corrected_vector_length(x2 - x1, y2 - y1, selected_circle)


def calculate_dimensions(outer_edges, pixels_per_mm, angle_tolerance_deg, selected_circle=None):
    if not outer_edges:
        return float("nan"), float("nan"), float("nan")

    measurements = []
    for line in outer_edges:
        length_px = _corrected_line_length(line, selected_circle)
        measurements.append((length_px, length_px / pixels_per_mm, line[1], line[2]))
    measurements.sort(reverse=True, key=lambda item: item[0])

    if len(measurements) == 1:
        return measurements[0][1], float("nan"), float("nan")

    angle_difference = angle_distance(measurements[0][2], measurements[1][2])
    if angle_difference > angle_tolerance_deg:
        return measurements[0][1], measurements[1][1], angle_difference

    width_px = abs(measurements[0][3] - measurements[1][3])
    if selected_circle is not None:
        normal = _unit_vector(measurements[0][2] + 90)
        width_px = perspective_corrected_vector_length(normal[0] * width_px, normal[1] * width_px, selected_circle)
    return measurements[0][1], width_px / pixels_per_mm, angle_difference


def _line_length_mm(line, coin_detection):
    return _corrected_line_length(line, coin_detection.selected_circle) / coin_detection.pixels_per_mm


def _matching_candidates(candidates, edge, angle_tolerance_deg, offset_tolerance_px):
    return [
        candidate
        for candidate in candidates
        if angle_distance(candidate[1], edge[1]) <= angle_tolerance_deg
        and abs(candidate[2] - edge[2]) <= offset_tolerance_px
    ]


def estimate_recovered_short_arm_width(line_detection, coin_detection, config, length_mm, width_mm):
    if (
        not config.get("SHORT_ARM_RECOVERY_ENABLED", False)
        or len(line_detection.outer_edges) < 2
        or not line_detection.raw_line_candidates
        or not np.isfinite(length_mm)
        or length_mm <= 0
        or not np.isfinite(width_mm)
        or width_mm / length_mm >= config.get("SHORT_ARM_RECOVERY_MIN_RATIO", 0.24)
    ):
        return width_mm

    long_edge, short_edge = sorted(
        line_detection.outer_edges,
        reverse=True,
        key=lambda line: _line_length_mm(line, coin_detection),
    )[:2]
    angle_tolerance = config.get("SHORT_ARM_RECOVERY_ANGLE_TOLERANCE_DEG", 8)
    offset_tolerance = config.get("SHORT_ARM_RECOVERY_OFFSET_TOLERANCE_PX", 80)
    max_width_mm = length_mm * config.get("SHORT_ARM_RECOVERY_MAX_RATIO", 0.30)
    min_width_mm = width_mm + config.get("SHORT_ARM_RECOVERY_MIN_IMPROVEMENT_MM", 1.0)
    long_candidates = _matching_candidates(
        line_detection.raw_line_candidates,
        long_edge,
        angle_tolerance,
        offset_tolerance,
    )
    short_candidates = _matching_candidates(
        line_detection.raw_line_candidates,
        short_edge,
        angle_tolerance,
        offset_tolerance,
    )

    recovered_width_mm = width_mm
    for long_candidate, short_candidate in product(long_candidates, short_candidates):
        intersection = line_intersection(long_candidate[3], short_candidate[3])
        if intersection is None:
            continue

        endpoints = _as_points(short_candidate[3])
        far_endpoint = endpoints[np.argmax(np.linalg.norm(endpoints - intersection, axis=1))]
        vector = far_endpoint - intersection
        candidate_width_mm = (
            perspective_corrected_vector_length(vector[0], vector[1], coin_detection.selected_circle)
            / coin_detection.pixels_per_mm
        )
        if min_width_mm < candidate_width_mm <= max_width_mm:
            recovered_width_mm = max(recovered_width_mm, candidate_width_mm)

    return recovered_width_mm


def hough_line(preprocessing, config):
    lines, raw_candidates = find_hough_line_candidates(
        preprocessing.edges,
        config["HOUGH_MIN_LINE_LENGTH_RATIO"],
        config["HOUGH_THRESHOLD"],
        config["HOUGH_MAX_LINE_GAP_RATIO"],
    )
    line_candidates = merge_collinear_line_candidates(
        raw_candidates,
        config["MERGE_LINE_ANGLE_TOLERANCE_DEG"],
        config["MERGE_LINE_DISTANCE_TOLERANCE_PX"],
        config["MERGE_LINE_STRICT_DISTANCE_TOLERANCE_PX"],
        config["MERGE_LINE_MAX_OVERLAP_RATIO"],
    )

    height, width = preprocessing.edges.shape[:2]
    min_length = np.hypot(width, height) * config.get("MERGED_LINE_MIN_LENGTH_RATIO", 0.0)
    line_candidates = [line for line in line_candidates if line[0] >= min_length]
    (
        best_edges,
        longest_edges,
        best_extensions,
        longest_extensions,
    ) = select_right_angle_edges(
        line_candidates,
        config["RIGHT_ANGLE_TOLERANCE_DEG"],
        config.get("RIGHT_ANGLE_EXTENSION_PAIR_COUNT", 3),
        config.get("MAX_ACCEPTED_EXTENSION_LENGTH_RATIO", 1.10),
    )
    outer_edges = longest_edges or best_edges

    return LineDetection(
        lines=lines,
        line_candidates=line_candidates,
        outer_edges=outer_edges,
        edge_distance_px=segment_distance(outer_edges[0][3], outer_edges[1][3]) if len(outer_edges) >= 2 else float("nan"),
        method="hough",
        display_edges=line_candidates,
        raw_line_candidates=raw_candidates,
        best_right_angle_edges=best_edges,
        longest_right_angle_edges=longest_edges,
        best_extension_segments=best_extensions,
        longest_extension_segments=longest_extensions,
    )


def measure_dimensions_by_hough(line_detection, coin_detection, config):
    length_mm, width_mm, angle_difference_deg = calculate_dimensions(
        line_detection.outer_edges,
        coin_detection.pixels_per_mm,
        config["ANGLE_TOLERANCE_DEG"],
        coin_detection.selected_circle,
    )
    width_mm = estimate_recovered_short_arm_width(
        line_detection,
        coin_detection,
        config,
        length_mm,
        width_mm,
    )
    return DimensionResult(
        length_mm=length_mm,
        width_mm=width_mm,
        angle_difference_deg=angle_difference_deg,
    )
