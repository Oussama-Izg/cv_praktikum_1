from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class CircleScore:
    score: float
    edge_support: int
    edge_roundness: float
    fill_ratio: float
    circle: tuple
    ellipse: Optional[tuple] = None
    scale_diameter_px: Optional[float] = None


@dataclass
class MeasurementResult:
    coin_detection_method: str
    dimension_method: str
    coin_center: tuple
    coin_radius: int
    pixels_per_mm: float
    circle_candidates: list
    selected_circle: CircleScore
    line_count: int
    line_candidate_count: int
    outer_edges: list
    angle_difference_deg: float
    edge_distance_px: float
    length_mm: float
    width_mm: float
    coin_debug: np.ndarray
    all_lines_debug: np.ndarray
    result_debug: np.ndarray


@dataclass
class CoinDetection:
    method: str
    circles: np.ndarray
    scored_circles: list
    selected_circle: CircleScore
    coin_center: tuple
    coin_radius: int
    pixels_per_mm: float


@dataclass
class LineDetection:
    lines: np.ndarray
    line_candidates: list
    outer_edges: list
    edge_distance_px: float
    method: str = "hough"
    extension_segments: list = field(default_factory=list)
    display_edges: list = field(default_factory=list)
    raw_line_candidates: list = field(default_factory=list)
    right_angle_edges: list = field(default_factory=list)
    best_right_angle_edges: list = field(default_factory=list)
    longest_right_angle_edges: list = field(default_factory=list)
    best_extension_segments: list = field(default_factory=list)
    longest_extension_segments: list = field(default_factory=list)


@dataclass
class DimensionResult:
    length_mm: float
    width_mm: float
    angle_difference_deg: float


@dataclass
class DebugImages:
    coin_debug: np.ndarray
    all_lines_debug: np.ndarray
    result_debug: np.ndarray


def find_contour_coin_candidates(
    clean,
    min_area,
    min_axis_ratio=0.35,
    min_ellipse_fill_ratio=0.55,
    max_ellipse_fill_ratio=1.35,
):
    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    scored = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius <= 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        circle_fill_ratio = area / (np.pi * radius ** 2)
        score = float(circularity)
        edge_roundness = float(circularity)
        fill_ratio = float(circle_fill_ratio)
        ellipse = None
        scale_diameter_px = radius * 2

        if len(contour) >= 5:
            (ellipse_x, ellipse_y), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
            major_axis = max(axis_a, axis_b)
            minor_axis = min(axis_a, axis_b)
            if major_axis > 0 and minor_axis > 0:
                ellipse_area = np.pi * (major_axis / 2) * (minor_axis / 2)
                ellipse_fill_ratio = area / ellipse_area
                axis_ratio = minor_axis / major_axis
                fill_quality = max(0.0, 1.0 - abs(1.0 - ellipse_fill_ratio))
                ellipse_score = axis_ratio * fill_quality * np.sqrt(area)

                if (
                    axis_ratio >= min_axis_ratio
                    and min_ellipse_fill_ratio <= ellipse_fill_ratio <= max_ellipse_fill_ratio
                ):
                    score = float(ellipse_score)
                    edge_roundness = float(axis_ratio)
                    fill_ratio = float(ellipse_fill_ratio)
                    ellipse = (
                        (float(ellipse_x), float(ellipse_y)),
                        (float(axis_a), float(axis_b)),
                        float(angle),
                    )
                    scale_diameter_px = float(major_axis)

        scored.append(
            CircleScore(
                score=score,
                edge_support=int(area),
                edge_roundness=edge_roundness,
                fill_ratio=fill_ratio,
                circle=(int(x), int(y), int(radius)),
                ellipse=ellipse,
                scale_diameter_px=float(scale_diameter_px),
            )
        )

    return sorted(scored, reverse=True, key=lambda item: item.score)


def selected_coin_scale_diameter_px(selected, coin_scale_axis):
    scale_diameter_px = selected.scale_diameter_px
    if selected.ellipse is None:
        return scale_diameter_px

    _, axes, _ = selected.ellipse
    major_axis = max(axes)
    minor_axis = min(axes)
    scale_axis = coin_scale_axis.lower()

    if scale_axis == "major":
        return major_axis
    if scale_axis == "minor":
        return minor_axis
    if scale_axis == "mean":
        return (major_axis + minor_axis) / 2
    if scale_axis == "geometric_mean":
        return np.sqrt(major_axis * minor_axis)
    return scale_diameter_px


def select_coin(scored_circles, coin_diameter_mm, coin_scale_axis="major"):
    if not scored_circles:
        raise RuntimeError("Keine Kreis-Kandidaten gefunden")

    selected = scored_circles[0]
    coin_x, coin_y, coin_radius = selected.circle
    coin_center = (int(coin_x), int(coin_y))
    coin_radius = int(coin_radius)
    scale_diameter_px = selected_coin_scale_diameter_px(selected, coin_scale_axis) or coin_radius * 2
    pixels_per_mm = scale_diameter_px / coin_diameter_mm
    return selected, coin_center, coin_radius, pixels_per_mm


def build_coin_detection(method, circles, scored_circles, config):
    selected_circle, coin_center, coin_radius, pixels_per_mm = select_coin(
        scored_circles,
        config["COIN_DIAMETER_MM"],
        config.get("COIN_SCALE_AXIS", "major"),
    )
    return CoinDetection(
        method=method,
        circles=circles,
        scored_circles=scored_circles,
        selected_circle=selected_circle,
        coin_center=coin_center,
        coin_radius=coin_radius,
        pixels_per_mm=pixels_per_mm,
    )


def draw_coin_candidate(debug, circle_score, color, thickness):
    if circle_score.ellipse is not None:
        center, axes, angle = circle_score.ellipse
        ellipse = (
            (int(round(center[0])), int(round(center[1]))),
            (int(round(axes[0])), int(round(axes[1]))),
            float(angle),
        )
        cv2.ellipse(debug, ellipse, color, thickness)
        return

    x, y, radius = circle_score.circle
    cv2.circle(debug, (int(x), int(y)), int(radius), color, thickness)


def create_coin_debug(img, scored_circles, selected_circle):
    debug = img.copy()
    for rank, circle_score in enumerate(scored_circles, start=1):
        x, y, radius = circle_score.circle
        center = (int(x), int(y))
        label_position = (center[0] - 60, center[1] + 35)
        draw_coin_candidate(debug, circle_score, (255, 0, 0), 2)
        cv2.putText(debug, str(rank), label_position, cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 8)

    x, y, radius = selected_circle.circle
    draw_coin_candidate(debug, selected_circle, (0, 255, 255), 4)
    cv2.circle(debug, (int(x), int(y)), 5, (0, 0, 255), -1)
    return debug


def detect_coin_by_contours(preprocessing, config):
    scored_circles = find_contour_coin_candidates(
        preprocessing.clean,
        config["CONTOUR_COIN_MIN_AREA"],
        config.get("CONTOUR_COIN_MIN_AXIS_RATIO", 0.35),
        config.get("CONTOUR_COIN_MIN_ELLIPSE_FILL_RATIO", 0.55),
        config.get("CONTOUR_COIN_MAX_ELLIPSE_FILL_RATIO", 1.35),
    )
    circles = np.array([circle.circle for circle in scored_circles], dtype=int)
    return build_coin_detection("contour", circles, scored_circles, config)


def find_hough_line_candidates(edges, min_line_length_ratio, threshold, max_line_gap_ratio):
    height, width = edges.shape[:2]
    diagonal = np.hypot(width, height)
    min_line_length = int(diagonal * min_line_length_ratio)
    max_line_gap = int(diagonal * max_line_gap_ratio)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    line_candidates = []
    if lines is None:
        return None, line_candidates

    for points in lines[:, 0]:
        line_candidate = line_candidate_from_points(points)
        length = line_candidate[0]
        if length < min_line_length:
            continue
        line_candidates.append(line_candidate)

    return lines, sorted(line_candidates, reverse=True, key=lambda item: item[0])


def normalized_line_angle(x1, y1, x2, y2):
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return ((angle + 90) % 180) - 90


def line_candidate_from_points(points):
    x1, y1, x2, y2 = map(int, points)
    length = np.hypot(x2 - x1, y2 - y1)
    angle = normalized_line_angle(x1, y1, x2, y2)
    normal_angle = np.radians(angle + 90)
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2
    offset = midpoint_x * np.cos(normal_angle) + midpoint_y * np.sin(normal_angle)
    return (length, angle, offset, (x1, y1, x2, y2))


def point_to_segment_distance(point, start, end):
    point = np.array(point, dtype=np.float32)
    start = np.array(start, dtype=np.float32)
    end = np.array(end, dtype=np.float32)
    segment = end - start
    segment_length_sq = np.dot(segment, segment)

    if segment_length_sq == 0:
        return np.linalg.norm(point - start)

    t = np.dot(point - start, segment) / segment_length_sq
    t = np.clip(t, 0, 1)
    projection = start + t * segment
    return np.linalg.norm(point - projection)


def segments_intersect(first_points, second_points):
    x1, y1, x2, y2 = map(float, first_points)
    x3, y3, x4, y4 = map(float, second_points)

    def orientation(ax, ay, bx, by, cx, cy):
        return (by - ay) * (cx - bx) - (bx - ax) * (cy - by)

    o1 = orientation(x1, y1, x2, y2, x3, y3)
    o2 = orientation(x1, y1, x2, y2, x4, y4)
    o3 = orientation(x3, y3, x4, y4, x1, y1)
    o4 = orientation(x3, y3, x4, y4, x2, y2)
    return o1 * o2 <= 0 and o3 * o4 <= 0


def segment_distance(first_points, second_points):
    if segments_intersect(first_points, second_points):
        return 0.0

    x1, y1, x2, y2 = first_points
    x3, y3, x4, y4 = second_points
    return min(
        point_to_segment_distance((x1, y1), (x3, y3), (x4, y4)),
        point_to_segment_distance((x2, y2), (x3, y3), (x4, y4)),
        point_to_segment_distance((x3, y3), (x1, y1), (x2, y2)),
        point_to_segment_distance((x4, y4), (x1, y1), (x2, y2)),
    )


def angle_distance(first_angle, second_angle):
    difference = abs(first_angle - second_angle)
    return min(difference, 180 - difference)


def extend_line_with_candidates(base_line, candidates, angle_tolerance_deg, distance_tolerance_px):
    matching_segments = matching_line_segments(
        base_line,
        candidates,
        angle_tolerance_deg,
        distance_tolerance_px,
    )
    if not matching_segments:
        matching_segments = [base_line]

    _, base_angle, base_offset, _ = base_line
    direction = np.array([np.cos(np.radians(base_angle)), np.sin(np.radians(base_angle))], dtype=np.float32)
    normal = np.array([np.cos(np.radians(base_angle + 90)), np.sin(np.radians(base_angle + 90))], dtype=np.float32)
    line_point = normal * base_offset
    projected_points = []

    for _, _, _, (x1, y1, x2, y2) in matching_segments:
        projected_points.append((np.array([x1, y1], dtype=np.float32) - line_point) @ direction)
        projected_points.append((np.array([x2, y2], dtype=np.float32) - line_point) @ direction)

    if len(projected_points) < 2:
        return base_line

    start_point = line_point + direction * np.min(projected_points)
    end_point = line_point + direction * np.max(projected_points)
    x1, y1 = np.round(start_point).astype(int)
    x2, y2 = np.round(end_point).astype(int)
    extended_length = np.hypot(x2 - x1, y2 - y1)
    return (extended_length, base_angle, base_offset, (x1, y1, x2, y2))


def matching_line_segments(base_line, candidates, angle_tolerance_deg, distance_tolerance_px):
    _, base_angle, base_offset, _ = base_line
    return [
        candidate
        for candidate in candidates
        if angle_distance(base_angle, candidate[1]) <= angle_tolerance_deg
        and abs(base_offset - candidate[2]) <= distance_tolerance_px
    ]


def line_projection_interval(line, reference_angle, reference_offset):
    direction = np.array([np.cos(np.radians(reference_angle)), np.sin(np.radians(reference_angle))], dtype=np.float32)
    normal = np.array([np.cos(np.radians(reference_angle + 90)), np.sin(np.radians(reference_angle + 90))], dtype=np.float32)
    line_point = normal * reference_offset
    _, _, _, (x1, y1, x2, y2) = line
    first_projection = (np.array([x1, y1], dtype=np.float32) - line_point) @ direction
    second_projection = (np.array([x2, y2], dtype=np.float32) - line_point) @ direction
    return sorted((first_projection, second_projection))


def interval_overlap_ratio(first_interval, second_interval):
    first_start, first_end = first_interval
    second_start, second_end = second_interval
    overlap = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    shorter_length = max(min(first_end - first_start, second_end - second_start), 1.0)
    return overlap / shorter_length


def lines_can_merge(
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

    first_interval = line_projection_interval(first, first[1], first[2])
    second_interval = line_projection_interval(second, first[1], first[2])
    return interval_overlap_ratio(first_interval, second_interval) <= max_overlap_ratio


def merge_collinear_line_candidates(
    line_candidates,
    angle_tolerance_deg,
    distance_tolerance_px,
    strict_distance_tolerance_px=8,
    max_overlap_ratio=0.25,
):
    groups = []

    for candidate in line_candidates:
        matching_group = None

        for group in groups:
            if any(
                lines_can_merge(
                    candidate,
                    group_line,
                    angle_tolerance_deg,
                    distance_tolerance_px,
                    strict_distance_tolerance_px,
                    max_overlap_ratio,
                )
                for group_line in group
            ):
                matching_group = group
                break

        if matching_group is None:
            groups.append([candidate])
        else:
            matching_group.append(candidate)

    merged_lines = []
    for group in groups:
        base_line = max(group, key=lambda item: item[0])
        merged_lines.append(extend_line_with_candidates(base_line, group, angle_tolerance_deg, distance_tolerance_px))

    return sorted(merged_lines, reverse=True, key=lambda item: item[0])


def filter_lines_by_min_length_ratio(line_candidates, image_shape, min_length_ratio):
    image_height, image_width = image_shape[:2]
    min_length = np.hypot(image_width, image_height) * min_length_ratio
    return [line for line in line_candidates if line[0] >= min_length]


def line_intersection(first_points, second_points):
    x1, y1, x2, y2 = map(float, first_points)
    x3, y3, x4, y4 = map(float, second_points)
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None

    intersection_x = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator
    intersection_y = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator
    return np.array([intersection_x, intersection_y], dtype=np.float32)


def extend_segment_to_point(line, point):
    _, angle, offset, points = line
    x1, y1, x2, y2 = points
    start = np.array([x1, y1], dtype=np.float32)
    end = np.array([x2, y2], dtype=np.float32)

    original_start = start.copy()
    original_end = end.copy()

    if np.linalg.norm(point - start) < np.linalg.norm(point - end):
        start = point
        extension_start = original_start
        extension_end = start
    else:
        end = point
        extension_start = original_end
        extension_end = end

    x1, y1 = np.round(start).astype(int)
    x2, y2 = np.round(end).astype(int)
    length = np.hypot(x2 - x1, y2 - y1)
    ex1, ey1 = np.round(extension_start).astype(int)
    ex2, ey2 = np.round(extension_end).astype(int)
    extension_segment = (ex1, ey1, ex2, ey2)
    return (length, angle, offset, (x1, y1, x2, y2)), extension_segment


def extend_edges_until_they_meet(outer_edges):
    if len(outer_edges) < 2:
        return outer_edges, float("nan"), []
    if segments_intersect(outer_edges[0][3], outer_edges[1][3]):
        return outer_edges, 0.0, []

    intersection = line_intersection(outer_edges[0][3], outer_edges[1][3])
    if intersection is None:
        return outer_edges, segment_distance(outer_edges[0][3], outer_edges[1][3]), []

    first_edge, first_extension = extend_segment_to_point(outer_edges[0], intersection)
    second_edge, second_extension = extend_segment_to_point(outer_edges[1], intersection)
    return [first_edge, second_edge], 0.0, [first_extension, second_extension]


def find_right_angle_edges(line_candidates, right_angle_tolerance_deg):
    right_angle_edges = []
    seen_lines = set()

    for i, first in enumerate(line_candidates):
        for second in line_candidates[i + 1:]:
            angle_difference = angle_distance(first[1], second[1])
            right_angle_error = abs(angle_difference - 90)
            if right_angle_error > right_angle_tolerance_deg:
                continue

            for line in (first, second):
                points = tuple(int(value) for value in line[3])
                if points in seen_lines:
                    continue
                seen_lines.add(points)
                right_angle_edges.append(line)

    return right_angle_edges


def select_longest_right_angle_pairs(line_candidates, right_angle_tolerance_deg, pair_count):
    right_angle_pairs = []

    for i, first in enumerate(line_candidates):
        for second in line_candidates[i + 1:]:
            angle_difference = angle_distance(first[1], second[1])
            right_angle_error = abs(angle_difference - 90)
            if right_angle_error > right_angle_tolerance_deg:
                continue

            length_sum = first[0] + second[0]
            pair_distance = segment_distance(first[3], second[3])
            right_angle_pairs.append(((-length_sum, right_angle_error, pair_distance), (first, second)))

    right_angle_pairs = sorted(right_angle_pairs, key=lambda item: item[0])
    return [pair for _, pair in right_angle_pairs[:pair_count]]


def accept_limited_extensions(original_edges, extended_edges, max_length_ratio):
    accepted_edges = []

    for original_edge, extended_edge in zip(original_edges, extended_edges):
        if extended_edge[0] <= original_edge[0] * max_length_ratio:
            accepted_edges.append(extended_edge)
        else:
            accepted_edges.append(original_edge)

    return accepted_edges


def extend_longest_right_angle_pairs(
    line_candidates,
    right_angle_tolerance_deg,
    pair_count,
    max_length_ratio,
):
    selected_edges = []
    extension_segments = []
    longest_pair_edges = []
    longest_pair_extension_segments = []
    longest_pair_score = None
    selected_pairs = select_longest_right_angle_pairs(line_candidates, right_angle_tolerance_deg, pair_count)
    best_right_angle_error = min(
        (
            abs(angle_distance(pair[0][1], pair[1][1]) - 90)
            for pair in selected_pairs
        ),
        default=None,
    )
    right_angle_error_margin_deg = 0.5

    for pair in selected_pairs:
        extended_edges, _, pair_extension_segments = extend_edges_until_they_meet(list(pair))
        accepted_edges = accept_limited_extensions(pair, extended_edges, max_length_ratio)
        extended_pair_score = max((edge[0] for edge in extended_edges), default=0.0)
        selected_edges.extend(accepted_edges)
        extension_segments.extend(pair_extension_segments)

        right_angle_error = abs(angle_distance(pair[0][1], pair[1][1]) - 90)
        if (
            best_right_angle_error is not None
            and right_angle_error > best_right_angle_error + right_angle_error_margin_deg
        ):
            continue

        if longest_pair_score is None or extended_pair_score > longest_pair_score:
            longest_pair_score = extended_pair_score
            longest_pair_edges = extended_edges
            longest_pair_extension_segments = pair_extension_segments

    return selected_edges, longest_pair_edges, extension_segments, longest_pair_extension_segments


def perspective_corrected_vector_length(dx, dy, selected_circle):
    if selected_circle.ellipse is None:
        return float(np.hypot(dx, dy))

    _, axes, angle = selected_circle.ellipse
    axis_a, axis_b = axes
    major_axis = max(axis_a, axis_b)
    minor_axis = min(axis_a, axis_b)
    if major_axis <= 0 or minor_axis <= 0:
        return float(np.hypot(dx, dy))

    major_angle = angle if axis_a >= axis_b else angle + 90
    theta = np.radians(major_angle)
    major_direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    minor_direction = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    vector = np.array([dx, dy], dtype=np.float32)
    major_component = float(vector @ major_direction)
    minor_component = float(vector @ minor_direction)
    axis_ratio = minor_axis / major_axis
    return float(np.hypot(major_component, minor_component / axis_ratio))


def perspective_corrected_line_length(points, selected_circle):
    x1, y1, x2, y2 = points
    return perspective_corrected_vector_length(x2 - x1, y2 - y1, selected_circle)


def calculate_dimensions(outer_edges, pixels_per_mm, angle_tolerance_deg, selected_circle=None):
    if len(outer_edges) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(outer_edges) == 1:
        length = (
            perspective_corrected_line_length(outer_edges[0][3], selected_circle)
            if selected_circle is not None
            else outer_edges[0][0]
        )
        return length / pixels_per_mm, float("nan"), float("nan")

    measurements = []
    for length, angle, offset, points in outer_edges:
        corrected_length = (
            perspective_corrected_line_length(points, selected_circle)
            if selected_circle is not None
            else length
        )
        measurements.append((corrected_length, corrected_length / pixels_per_mm, angle, offset, points))

    measurements = sorted(measurements, reverse=True, key=lambda item: item[0])
    angle_difference = angle_distance(measurements[0][2], measurements[1][2])
    length_mm = measurements[0][1]

    if angle_difference <= angle_tolerance_deg:
        width_px = abs(measurements[0][3] - measurements[1][3])
        if selected_circle is not None:
            normal_angle = np.radians(measurements[0][2] + 90)
            width_px = perspective_corrected_vector_length(
                np.cos(normal_angle) * width_px,
                np.sin(normal_angle) * width_px,
                selected_circle,
            )
        width_mm = width_px / pixels_per_mm
    else:
        width_mm = measurements[1][1]

    return length_mm, width_mm, angle_difference


def corrected_length_mm_from_vector(vector, coin_detection):
    corrected_length = perspective_corrected_vector_length(
        vector[0],
        vector[1],
        coin_detection.selected_circle,
    )
    return corrected_length / coin_detection.pixels_per_mm


def corrected_line_length_mm(points, coin_detection):
    x1, y1, x2, y2 = points
    vector = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    return corrected_length_mm_from_vector(vector, coin_detection)


def estimate_recovered_short_arm_width(line_detection, coin_detection, config, length_mm, width_mm):
    if not config.get("SHORT_ARM_RECOVERY_ENABLED", False):
        return width_mm
    if len(line_detection.outer_edges) < 2 or not line_detection.raw_line_candidates:
        return width_mm
    if not np.isfinite(length_mm) or length_mm <= 0 or not np.isfinite(width_mm):
        return width_mm

    min_ratio = config.get("SHORT_ARM_RECOVERY_MIN_RATIO", 0.24)
    if width_mm / length_mm >= min_ratio:
        return width_mm

    long_edge, short_edge = sorted(
        line_detection.outer_edges,
        reverse=True,
        key=lambda item: corrected_line_length_mm(item[3], coin_detection),
    )[:2]
    angle_tolerance_deg = config.get("SHORT_ARM_RECOVERY_ANGLE_TOLERANCE_DEG", 8)
    offset_tolerance_px = config.get("SHORT_ARM_RECOVERY_OFFSET_TOLERANCE_PX", 80)
    max_ratio = config.get("SHORT_ARM_RECOVERY_MAX_RATIO", 0.30)
    max_width_mm = length_mm * max_ratio
    min_improvement_mm = config.get("SHORT_ARM_RECOVERY_MIN_IMPROVEMENT_MM", 1.0)

    long_candidates = [
        candidate
        for candidate in line_detection.raw_line_candidates
        if angle_distance(candidate[1], long_edge[1]) <= angle_tolerance_deg
        and abs(candidate[2] - long_edge[2]) <= offset_tolerance_px
    ]
    short_candidates = [
        candidate
        for candidate in line_detection.raw_line_candidates
        if angle_distance(candidate[1], short_edge[1]) <= angle_tolerance_deg
        and abs(candidate[2] - short_edge[2]) <= offset_tolerance_px
    ]

    recovered_width_mm = width_mm
    for long_candidate in long_candidates:
        for short_candidate in short_candidates:
            intersection = line_intersection(long_candidate[3], short_candidate[3])
            if intersection is None:
                continue

            x1, y1, x2, y2 = short_candidate[3]
            endpoints = (
                np.array([x1, y1], dtype=np.float32),
                np.array([x2, y2], dtype=np.float32),
            )
            far_endpoint = max(endpoints, key=lambda point: np.linalg.norm(point - intersection))
            vector = far_endpoint - intersection
            candidate_width_mm = corrected_length_mm_from_vector(vector, coin_detection)

            if candidate_width_mm <= width_mm + min_improvement_mm:
                continue
            if candidate_width_mm > max_width_mm:
                continue
            recovered_width_mm = max(recovered_width_mm, candidate_width_mm)

    return recovered_width_mm


def create_line_debug_images(
    img,
    coin_detection,
    line_detection,
):
    all_lines_debug = img.copy()
    result_debug = img.copy()

    for circle_score in coin_detection.scored_circles:
        x, y, radius = circle_score.circle
        draw_coin_candidate(all_lines_debug, circle_score, (255, 0, 0), 2)
        cv2.circle(all_lines_debug, (int(x), int(y)), 5, (0, 0, 255), -1)

    draw_coin_candidate(all_lines_debug, coin_detection.selected_circle, (0, 255, 255), 5)
    cv2.circle(all_lines_debug, coin_detection.coin_center, 5, (0, 0, 255), -1)
    draw_coin_candidate(result_debug, coin_detection.selected_circle, (0, 255, 255), 5)

    line_candidates = line_detection.raw_line_candidates or line_detection.line_candidates
    for _, _, _, (x1, y1, x2, y2) in line_candidates:
        cv2.line(all_lines_debug, (x1, y1), (x2, y2), (255, 0, 255), 3)

    drawn_edges = line_detection.display_edges or line_detection.outer_edges
    for _, _, _, (x1, y1, x2, y2) in drawn_edges:
        cv2.line(all_lines_debug, (x1, y1), (x2, y2), (0, 255, 0), 8)

    for _, _, _, (x1, y1, x2, y2) in line_detection.right_angle_edges:
        cv2.line(all_lines_debug, (x1, y1), (x2, y2), (255, 0, 0), 10)

    for _, _, _, (x1, y1, x2, y2) in line_detection.best_right_angle_edges:
        cv2.line(all_lines_debug, (x1, y1), (x2, y2), (0, 120, 0), 14)

    for _, _, _, (x1, y1, x2, y2) in line_detection.longest_right_angle_edges:
        cv2.line(all_lines_debug, (x1, y1), (x2, y2), (0, 165, 255), 18)
        cv2.line(result_debug, (x1, y1), (x2, y2), (0, 165, 255), 18)

    for x1, y1, x2, y2 in line_detection.best_extension_segments:
        cv2.line(all_lines_debug, (x1, y1), (x2, y2), (255, 255, 0), 12)

    for x1, y1, x2, y2 in line_detection.longest_extension_segments:
        cv2.line(result_debug, (x1, y1), (x2, y2), (255, 255, 0), 12)

    for x1, y1, x2, y2 in line_detection.extension_segments:
        cv2.line(all_lines_debug, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return all_lines_debug, result_debug


def detect_coin(preprocessing, config):
    method = config.get("COIN_DETECTION_METHOD", "contour").lower()
    if method in ("contour", "circularity"):
        return detect_coin_by_contours(preprocessing, config)
    raise ValueError(f"Unbekannte Muenzerkennung: {method}")


def hough_line(preprocessing, config):
    lines, raw_line_candidates = find_hough_line_candidates(
        preprocessing.edges,
        config["HOUGH_MIN_LINE_LENGTH_RATIO"],
        config["HOUGH_THRESHOLD"],
        config["HOUGH_MAX_LINE_GAP_RATIO"],
    )
    line_candidates = merge_collinear_line_candidates(
        raw_line_candidates,
        config["MERGE_LINE_ANGLE_TOLERANCE_DEG"],
        config["MERGE_LINE_DISTANCE_TOLERANCE_PX"],
        config["MERGE_LINE_STRICT_DISTANCE_TOLERANCE_PX"],
        config["MERGE_LINE_MAX_OVERLAP_RATIO"],
    )
    line_candidates = filter_lines_by_min_length_ratio(
        line_candidates,
        preprocessing.edges.shape,
        config.get("MERGED_LINE_MIN_LENGTH_RATIO", 0.0),
    )
    right_angle_edges = find_right_angle_edges(
        line_candidates,
        config["RIGHT_ANGLE_TOLERANCE_DEG"],
    )
    (
        best_right_angle_edges,
        longest_right_angle_edges,
        best_extension_segments,
        longest_extension_segments,
    ) = extend_longest_right_angle_pairs(
        line_candidates,
        config["RIGHT_ANGLE_TOLERANCE_DEG"],
        config.get("RIGHT_ANGLE_EXTENSION_PAIR_COUNT", 3),
        config.get("MAX_ACCEPTED_EXTENSION_LENGTH_RATIO", 1.10),
    )
    outer_edges = longest_right_angle_edges or best_right_angle_edges
    extension_segments = longest_extension_segments or best_extension_segments
    edge_distance_px = (
        segment_distance(outer_edges[0][3], outer_edges[1][3])
        if len(outer_edges) >= 2
        else float("nan")
    )

    return LineDetection(
        lines=lines,
        line_candidates=line_candidates,
        outer_edges=outer_edges,
        edge_distance_px=edge_distance_px,
        method="hough",
        extension_segments=extension_segments,
        display_edges=line_candidates,
        raw_line_candidates=raw_line_candidates,
        right_angle_edges=right_angle_edges,
        best_right_angle_edges=best_right_angle_edges,
        longest_right_angle_edges=longest_right_angle_edges,
        best_extension_segments=best_extension_segments,
        longest_extension_segments=longest_extension_segments,
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


def create_debug_images(preprocessing, coin_detection, line_detection):
    coin_debug = create_coin_debug(
        preprocessing.img,
        coin_detection.scored_circles,
        coin_detection.selected_circle,
    )
    all_lines_debug, result_debug = create_line_debug_images(
        preprocessing.img,
        coin_detection,
        line_detection,
    )
    return DebugImages(
        coin_debug=coin_debug,
        all_lines_debug=all_lines_debug,
        result_debug=result_debug,
    )


def build_measurement_result(coin_detection, line_detection, dimension_result, debug_images):
    return MeasurementResult(
        coin_detection_method=coin_detection.method,
        dimension_method=line_detection.method,
        coin_center=coin_detection.coin_center,
        coin_radius=coin_detection.coin_radius,
        pixels_per_mm=coin_detection.pixels_per_mm,
        circle_candidates=coin_detection.scored_circles,
        selected_circle=coin_detection.selected_circle,
        line_count=0 if line_detection.lines is None else len(line_detection.lines),
        line_candidate_count=len(line_detection.line_candidates),
        outer_edges=line_detection.outer_edges,
        angle_difference_deg=dimension_result.angle_difference_deg,
        edge_distance_px=line_detection.edge_distance_px,
        length_mm=dimension_result.length_mm,
        width_mm=dimension_result.width_mm,
        coin_debug=debug_images.coin_debug,
        all_lines_debug=debug_images.all_lines_debug,
        result_debug=debug_images.result_debug,
    )


def measure_inbus(preprocessing, config):
    coin_detection = detect_coin(preprocessing, config)
    line_detection = hough_line(preprocessing, config)
    dimension_result = measure_dimensions_by_hough(line_detection, coin_detection, config)
    debug_images = create_debug_images(preprocessing, coin_detection, line_detection)
    return build_measurement_result(coin_detection, line_detection, dimension_result, debug_images)
