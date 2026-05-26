from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CircleScore:
    score: float
    edge_support: int
    edge_roundness: float
    fill_ratio: float
    circle: tuple


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


def find_hough_circles(
    gray,
    dp,
    min_dist,
    param1,
    param2,
    min_radius,
    max_radius,
):
    hough_input = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        hough_input,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        raise RuntimeError("Keine Muenze per Hough-Kreis-Transformation gefunden")
    return np.round(circles[0]).astype(int)


def circle_edge_support(circle, edges):
    x, y, radius = circle
    mask = np.zeros_like(edges, dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, 8)
    return cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))


def circle_edge_roundness(circle, edges, bin_count=72):
    x, y, radius = circle
    mask = np.zeros_like(edges, dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, 8)
    circle_edges = cv2.bitwise_and(edges, edges, mask=mask)
    edge_y, edge_x = np.where(circle_edges > 0)

    if len(edge_x) == 0:
        return 0.0

    angles = np.arctan2(edge_y - y, edge_x - x)
    angle_bins = ((angles + np.pi) / (2 * np.pi) * bin_count).astype(int)
    angle_bins = np.clip(angle_bins, 0, bin_count - 1)
    occupied_bins = np.unique(angle_bins)
    return len(occupied_bins) / bin_count


def circle_fill_ratio(circle, clean):
    x, y, radius = circle
    mask = np.zeros_like(clean, dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, -1)
    filled_pixels = cv2.countNonZero(cv2.bitwise_and(clean, clean, mask=mask))
    circle_area = np.pi * radius ** 2
    return filled_pixels / circle_area


def score_circles(circles, edges, clean):
    scored = []
    for circle in circles:
        edge_support = circle_edge_support(circle, edges)
        edge_roundness = circle_edge_roundness(circle, edges)
        fill_ratio = circle_fill_ratio(circle, clean)
        score = edge_roundness * edge_support * max(fill_ratio, 0.01)
        scored.append(
            CircleScore(
                score=score,
                edge_support=edge_support,
                edge_roundness=edge_roundness,
                fill_ratio=fill_ratio,
                circle=tuple(int(value) for value in circle),
            )
        )
    return sorted(scored, reverse=True, key=lambda item: item.score)


def find_contour_coin_candidates(clean, min_area):
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

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius <= 0:
            continue

        fill_ratio = area / (np.pi * radius ** 2)
        scored.append(
            CircleScore(
                score=float(circularity),
                edge_support=int(area),
                edge_roundness=float(circularity),
                fill_ratio=float(fill_ratio),
                circle=(int(x), int(y), int(radius)),
            )
        )

    return sorted(scored, reverse=True, key=lambda item: item.score)


def select_coin(scored_circles, coin_diameter_mm):
    if not scored_circles:
        raise RuntimeError("Keine Kreis-Kandidaten gefunden")

    selected = scored_circles[0]
    coin_x, coin_y, coin_radius = selected.circle
    coin_center = (int(coin_x), int(coin_y))
    coin_radius = int(coin_radius)
    pixels_per_mm = (coin_radius * 2) / coin_diameter_mm
    return selected, coin_center, coin_radius, pixels_per_mm


def create_coin_debug(img, scored_circles, selected_circle):
    debug = img.copy()
    for rank, circle_score in enumerate(scored_circles, start=1):
        x, y, radius = circle_score.circle
        center = (int(x), int(y))
        label_position = (center[0] - 60, center[1] + 35)
        cv2.circle(debug, center, int(radius), (255, 0, 0), 2)
        cv2.putText(debug, str(rank), label_position, cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 8)

    x, y, radius = selected_circle.circle
    cv2.circle(debug, (int(x), int(y)), int(radius), (0, 255, 255), 4)
    cv2.circle(debug, (int(x), int(y)), 5, (0, 0, 255), -1)
    return debug


def detect_coin_by_hough(preprocessing, config):
    min_image_dimension = min(preprocessing.gray.shape[:2])
    circles = find_hough_circles(
        preprocessing.gray,
        dp=config["HOUGH_CIRCLE_DP"],
        min_dist=int(min_image_dimension * config["HOUGH_CIRCLE_MIN_DIST_RATIO"]),
        param1=config["HOUGH_CIRCLE_PARAM1"],
        param2=config["HOUGH_CIRCLE_PARAM2"],
        min_radius=int(min_image_dimension * config["HOUGH_CIRCLE_MIN_RADIUS_RATIO"]),
        max_radius=int(min_image_dimension * config["HOUGH_CIRCLE_MAX_RADIUS_RATIO"]),
    )
    scored_circles = score_circles(circles, preprocessing.edges, preprocessing.clean)
    selected_circle, coin_center, coin_radius, pixels_per_mm = select_coin(
        scored_circles,
        config["COIN_DIAMETER_MM"],
    )
    return CoinDetection(
        method="hough",
        circles=circles,
        scored_circles=scored_circles,
        selected_circle=selected_circle,
        coin_center=coin_center,
        coin_radius=coin_radius,
        pixels_per_mm=pixels_per_mm,
    )


def detect_coin_by_contours(preprocessing, config):
    scored_circles = find_contour_coin_candidates(
        preprocessing.clean,
        config["CONTOUR_COIN_MIN_AREA"],
    )
    selected_circle, coin_center, coin_radius, pixels_per_mm = select_coin(
        scored_circles,
        config["COIN_DIAMETER_MM"],
    )
    return CoinDetection(
        method="contour",
        circles=np.array([circle.circle for circle in scored_circles], dtype=int),
        scored_circles=scored_circles,
        selected_circle=selected_circle,
        coin_center=coin_center,
        coin_radius=coin_radius,
        pixels_per_mm=pixels_per_mm,
    )


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

    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        length = np.hypot(x2 - x1, y2 - y1)
        if length < min_line_length:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = ((angle + 90) % 180) - 90
        normal_angle = np.radians(angle + 90)
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        offset = midpoint_x * np.cos(normal_angle) + midpoint_y * np.sin(normal_angle)
        line_candidates.append((length, angle, offset, (x1, y1, x2, y2)))

    return lines, sorted(line_candidates, reverse=True, key=lambda item: item[0])


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

    points = []
    for _, _, _, (x1, y1, x2, y2) in matching_segments:
        points.append([x1, y1])
        points.append([x2, y2])

    if len(points) < 2:
        return base_line

    points = np.array(points, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([vx, vy], dtype=np.float32)
    direction = direction / np.linalg.norm(direction)
    line_point = np.array([x0, y0], dtype=np.float32)
    projected_points = (points - line_point) @ direction

    start_point = line_point + direction * np.min(projected_points)
    end_point = line_point + direction * np.max(projected_points)
    x1, y1 = np.round(start_point).astype(int)
    x2, y2 = np.round(end_point).astype(int)
    extended_length = np.hypot(x2 - x1, y2 - y1)
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    angle = ((angle + 90) % 180) - 90
    normal_angle = np.radians(angle + 90)
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2
    offset = midpoint_x * np.cos(normal_angle) + midpoint_y * np.sin(normal_angle)
    return (extended_length, angle, offset, (x1, y1, x2, y2))


def matching_line_segments(base_line, candidates, angle_tolerance_deg, distance_tolerance_px):
    _, base_angle, base_offset, _ = base_line
    return [
        candidate
        for candidate in candidates
        if angle_distance(base_angle, candidate[1]) <= angle_tolerance_deg
        and abs(base_offset - candidate[2]) <= distance_tolerance_px
    ]


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


def select_fallback_outer_edges(line_candidates):
    best_pair = None
    best_score = None

    for i, first in enumerate(line_candidates):
        for second in line_candidates[i + 1:]:
            angle_difference = angle_distance(first[1], second[1])
            if angle_difference < 5:
                continue

            length_sum = first[0] + second[0]
            right_angle_error = abs(angle_difference - 90)
            score = (right_angle_error, -length_sum)
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (first, second)

    if best_pair is not None:
        return list(best_pair)
    return line_candidates[:2]


def select_outer_edges(
    line_candidates,
    right_angle_tolerance_deg,
    max_right_angle_distance_px,
    extend_angle_tolerance_deg,
    extend_distance_tolerance_px,
):
    best_pair = None
    best_score = None
    best_pair_distance = None

    for i, first in enumerate(line_candidates):
        for second in line_candidates[i + 1:]:
            angle_difference = angle_distance(first[1], second[1])
            right_angle_error = abs(angle_difference - 90)
            if right_angle_error > right_angle_tolerance_deg:
                continue

            pair_distance = segment_distance(first[3], second[3])
            if pair_distance > max_right_angle_distance_px:
                continue

            extended_first = extend_line_with_candidates(
                first,
                line_candidates,
                extend_angle_tolerance_deg,
                extend_distance_tolerance_px,
            )
            extended_second = extend_line_with_candidates(
                second,
                line_candidates,
                extend_angle_tolerance_deg,
                extend_distance_tolerance_px,
            )
            length_sum = extended_first[0] + extended_second[0]
            score = (right_angle_error, pair_distance, -length_sum)
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (extended_first, extended_second)
                best_pair_distance = pair_distance

    if best_pair is None:
        raise RuntimeError("Kein nahes Linienpaar mit ca. 90 Grad gefunden")

    outer_edges = [best_pair[0], best_pair[1]]
    outer_edges, extended_distance, extension_segments = extend_edges_until_they_meet(outer_edges)
    return outer_edges, min(best_pair_distance, extended_distance), extension_segments


def calculate_dimensions(outer_edges, pixels_per_mm, angle_tolerance_deg):
    if len(outer_edges) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(outer_edges) == 1:
        length = outer_edges[0][0]
        return length / pixels_per_mm, float("nan"), float("nan")

    measurements = []
    for length, angle, offset, points in outer_edges:
        measurements.append((length, length / pixels_per_mm, angle, offset, points))

    measurements = sorted(measurements, reverse=True, key=lambda item: item[0])
    angle_difference = angle_distance(measurements[0][2], measurements[1][2])
    length_mm = measurements[0][1]

    if angle_difference <= angle_tolerance_deg:
        width_px = abs(measurements[0][3] - measurements[1][3])
        width_mm = width_px / pixels_per_mm
    else:
        width_mm = measurements[1][1]

    return length_mm, width_mm, angle_difference


def create_line_debug_images(
    img,
    coin_center,
    coin_radius,
    circle_candidates,
    line_candidates,
    outer_edges,
    display_edges,
    extension_segments,
    pixels_per_mm,
):
    all_lines_debug = img.copy()
    result_debug = img.copy()
    cv2.circle(result_debug, coin_center, coin_radius, (0, 255, 255), 3)
    cv2.circle(result_debug, coin_center, 5, (0, 0, 255), -1)

    for circle_score in circle_candidates:
        x, y, radius = circle_score.circle
        cv2.circle(all_lines_debug, (int(x), int(y)), int(radius), (255, 0, 0), 2)
        cv2.circle(all_lines_debug, (int(x), int(y)), 5, (0, 0, 255), -1)

    for _, _, _, (x1, y1, x2, y2) in line_candidates:
        cv2.line(all_lines_debug, (x1, y1), (x2, y2), (255, 0, 255), 2)

    drawn_edges = display_edges if display_edges else outer_edges
    for _, _, _, (x1, y1, x2, y2) in drawn_edges:
        cv2.line(result_debug, (x1, y1), (x2, y2), (0, 255, 0), 3)

    for x1, y1, x2, y2 in extension_segments:
        cv2.line(result_debug, (x1, y1), (x2, y2), (255, 255, 0), 4)

    return all_lines_debug, result_debug


def find_inbus_contour_by_largest_area(clean, coin_center, coin_radius, min_area):
    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    inbus_contour = None
    max_area = 0
    coin_x, coin_y = coin_center

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        center_x = x + width // 2
        center_y = y + height // 2
        distance = np.hypot(center_x - coin_x, center_y - coin_y)
        if distance < coin_radius:
            continue

        if area > max_area:
            max_area = area
            inbus_contour = contour

    if inbus_contour is None:
        raise RuntimeError("Kein Inbus per Kontur gefunden")

    return inbus_contour


def box_edges_from_points(box):
    edges = []
    for index in range(4):
        x1, y1 = box[index]
        x2, y2 = box[(index + 1) % 4]
        length = np.hypot(x2 - x1, y2 - y1)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = ((angle + 90) % 180) - 90
        normal_angle = np.radians(angle + 90)
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        offset = midpoint_x * np.cos(normal_angle) + midpoint_y * np.sin(normal_angle)
        edges.append((length, angle, offset, (int(x1), int(y1), int(x2), int(y2))))
    return sorted(edges, reverse=True, key=lambda item: item[0])


def detect_inbus_box(preprocessing, coin_detection, config):
    contour = find_inbus_contour_by_largest_area(
        preprocessing.clean,
        coin_detection.coin_center,
        coin_detection.coin_radius,
        config["CONTOUR_INBUS_MIN_AREA"],
    )
    rect = cv2.minAreaRect(contour)
    _, (width, height), _ = rect
    box = np.int32(cv2.boxPoints(rect))
    box_edges = box_edges_from_points(box)
    outer_edges = [box_edges[0], box_edges[2]]

    line_detection = LineDetection(
        lines=None,
        line_candidates=box_edges,
        outer_edges=outer_edges,
        edge_distance_px=0.0,
        method="contour",
    )
    line_detection.box_length_px = float(max(width, height))
    line_detection.box_width_px = float(min(width, height))
    return line_detection


def detect_coin(preprocessing, config):
    method = config.get("COIN_DETECTION_METHOD", "hough").lower()
    if method == "hough":
        return detect_coin_by_hough(preprocessing, config)
    if method in ("contour", "circularity"):
        return detect_coin_by_contours(preprocessing, config)
    raise ValueError(f"Unbekannte Muenzerkennung: {method}")


def hough_line(preprocessing, config):
    lines, line_candidates = find_hough_line_candidates(
        preprocessing.edges,
        config["HOUGH_MIN_LINE_LENGTH_RATIO"],
        config["HOUGH_THRESHOLD"],
        config["HOUGH_MAX_LINE_GAP_RATIO"],
    )
    try:
        outer_edges, edge_distance_px, extension_segments = select_outer_edges(
            line_candidates,
            config["RIGHT_ANGLE_TOLERANCE_DEG"],
            config["MAX_RIGHT_ANGLE_DISTANCE_PX"],
            config["EXTEND_LINE_ANGLE_TOLERANCE_DEG"],
            config["EXTEND_LINE_DISTANCE_TOLERANCE_PX"],
        )
        display_edges = [
            extend_line_with_candidates(
                edge,
                line_candidates,
                config["EXTEND_LINE_ANGLE_TOLERANCE_DEG"],
                config["EXTEND_LINE_DISTANCE_TOLERANCE_PX"],
            )
            for edge in outer_edges
        ]
    except RuntimeError:
        selected_edges = select_fallback_outer_edges(line_candidates)
        outer_edges = [
            extend_line_with_candidates(
                edge,
                line_candidates,
                config["EXTEND_LINE_ANGLE_TOLERANCE_DEG"],
                config["EXTEND_LINE_DISTANCE_TOLERANCE_PX"],
            )
            for edge in selected_edges
        ] if len(selected_edges) >= 2 else selected_edges
        display_edges = outer_edges
        outer_edges, edge_distance_px, extension_segments = extend_edges_until_they_meet(outer_edges)
        if len(outer_edges) < 2:
            edge_distance_px = float("nan")
    else:
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
        display_edges=display_edges,
    )


def measure_dimensions_by_hough(line_detection, coin_detection, config):
    length_mm, width_mm, angle_difference_deg = calculate_dimensions(
        line_detection.outer_edges,
        coin_detection.pixels_per_mm,
        config["ANGLE_TOLERANCE_DEG"],
    )
    return DimensionResult(
        length_mm=length_mm,
        width_mm=width_mm,
        angle_difference_deg=angle_difference_deg,
    )


def measure_dimensions_by_contours(preprocessing, coin_detection, config):
    line_detection = detect_inbus_box(preprocessing, coin_detection, config)
    return DimensionResult(
        length_mm=line_detection.box_length_px / coin_detection.pixels_per_mm,
        width_mm=line_detection.box_width_px / coin_detection.pixels_per_mm,
        angle_difference_deg=90.0,
    )


def create_debug_images(preprocessing, coin_detection, line_detection):
    coin_debug = create_coin_debug(
        preprocessing.img,
        coin_detection.scored_circles,
        coin_detection.selected_circle,
    )
    all_lines_debug, result_debug = create_line_debug_images(
        preprocessing.img,
        coin_detection.coin_center,
        coin_detection.coin_radius,
        coin_detection.scored_circles,
        line_detection.line_candidates,
        line_detection.outer_edges,
        line_detection.display_edges,
        line_detection.extension_segments,
        coin_detection.pixels_per_mm,
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
