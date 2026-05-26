import cv2
import numpy as np

from .hough_bemessung import angle_distance, line_candidate_from_points
from .models import DimensionResult, LineDetection


def find_inbus_contour_by_area(clean, coin_center, coin_radius, min_area=500, coin_exclusion_radius_factor=1.0):
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coin_x, coin_y = coin_center
    best_contour = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        center_x = x + width / 2
        center_y = y + height / 2
        distance_to_coin = np.hypot(center_x - coin_x, center_y - coin_y)
        if distance_to_coin < coin_radius * coin_exclusion_radius_factor:
            continue

        if area > best_area:
            best_area = area
            best_contour = contour

    return best_contour


def box_edges_from_points(box_points):
    edges = []
    for index in range(len(box_points)):
        x1, y1 = box_points[index]
        x2, y2 = box_points[(index + 1) % len(box_points)]
        edges.append(line_candidate_from_points((x1, y1, x2, y2)))
    return sorted(edges, reverse=True, key=lambda item: item[0])


def detect_inbus_box(preprocessing, coin_detection, config):
    inbus_contour = find_inbus_contour_by_area(
        preprocessing.clean,
        coin_detection.coin_center,
        coin_detection.coin_radius,
        config.get("CONTOUR_INBUS_MIN_AREA", 500),
        config.get("CONTOUR_INBUS_COIN_EXCLUSION_RADIUS_FACTOR", 1.0),
    )
    if inbus_contour is None:
        raise RuntimeError("Kein Inbus gefunden!")

    rect = cv2.minAreaRect(inbus_contour)
    box = np.int32(cv2.boxPoints(rect))
    box_edges = box_edges_from_points(box)
    width_px, height_px = rect[1]

    return LineDetection(
        lines=None,
        line_candidates=box_edges,
        outer_edges=box_edges,
        edge_distance_px=float(min(width_px, height_px)),
        method="contour",
        display_edges=box_edges,
        inbus_contour=inbus_contour,
        min_area_rect=rect,
        box_points=box,
        box_dimensions_px=(float(width_px), float(height_px)),
    )


def measure_dimensions_by_contours(preprocessing, coin_detection, config, line_detection=None):
    if line_detection is None:
        line_detection = detect_inbus_box(preprocessing, coin_detection, config)
    return measure_dimensions_from_inbus_box(line_detection, coin_detection)


def measure_dimensions_from_inbus_box(line_detection, coin_detection):
    if line_detection.box_dimensions_px is not None:
        width_px, height_px = line_detection.box_dimensions_px
    elif line_detection.min_area_rect is not None:
        _, (width_px, height_px), _ = line_detection.min_area_rect
    else:
        edge_lengths = sorted((edge[0] for edge in line_detection.outer_edges), reverse=True)
        if len(edge_lengths) < 2:
            return DimensionResult(float("nan"), float("nan"), float("nan"))
        height_px, width_px = edge_lengths[0], edge_lengths[-1]

    short_side_px = min(width_px, height_px)
    long_side_px = max(width_px, height_px)
    if len(line_detection.outer_edges) >= 2:
        long_edge = max(line_detection.outer_edges, key=lambda edge: edge[0])
        short_edge = min(line_detection.outer_edges, key=lambda edge: edge[0])
        angle_difference_deg = angle_distance(long_edge[1], short_edge[1])
    else:
        angle_difference_deg = float("nan")

    return DimensionResult(
        length_mm=long_side_px / coin_detection.pixels_per_mm,
        width_mm=short_side_px / coin_detection.pixels_per_mm,
        angle_difference_deg=angle_difference_deg,
    )
