import cv2
import numpy as np

from .models import CircleScore, CoinDetection


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


def detect_coin(preprocessing, config):
    method = config.get("COIN_DETECTION_METHOD", "contour").lower()
    if method in ("contour", "circularity"):
        return detect_coin_by_contours(preprocessing, config)
    raise ValueError(f"Unbekannte Muenzerkennung: {method}")
