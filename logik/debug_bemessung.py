import cv2

from .kreis_bemessung import create_coin_debug, draw_coin_candidate
from .models import DebugImages


def save_image(output_dir, filename, image):
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / filename), image)


def save_debug_output(output_dir, preprocessing, debug_images):
    save_image(output_dir, "vorverarbeitung.png", preprocessing.edges)
    save_image(output_dir, "bemessung_debug.png", debug_images.all_lines_debug)
    save_image(output_dir, "bemessung.png", debug_images.result_debug)


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

    if line_detection.method == "contour":
        for _, _, _, (x1, y1, x2, y2) in line_detection.outer_edges:
            cv2.line(result_debug, (x1, y1), (x2, y2), (0, 255, 0), 8)

    for _, _, _, (x1, y1, x2, y2) in line_detection.best_right_angle_edges:
        cv2.line(all_lines_debug, (x1, y1), (x2, y2), (0, 120, 0), 14)

    for _, _, _, (x1, y1, x2, y2) in line_detection.longest_right_angle_edges:
        cv2.line(all_lines_debug, (x1, y1), (x2, y2), (0, 165, 255), 18)
        cv2.line(result_debug, (x1, y1), (x2, y2), (0, 165, 255), 18)

    for x1, y1, x2, y2 in line_detection.best_extension_segments:
        cv2.line(all_lines_debug, (x1, y1), (x2, y2), (255, 255, 0), 12)

    for x1, y1, x2, y2 in line_detection.longest_extension_segments:
        cv2.line(result_debug, (x1, y1), (x2, y2), (255, 255, 0), 12)

    return all_lines_debug, result_debug


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
