from pathlib import Path

import cv2

from logik.bemessung import (
    create_debug_images,
    detect_coin_by_contours,
    detect_coin_by_hough,
    detect_inbus_box,
    hough_line,
    measure_dimensions_by_hough,
    measure_dimensions_by_contours,
)
from logik.vorverarbeitung import (
    build_preprocessing_result,
    blur_gray,
    clean_binary,
    convert_to_gray,
    convert_to_rgb,
    create_edges,
    load_image,
    threshold_otsu,
)


IMAGE_PATH = "bilder/1.jpg"
OUTPUT_DIR = Path("output")


def save_image(filename, image):
    OUTPUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / filename), image)


def create_config():
    return {
        "COIN_DIAMETER_MM": 22.25,
        "CONTOUR_COIN_MIN_AREA": 200,
        "CONTOUR_INBUS_MIN_AREA": 500,
        "HOUGH_CIRCLE_DP": 1.2,
        "HOUGH_CIRCLE_MIN_DIST_RATIO": 0.20,
        "HOUGH_CIRCLE_PARAM1": 100,
        "HOUGH_CIRCLE_PARAM2": 30,
        "HOUGH_CIRCLE_MIN_RADIUS_RATIO": 0.03,
        "HOUGH_CIRCLE_MAX_RADIUS_RATIO": 0.12,
        "HOUGH_MIN_LINE_LENGTH_RATIO": 0.04,
        "HOUGH_THRESHOLD": 35,
        "HOUGH_MAX_LINE_GAP": 160,
        "ANGLE_TOLERANCE_DEG": 15,
        "RIGHT_ANGLE_TOLERANCE_DEG": 20,
        "MAX_RIGHT_ANGLE_DISTANCE_PX": 120,
        "EXTEND_LINE_ANGLE_TOLERANCE_DEG": 15,
        "EXTEND_LINE_DISTANCE_TOLERANCE_PX": 90,
    }


def print_vorverarbeitung(preprocessing):
    save_image("vorverarbeitung.png", preprocessing.edges)


def print_bemessung_debug(debug_images):
    save_image("bemessung_debug.png", debug_images.all_lines_debug)


def print_bemessung(debug_images):
    save_image("bemessung.png", debug_images.result_debug)


def print_hough_line_infos(line_detection, dimension_result):
    print(f"Gefundene Hough-Linien: {0 if line_detection.lines is None else len(line_detection.lines)}")
    print(f"Kandidaten nach Laengenfilter: {len(line_detection.line_candidates)}")
    print(f"Verwendete Aussenkanten: {len(line_detection.outer_edges)}")
    print(f"Winkel zwischen den Kanten: {dimension_result.angle_difference_deg:.2f} Grad")
    print(f"Abstand zwischen den Kanten: {line_detection.edge_distance_px:.2f} px")


def print_abmessungen(dimension_result):
    print(f"Laenge: {dimension_result.length_mm:.2f} mm")
    print(f"Breite: {dimension_result.width_mm:.2f} mm")


def main():
    config = create_config()

    # Vorverarbeitung
    img = load_image(IMAGE_PATH)
    img_rgb = convert_to_rgb(img)
    gray = convert_to_gray(img)
    blurred = blur_gray(gray)
    binary = threshold_otsu(blurred)
    clean = clean_binary(binary)
    edges = create_edges(clean)
    preprocessing = build_preprocessing_result(IMAGE_PATH, img, img_rgb, gray, blurred, binary, clean, edges)

    # Bemessung
    #coin_detection = detect_coin_by_hough(preprocessing, config)
    coin_detection = detect_coin_by_contours(preprocessing, config)

    ## hough
    line_detection = hough_line(preprocessing, config)
    dimension_result = measure_dimensions_by_hough(line_detection, coin_detection, config)
    
    ## by_contours
    #line_detection = detect_inbus_box(preprocessing, coin_detection, config)
    #dimension_result = measure_dimensions_by_contours(preprocessing, coin_detection, config)

    debug_images = create_debug_images(preprocessing, coin_detection, line_detection)

    # Ausgabe
    print_vorverarbeitung(preprocessing)
    print_bemessung_debug(debug_images)
    print_bemessung(debug_images)
    ## Print-Outputs
    print_hough_line_infos(line_detection, dimension_result)
    print_abmessungen(dimension_result)


if __name__ == "__main__":
    main()
