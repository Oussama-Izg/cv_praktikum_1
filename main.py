from pathlib import Path

import cv2

from logik.bemessung import (
    create_debug_images,
    detect_coin_by_contours,
    detect_inbus_box,
    hough_line,
    measure_dimensions_by_contours,
    measure_dimensions_by_hough,
)
from logik.vorverarbeitung import (
    blur_gray,
    build_preprocessing_result,
    clean_binary,
    convert_to_gray,
    create_edges,
    load_image,
    threshold_otsu,
)


BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = "bilder/2.jpg"
OUTPUT_DIR = BASE_DIR / "output"


def save_image(filename, image):
    OUTPUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / filename), image)


def create_config():
    return {
        "COIN_DIAMETER_MM": 22.25,
        "COIN_DETECTION_METHOD": "contour",
        "COIN_SCALE_AXIS": "mean",
        "CONTOUR_COIN_MIN_AREA": 200,
        "CONTOUR_COIN_MIN_AXIS_RATIO": 0.35,
        "CONTOUR_COIN_MIN_ELLIPSE_FILL_RATIO": 0.55,
        "CONTOUR_COIN_MAX_ELLIPSE_FILL_RATIO": 1.35,
        "HOUGH_MIN_LINE_LENGTH_RATIO": 0.01,
        "HOUGH_THRESHOLD": 10,
        "HOUGH_MAX_LINE_GAP_RATIO": 0.01,
        "ANGLE_TOLERANCE_DEG": 15,
        "RIGHT_ANGLE_TOLERANCE_DEG": 20,
        "MAX_RIGHT_ANGLE_DISTANCE_PX": 120,
        "MERGE_LINE_ANGLE_TOLERANCE_DEG": 5,
        "MERGE_LINE_DISTANCE_TOLERANCE_PX": 60,
        "MERGE_LINE_STRICT_DISTANCE_TOLERANCE_PX": 8,
        "MERGE_LINE_MAX_OVERLAP_RATIO": 0.25,
        "MERGED_LINE_MIN_LENGTH_RATIO": 0.05,
        "RIGHT_ANGLE_EXTENSION_PAIR_COUNT": 3,
        "MAX_ACCEPTED_EXTENSION_LENGTH_RATIO": 1.10,
        "SHORT_ARM_RECOVERY_ENABLED": True,
        "SHORT_ARM_RECOVERY_MIN_RATIO": 0.24,
        "SHORT_ARM_RECOVERY_MAX_RATIO": 0.30,
        "SHORT_ARM_RECOVERY_ANGLE_TOLERANCE_DEG": 8,
        "SHORT_ARM_RECOVERY_OFFSET_TOLERANCE_PX": 80,
    }


def print_vorverarbeitung(preprocessing):
    image_name = Path(preprocessing.image_path).stem
    save_image(f"{image_name}_vorverarbeitung.png", preprocessing.edges)


def print_bemessung_debug(debug_images):
    image_name = Path(IMAGE_PATH).stem
    save_image(f"{image_name}_bemessung_debug.png", debug_images.all_lines_debug)


def print_bemessung(debug_images):
    image_name = Path(IMAGE_PATH).stem
    save_image(f"{image_name}_bemessung.png", debug_images.result_debug)


def print_hough_line_infos(line_detection, dimension_result):
    line_count = 0 if line_detection.lines is None else len(line_detection.lines)
    if line_detection.method == "hough":
        print(f"Gefundene Hough-Linien: {line_count}")
        print(f"Angezeigte Hough-Linien: {len(line_detection.line_candidates)}")
    else:
        print(f"Erkannte Kontur-Kanten: {len(line_detection.line_candidates)}")


def print_abmessungen(dimension_result):
    print(f"Laenge: {dimension_result.length_mm:.2f} mm")
    print(f"Breite: {dimension_result.width_mm:.2f} mm")


def main():
    config = create_config()

    print(f"\nBild: {IMAGE_PATH}")

    # Vorverarbeitung
    img = load_image(str(BASE_DIR / IMAGE_PATH))
    gray = convert_to_gray(img)
    blurred = blur_gray(gray)
    binary = threshold_otsu(blurred)
    clean = clean_binary(binary)
    edges = create_edges(clean)
    preprocessing = build_preprocessing_result(IMAGE_PATH, img, gray, blurred, binary, clean, edges)

    # Bemessung
    # coin_detection = detect_coin_by_hough(preprocessing, config)
    coin_detection = detect_coin_by_contours(preprocessing, config)

    # hough
    #line_detection = hough_line(preprocessing, config)
    #dimension_result = measure_dimensions_by_hough(line_detection, coin_detection, config)

    # by_contours
    line_detection = detect_inbus_box(preprocessing, coin_detection, config)
    dimension_result = measure_dimensions_by_contours(
        preprocessing,
        coin_detection,
        config,
        line_detection,
    )

    debug_images = create_debug_images(preprocessing, coin_detection, line_detection)

    # Ausgabe
    print_vorverarbeitung(preprocessing)
    print_bemessung_debug(debug_images)
    print_bemessung(debug_images)
    print_hough_line_infos(line_detection, dimension_result)
    print_abmessungen(dimension_result)


if __name__ == "__main__":
    main()
