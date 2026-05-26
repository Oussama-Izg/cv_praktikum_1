from pathlib import Path

import cv2

from logik.bemessung import measure_inbus
from logik.vorverarbeitung import run_preprocessing


IMAGE_PATHS = ["bilder/1.jpg", "bilder/2.jpg", "bilder/5.jpg"]
OUTPUT_DIR = Path("output")


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


def print_bemessung_debug(result, image_path):
    image_name = Path(image_path).stem
    save_image(f"{image_name}_bemessung_debug.png", result.all_lines_debug)


def print_bemessung(result, image_path):
    image_name = Path(image_path).stem
    save_image(f"{image_name}_bemessung.png", result.result_debug)


def print_hough_line_infos(result):
    print(f"Gefundene Hough-Linien: {result.line_count}")
    print(f"Angezeigte Hough-Linien: {result.line_candidate_count}")


def print_abmessungen(result):
    print(f"Laenge: {result.length_mm:.2f} mm")
    print(f"Breite: {result.width_mm:.2f} mm")


def measure_image(image_path, config):
    preprocessing = run_preprocessing(image_path)
    result = measure_inbus(preprocessing, config)

    print_vorverarbeitung(preprocessing)
    print_bemessung_debug(result, image_path)
    print_bemessung(result, image_path)
    print_hough_line_infos(result)
    print_abmessungen(result)


def main():
    config = create_config()

    for image_path in IMAGE_PATHS:
        print(f"\nBild: {image_path}")
        measure_image(image_path, config)


if __name__ == "__main__":
    main()
