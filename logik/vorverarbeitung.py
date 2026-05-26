from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PreprocessingResult:
    image_path: str
    img: np.ndarray
    gray: np.ndarray
    blurred: np.ndarray
    binary: np.ndarray
    clean: np.ndarray
    edges: np.ndarray


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    return img


def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def blur_gray(gray, kernel_size=(7, 7)):
    return cv2.GaussianBlur(gray, kernel_size, 0)


def threshold_otsu(blurred):
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary


def clean_binary(binary, kernel_size=(5, 5), close_iterations=2, open_iterations=2):
    kernel = np.ones(kernel_size, np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    return clean


def create_edges(clean, threshold1=50, threshold2=150, aperture_size=3):
    return cv2.Canny(clean, threshold1, threshold2, apertureSize=aperture_size)


def save_preprocessing(result, output_path):
    np.savez_compressed(
        output_path,
        img=result.img,
        gray=result.gray,
        blurred=result.blurred,
        binary=result.binary,
        clean=result.clean,
        edges=result.edges,
        image_path=np.array(result.image_path),
    )
    return Path(output_path)


def build_preprocessing_result(image_path, img, gray, blurred, binary, clean, edges):
    return PreprocessingResult(
        image_path=image_path,
        img=img,
        gray=gray,
        blurred=blurred,
        binary=binary,
        clean=clean,
        edges=edges,
    )


def run_preprocessing(image_path, output_path=None):
    img = load_image(image_path)
    gray = convert_to_gray(img)
    blurred = blur_gray(gray)
    binary = threshold_otsu(blurred)
    clean = clean_binary(binary)
    edges = create_edges(clean)

    result = build_preprocessing_result(image_path, img, gray, blurred, binary, clean, edges)

    if output_path is not None:
        save_preprocessing(result, output_path)

    return result
