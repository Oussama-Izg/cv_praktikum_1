from dataclasses import dataclass, field
from typing import Optional

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
    inbus_contour: Optional[np.ndarray] = None
    min_area_rect: Optional[tuple] = None
    box_points: Optional[np.ndarray] = None
    box_dimensions_px: Optional[tuple] = None


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
