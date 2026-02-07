from __future__ import annotations

import collections
import os
import pickle
import threading
import tkinter as tk
from tkinter import ttk, filedialog
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional, Tuple, Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
BUILD_DIR = REPO_ROOT / "coin_counter_cpp" / "build"
CLASSIFIER_MODEL_NAMES = ["coin_svm.yaml", "coin_knn.yaml", "coin_rtrees.yaml", "coin_nb.yaml"]
CLASSIFIER_NAMES = ["SVM", "KNN", "RandomForest", "NaiveBayes"]
SCALER_NAME = "coin_scaler.yaml"
CLUSTER_COLORS_BGR = [
    (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148), (75, 86, 140),
]
CLASS_TO_VALUE_EUR = [0.20, 0.10, 1.0, 0.01, 0.02, 0.05]


def default_params() -> dict:
    return {
        "CLAHE_CLIP": 5,
        "CLAHE_GRID": 2,
        "BLUR_KSIZE": 5,
        "CANNY_THRESHOLD1": 50.0,
        "CANNY_THRESHOLD2": 150.0,
        "HOUGH_DP": 1.5,
        "HOUGH_MIN_DIST": 47,
        "HOUGH_PARAM1": 105.0,
        "HOUGH_PARAM2": 30.0,
        "MIN_RADIUS_PX": 22,
        "MAX_RADIUS_PX": 43,
        "MIN_CONTOUR_AREA": 100.0,
        "MIN_CIRCULARITY": 0.5,
        "DIAMETER_MM_MIN": 10.0,
        "DIAMETER_MM_MAX": 40.0,
        "CENTER_MATCH_PX": 20,
        "DIAMETER_HISTORY_LEN": 100,
        "MAX_FRAMES_MISSING": 5,
        "MAX_DIAMETER_DEVIATION_MM": 3.0,
        "MIN_SAMPLES_FOR_STABLE": 3,
        "HIST_WIDTH": 320,
        "HIST_HEIGHT": 240,
        "HIST_BIN_MIN": 10.0,
        "HIST_BIN_MAX": 41.0,
        "HIST_BIN_STEP": 0.2,
        "PAPER_WIDTH_MM": 330.0,
        "PAPER_HEIGHT_MM": 216.0,
        "SCALE_FACTOR": 3.0,
        "keep_frac": 0.8,
        "stabilizer_window": 10,
        "paper_line_min_length": 100,
        "paper_morph_kernel": 3,
        "paper_approx_eps_factor": 0.02,
        "paper_min_area": 10000,
        "camera_index": 2,
        "frame_width": 1280,
        "frame_height": 720,
        "inner_radius_frac": 0.7,
        "build_dir": str(BUILD_DIR),
        "calibration_path": "",
        "classifier_index": 0,
        "show_debug": True,
    }

_params = default_params()
_params_lock = threading.Lock()

def get_params() -> dict:
    with _params_lock:
        return _params.copy()

def set_param(key: str, value: Any) -> None:
    with _params_lock:
        _params[key] = value


def load_calibration_yaml(path: str) -> Optional[dict]:
    try:
        try:
            import yaml
        except ImportError:
            return None
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not data:
            return None
        cal = {
            "base_ratio": float(data["base_ratio"]),
            "distortion_k": float(data["distortion_k"]),
            "distortion_p_y": float(data.get("distortion_p_y", 0)),
            "distortion_p_x": float(data.get("distortion_p_x", 0)),
            "img_center": (float(data["img_center"][0]), float(data["img_center"][1])),
            "max_dist_sq": float(data["max_dist_sq"]),
            "width_half": float(data.get("width_half", 1)),
            "height_half": float(data.get("height_half", 1)),
            "height_px": float(data.get("height_px", 0)),
            "use_height_px": bool(data.get("use_height_px", False)),
        }
        if cal["height_px"] > 0 and not cal["use_height_px"]:
            cal["use_height_px"] = True
        return cal
    except Exception:
        return None

def load_calibration_pkl(path: str) -> Optional[dict]:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def load_calibration(path: str) -> Optional[dict]:
    if not path or not os.path.isfile(path):
        return None
    if path.lower().endswith((".yaml", ".yml")):
        return load_calibration_yaml(path)
    return load_calibration_pkl(path)

def pixel_diameter_to_mm(cx: float, cy: float, diameter_px: float, ratio_px_to_mm: float, calibration: Optional[dict]) -> float:
    if calibration is None:
        return diameter_px * ratio_px_to_mm
    img_cx, img_cy = calibration["img_center"]
    dist_sq = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) / calibration["max_dist_sq"]
    base_ratio = calibration["base_ratio"]
    k_rad = calibration["distortion_k"]
    p_y = calibration.get("distortion_p_y", 0.0)
    p_x = calibration.get("distortion_p_x", 0.0)
    width_half = calibration.get("width_half", 1.0)
    if calibration.get("use_height_px") and calibration.get("height_px"):
        y_norm = (calibration["height_px"] - cy) / calibration["height_px"]
    else:
        y_norm = (cy - img_cy) / calibration.get("height_half", 1.0)
    x_norm = (cx - img_cx) / width_half
    correction = 1.0 + (k_rad * dist_sq) + (p_y * y_norm) + (p_x * x_norm)
    return (diameter_px * base_ratio) * correction

def diameter_mm_to_radius_px(cx: float, cy: float, diameter_mm: float, ratio_px_to_mm: float, calibration: Optional[dict]) -> int:
    if calibration is None:
        return max(2, int((diameter_mm / ratio_px_to_mm) / 2))
    img_cx, img_cy = calibration["img_center"]
    dist_sq = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) / calibration["max_dist_sq"]
    base_ratio = calibration["base_ratio"]
    k_rad = calibration["distortion_k"]
    p_y = calibration.get("distortion_p_y", 0.0)
    p_x = calibration.get("distortion_p_x", 0.0)
    width_half = calibration.get("width_half", 1.0)
    if calibration.get("use_height_px") and calibration.get("height_px"):
        y_norm = (calibration["height_px"] - cy) / calibration["height_px"]
    else:
        y_norm = (cy - img_cy) / calibration.get("height_half", 1.0)
    x_norm = (cx - img_cx) / width_half
    correction = 1.0 + (k_rad * dist_sq) + (p_y * y_norm) + (p_x * x_norm)
    return max(2, int((diameter_mm / (base_ratio * correction)) / 2))


def load_scaler_yaml(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    try:
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            return None, None, None
        mean_node, scale_node = fs.getNode("mean"), fs.getNode("scale")
        if mean_node.empty() or scale_node.empty():
            fs.release()
            return None, None, None
        mean, scale = mean_node.mat(), scale_node.mat()
        model_type = "SVM"
        mt_node = fs.getNode("model_type")
        if not mt_node.empty():
            model_type = mt_node.string() or "SVM"
        fs.release()
        return mean, scale, model_type
    except Exception:
        return None, None, None

def load_opencv_classifier(model_path: str, model_type: str):
    if not os.path.isfile(model_path):
        return None
    try:
        if model_type == "KNN":
            return cv2.ml.KNearest_load(model_path)
        if model_type == "RandomForest":
            return cv2.ml.RTrees_load(model_path)
        if model_type == "NaiveBayes":
            return cv2.ml.NormalBayesClassifier_load(model_path)
        return cv2.ml.SVM_load(model_path)
    except Exception:
        return None


class ClassifierHolder:
    def __init__(self):
        self.model = None
        self.mean = None
        self.scale = None
        self.name = "None"

    def load(self, build_dir: str, index: int) -> bool:
        build = Path(build_dir)
        mean, scale, model_type = load_scaler_yaml(str(build / SCALER_NAME))
        if mean is None or scale is None:
            self.model = None
            self.name = CLASSIFIER_NAMES[index] + " (no scaler)"
            return False
        model = load_opencv_classifier(str(build / CLASSIFIER_MODEL_NAMES[index]), model_type)
        if model is None:
            self.model = None
            self.mean, self.scale = mean, scale
            self.name = CLASSIFIER_NAMES[index] + " (load failed)"
            return False
        self.model, self.mean, self.scale = model, mean, scale
        self.name = CLASSIFIER_NAMES[index]
        return True

    def predict(self, diameter_mm: float, L: float, a: float, b: float) -> int:
        if self.model is None or self.mean is None or self.scale is None:
            return 0
        feat = np.array([[diameter_mm, L, a, b]], dtype=np.float32)
        for c in range(4):
            s = self.scale.flat[c]
            if s > 1e-9:
                feat[0, c] = (feat[0, c] - self.mean.flat[c]) / s
        ret = self.model.predict(feat)
        return int(ret[1].flat[0]) if isinstance(ret, tuple) and len(ret) > 1 and ret[1].size else int(ret)


class CornerStabilizer:
    def __init__(self, window_size: int = 5):
        self._history = collections.deque(maxlen=max(1, window_size))

    def set_window(self, n: int) -> None:
        self._history = collections.deque(self._history, maxlen=max(1, n))

    def update(self, corners: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if corners is None:
            return None
        self._history.append(corners.copy())
        return np.mean(self._history, axis=0).astype(np.float32)


@dataclass
class CoinTrack:
    center: Tuple[int, int]
    diameters: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=100))
    frames_since_seen: int = 0

    def accept_diameter(self, diameter_mm: float, max_deviation: float) -> bool:
        if len(self.diameters) == 0:
            self.diameters.append(diameter_mm)
            return True
        if abs(diameter_mm - float(np.median(self.diameters))) <= max_deviation:
            self.diameters.append(diameter_mm)
            return True
        return False

    def stable_diameter_mm(self, min_samples: int) -> Optional[float]:
        return float(np.median(self.diameters)) if len(self.diameters) >= min_samples else None

    def tick_missed(self) -> None:
        self.frames_since_seen += 1

    def mark_seen(self) -> None:
        self.frames_since_seen = 0


class CoinTracker:
    def __init__(self, params: dict):
        self._tracks: List[CoinTrack] = []
        self._params = params

    def update_params(self, params: dict) -> None:
        self._params = params

    def update(self, detections: List[Tuple[Tuple[int, int], float]]) -> None:
        p = self._params
        center_match = p["CENTER_MATCH_PX"]
        max_missing = p["MAX_FRAMES_MISSING"]
        max_dev = p["MAX_DIAMETER_DEVIATION_MM"]
        for t in self._tracks:
            t.tick_missed()
        used = [False] * len(self._tracks)
        for (cx, cy), diameter_mm in detections:
            best_i, best_dist = None, center_match + 1.0
            for i, track in enumerate(self._tracks):
                if used[i]:
                    continue
                d = np.hypot(cx - track.center[0], cy - track.center[1])
                if d < best_dist:
                    best_dist, best_i = d, i
            if best_i is not None:
                t = self._tracks[best_i]
                t.center = (cx, cy)
                t.accept_diameter(diameter_mm, max_dev)
                t.mark_seen()
                used[best_i] = True
            else:
                new_track = CoinTrack(center=(cx, cy))
                new_track.diameters = collections.deque(maxlen=p["DIAMETER_HISTORY_LEN"])
                new_track.accept_diameter(diameter_mm, max_dev)
                new_track.mark_seen()
                self._tracks.append(new_track)
                used.append(True)
        min_samples = p["MIN_SAMPLES_FOR_STABLE"]
        self._tracks = [
            t for i, t in enumerate(self._tracks)
            if (i < len(used) and used[i]) or t.frames_since_seen <= max_missing
        ]

    def get_stable_entries(self) -> List[Tuple[Tuple[int, int], float]]:
        min_samples = self._params["MIN_SAMPLES_FOR_STABLE"]
        return [(t.center, t.stable_diameter_mm(min_samples)) for t in self._tracks if t.stable_diameter_mm(min_samples) is not None]


def preprocess_for_circles(frame: np.ndarray, p: dict) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=p["CLAHE_CLIP"], tileGridSize=(p["CLAHE_GRID"], p["CLAHE_GRID"]))
    enhanced = clahe.apply(gray)
    k = p["BLUR_KSIZE"] | 1
    return cv2.medianBlur(enhanced, k)

def find_circle_candidates(blurred: np.ndarray, p: dict) -> Optional[np.ndarray]:
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=p["HOUGH_DP"], minDist=p["HOUGH_MIN_DIST"],
        param1=p["HOUGH_PARAM1"], param2=p["HOUGH_PARAM2"],
        minRadius=p["MIN_RADIUS_PX"], maxRadius=p["MAX_RADIUS_PX"],
    )
    if circles is None:
        return None
    return np.round(circles[0, :]).astype(np.int32)

def measure_circle_diameter(frame_gray: np.ndarray, x: int, y: int, r: int, ratio_px_to_mm: float, calibration: Optional[dict], p: dict) -> Optional[Tuple[Tuple[int, int], float]]:
    h, w = frame_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < p["MIN_CONTOUR_AREA"]:
        return None
    perimeter = cv2.arcLength(cnt, True)
    if perimeter <= 0:
        return None
    if 4 * np.pi * (area / (perimeter * perimeter)) < p["MIN_CIRCULARITY"]:
        return None
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    cx, cy = int(cx), int(cy)
    diameter_mm = pixel_diameter_to_mm(cx, cy, 2.0 * radius, ratio_px_to_mm, calibration)
    if not (p["DIAMETER_MM_MIN"] <= diameter_mm <= p["DIAMETER_MM_MAX"]):
        return None
    return ((cx, cy), float(diameter_mm))

def detect_and_measure_coins(frame: np.ndarray, ratio_px_to_mm: float, calibration: Optional[dict], p: dict) -> List[Tuple[Tuple[int, int], float]]:
    blurred = preprocess_for_circles(frame, p)
    circles = find_circle_candidates(blurred, p)
    if circles is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = []
    for row in circles:
        x, y, r = int(row[0]), int(row[1]), int(row[2])
        res = measure_circle_diameter(gray, x, y, r, ratio_px_to_mm, calibration, p)
        if res is not None:
            detections.append(res)
    kept = []
    for (c, d) in detections:
        if not any(np.hypot(c[0] - k[0][0], c[1] - k[0][1]) < p["CENTER_MATCH_PX"] for k in kept):
            kept.append((c, d))
    return kept

def sample_mean_lab_inside_circle(frame_bgr: np.ndarray, center: Tuple[int, int], radius_px: int, inner_frac: float) -> Optional[Tuple[float, float, float]]:
    h, w = frame_bgr.shape[:2]
    cx, cy = center
    inner_r = max(2, int(radius_px * inner_frac))
    y0, y1 = max(0, cy - inner_r), min(h, cy + inner_r + 1)
    x0, x1 = max(0, cx - inner_r), min(w, cx + inner_r + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    roi = lab[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (inner_r ** 2)
    if not np.any(mask):
        return None
    return (float(np.mean(roi[mask, 0])), float(np.mean(roi[mask, 1])), float(np.mean(roi[mask, 2])))

def collect_coin_features(frame_bgr: np.ndarray, entries: List[Tuple[Tuple[int, int], float]], ratio_px_to_mm: float, calibration: Optional[dict], p: dict) -> List[Tuple[float, float, float, float]]:
    rows = []
    for (cx, cy), diameter_mm in entries:
        r = diameter_mm_to_radius_px(cx, cy, diameter_mm, ratio_px_to_mm, calibration)
        lab = sample_mean_lab_inside_circle(frame_bgr, (cx, cy), r, p["inner_radius_frac"])
        if lab is not None:
            rows.append((diameter_mm, lab[0], lab[1], lab[2]))
    return rows

def _ensure_odd(k: int) -> int:
    return max(1, k | 1)


def find_paper_corners(frame: np.ndarray, p: dict) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    gray_filtered = cv2.medianBlur(gray_blurred, 7)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(gray_filtered)
    lines = lines[0] if lines is not None and len(lines) > 0 else []
    line_mask = np.zeros_like(gray_filtered)
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        if np.hypot(x2 - x1, y2 - y1) > p["paper_line_min_length"]:
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
    k = _ensure_odd(p["paper_morph_kernel"])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.dilate(cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel), kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        eps = p["paper_approx_eps_factor"] * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) == 4 and cv2.contourArea(approx) > p["paper_min_area"]:
            return approx.reshape(4, 2).astype(np.float32)
    return None

def order_corners(corners: np.ndarray) -> np.ndarray:
    y_sorted = corners[np.argsort(corners[:, 1])]
    top, bottom = y_sorted[:2], y_sorted[2:]
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = top[np.argmin(top[:, 0])]
    rect[1] = top[np.argmax(top[:, 0])]
    rect[3] = bottom[np.argmin(bottom[:, 0])]
    rect[2] = bottom[np.argmax(bottom[:, 0])]
    return rect

def draw_coins_and_histogram(frame: np.ndarray, tracker: CoinTracker, ratio_px_to_mm: float, calibration: Optional[dict], classifier: ClassifierHolder, p: dict) -> np.ndarray:
    display = frame.copy()
    entries = tracker.get_stable_entries()
    rows = collect_coin_features(frame, entries, ratio_px_to_mm, calibration, p)
    total_eur = 0.0
    for i, ((cx, cy), diameter_mm) in enumerate(entries):
        r = diameter_mm_to_radius_px(cx, cy, diameter_mm, ratio_px_to_mm, calibration)
        display_diameter_mm = pixel_diameter_to_mm(cx, cy, 2.0 * r, ratio_px_to_mm, calibration) if calibration else diameter_mm
        color = (0, 255, 0)
        if i < len(rows):
            cid = classifier.predict(rows[i][0], rows[i][1], rows[i][2], rows[i][3]) % 6
            color = CLUSTER_COLORS_BGR[cid]
            total_eur += CLASS_TO_VALUE_EUR[cid]
        cv2.circle(display, (cx, cy), r, color, 2)
        cv2.putText(display, f"{display_diameter_mm:.1f}mm", (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.rectangle(display, (10, 10), (280, 90), (0, 0, 0), -1)
    cv2.rectangle(display, (10, 10), (280, 90), (255, 255, 255), 2)
    cv2.putText(display, f"Coins: {len(entries)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, f"Total: {total_eur:.2f} EUR", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, f"Clf: {classifier.name} (1-4)", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if p["show_debug"] and entries:
        diameters = [d for (_, d) in entries]
        n_bins = int((p["HIST_BIN_MAX"] - p["HIST_BIN_MIN"]) / p["HIST_BIN_STEP"])
        counts = [0] * n_bins
        for d in diameters:
            bi = int((d - p["HIST_BIN_MIN"]) / p["HIST_BIN_STEP"])
            if 0 <= bi < n_bins:
                counts[bi] += 1
        max_count = max(max(counts), 1)
        hw, hh = p["HIST_WIDTH"], p["HIST_HEIGHT"]
        hist_img = np.ones((hh, hw, 3), dtype=np.uint8) * 255
        bar_w = max(1, (hw - 40) // n_bins - 2)
        for i in range(n_bins):
            bar_h = int((counts[i] / max_count) * (hh - 50))
            x1 = 30 + i * (bar_w + 2)
            cv2.rectangle(hist_img, (x1, hh - 30 - bar_h), (x1 + bar_w, hh - 30), (180, 130, 70), -1)
            cv2.rectangle(hist_img, (x1, hh - 30 - bar_h), (x1 + bar_w, hh - 30), (50, 50, 50), 1)
        cv2.putText(hist_img, "Diameter (mm)", (hw // 2 - 50, hh - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(hist_img, f"n={len(diameters)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow("Diameter histogram", hist_img)
    elif p["show_debug"]:
        hw, hh = p["HIST_WIDTH"], p["HIST_HEIGHT"]
        hist_img = np.ones((hh, hw, 3), dtype=np.uint8) * 255
        cv2.putText(hist_img, "No diameters", (hw // 2 - 50, hh // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.imshow("Diameter histogram", hist_img)
    return display


def run_camera_thread(stop_event: threading.Event, classifier_holder: ClassifierHolder, calibration_path_var: tk.StringVar) -> None:
    p = get_params()
    cap = cv2.VideoCapture(p["camera_index"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, p["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, p["frame_height"])
    ratio_px_to_mm = 1.0 / p["SCALE_FACTOR"]
    width_px = int(p["PAPER_WIDTH_MM"] * p["SCALE_FACTOR"])
    height_px = int(p["PAPER_HEIGHT_MM"] * p["SCALE_FACTOR"])
    dst_corners = np.array([[0, 0], [width_px - 1, 0], [width_px - 1, height_px - 1], [0, height_px - 1]], dtype=np.float32)
    stabilizer = CornerStabilizer(p["stabilizer_window"])
    tracker = CoinTracker(p)
    calibration = load_calibration(calibration_path_var.get().strip() or "") if calibration_path_var.get().strip() else None

    while not stop_event.is_set():
        p = get_params()
        tracker.update_params(p)
        if stabilizer._history.maxlen != p["stabilizer_window"]:
            stabilizer.set_window(p["stabilizer_window"])
        cal_path = calibration_path_var.get().strip() or ""
        if cal_path and os.path.isfile(cal_path):
            calibration = load_calibration(cal_path)

        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        try:
            ratio_px_to_mm = 1.0 / p["SCALE_FACTOR"]
            width_px = int(p["PAPER_WIDTH_MM"] * p["SCALE_FACTOR"])
            height_px = int(p["PAPER_HEIGHT_MM"] * p["SCALE_FACTOR"])
            dst_corners = np.array([[0, 0], [width_px - 1, 0], [width_px - 1, height_px - 1], [0, height_px - 1]], dtype=np.float32)
            raw_corners = find_paper_corners(frame, p)
            stable_corners = stabilizer.update(raw_corners)
            if stable_corners is not None:
                rect = order_corners(stable_corners)
                M = cv2.getPerspectiveTransform(rect, dst_corners)
                if abs(np.linalg.det(M)) > 1e-6:
                    warped = cv2.warpPerspective(frame, M, (width_px, height_px))
                    if warped.size > 0:
                        keep = p["keep_frac"]
                        cw, ch = int(warped.shape[1] * keep), int(warped.shape[0] * keep)
                        cx, cy = (warped.shape[1] - cw) // 2, (warped.shape[0] - ch) // 2
                        roi = warped[cy : cy + ch, cx : cx + cw].copy()
                        detections = detect_and_measure_coins(roi, ratio_px_to_mm, calibration, p)
                        detections = [((c[0][0] + cx, c[0][1] + cy), c[1]) for c in detections]
                        tracker.update(detections)
                        display = draw_coins_and_histogram(warped, tracker, ratio_px_to_mm, calibration, classifier_holder, p)
                        cv2.imshow("Anti-Glare Detection", display)
                        if p["show_debug"]:
                            cv2.imshow("Preprocess", preprocess_for_circles(warped, p))
                            cv2.imshow("Canny", cv2.Canny(preprocess_for_circles(warped, p), p["CANNY_THRESHOLD1"], p["CANNY_THRESHOLD2"]))
                        cv2.imshow("Warped", cv2.resize(warped, (0, 0), fx=0.5, fy=0.5))
            if raw_corners is not None:
                cv2.drawContours(frame, [raw_corners.astype(np.int32)], -1, (0, 255, 0), 2)
            cv2.imshow("Scanner", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set()
                break
            if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                idx = key - ord("1")
                if classifier_holder.load(p["build_dir"], idx):
                    set_param("classifier_index", idx)
        except Exception as e:
            print("Frame error:", e)
    cap.release()
    cv2.destroyAllWindows()


def add_section(parent: ttk.Frame, title: str) -> ttk.LabelFrame:
    return ttk.LabelFrame(parent, text=title, padding=4)

def _sync_int(key: str, var: tk.IntVar, spinbox: ttk.Spinbox) -> None:
    try:
        val = int(spinbox.get())
        var.set(val)
        set_param(key, val)
    except ValueError:
        pass

def _sync_float(key: str, var: tk.DoubleVar, spinbox: ttk.Spinbox) -> None:
    try:
        val = float(spinbox.get())
        var.set(val)
        set_param(key, val)
    except ValueError:
        pass

def add_param_int(parent: ttk.Frame, name: str, key: str, low: int, high: int, params: dict, vars_dict: dict) -> None:
    row = ttk.Frame(parent)
    row.pack(fill=tk.X, pady=1)
    ttk.Label(row, text=name, width=28, anchor=tk.W).pack(side=tk.LEFT)
    v = tk.IntVar(value=params[key])
    vars_dict[key] = v
    sb = ttk.Spinbox(row, from_=low, to=high, width=8, textvariable=v)
    sb.pack(side=tk.RIGHT)
    sb.bind("<Return>", lambda e: _sync_int(key, v, sb))
    sb.bind("<FocusOut>", lambda e: _sync_int(key, v, sb))

def add_param_float(parent: ttk.Frame, name: str, key: str, low: float, high: float, params: dict, vars_dict: dict) -> None:
    row = ttk.Frame(parent)
    row.pack(fill=tk.X, pady=1)
    ttk.Label(row, text=name, width=28, anchor=tk.W).pack(side=tk.LEFT)
    v = tk.DoubleVar(value=params[key])
    vars_dict[key] = v
    sb = ttk.Spinbox(row, from_=low, to=high, width=8, textvariable=v)
    sb.pack(side=tk.RIGHT)
    sb.bind("<Return>", lambda e: _sync_float(key, v, sb))
    sb.bind("<FocusOut>", lambda e: _sync_float(key, v, sb))

def build_param_panel(parent: ttk.Frame, calibration_path_var: tk.StringVar) -> dict:
    params = get_params()
    vars_dict = {}

    for f_title, items in [
        ("Camera", [
            ("camera_index", "Camera index", "int", 0, 10),
            ("frame_width", "Frame width", "int", 320, 1920),
            ("frame_height", "Frame height", "int", 240, 1080),
        ]),
        ("Paper & warp", [
            ("PAPER_WIDTH_MM", "Paper width (mm)", "float", 100, 500),
            ("PAPER_HEIGHT_MM", "Paper height (mm)", "float", 100, 400),
            ("SCALE_FACTOR", "Scale factor", "float", 1.0, 10.0),
            ("keep_frac", "Keep fraction (crop)", "float", 0.5, 1.0),
            ("stabilizer_window", "Stabilizer window", "int", 1, 30),
        ]),
        ("Paper detection", [
            ("paper_line_min_length", "Line min length", "int", 50, 300),
            ("paper_morph_kernel", "Morph kernel", "int", 3, 15),
            ("paper_approx_eps_factor", "Approx epsilon factor", "float", 0.01, 0.1),
            ("paper_min_area", "Paper min area", "int", 1000, 50000),
        ]),
        ("Preprocessing", [
            ("CLAHE_CLIP", "CLAHE clip", "int", 1, 20),
            ("CLAHE_GRID", "CLAHE grid", "int", 1, 8),
            ("BLUR_KSIZE", "Blur ksize (odd)", "int", 1, 15),
        ]),
        ("Canny", [
            ("CANNY_THRESHOLD1", "Canny threshold1", "float", 1, 255),
            ("CANNY_THRESHOLD2", "Canny threshold2", "float", 1, 255),
        ]),
        ("Hough circles", [
            ("HOUGH_DP", "Hough dp", "float", 1.0, 3.0),
            ("HOUGH_MIN_DIST", "Hough minDist", "int", 10, 200),
            ("HOUGH_PARAM1", "Hough param1", "float", 1, 200),
            ("HOUGH_PARAM2", "Hough param2", "float", 1, 100),
            ("MIN_RADIUS_PX", "Min radius (px)", "int", 5, 100),
            ("MAX_RADIUS_PX", "Max radius (px)", "int", 10, 150),
        ]),
        ("Circle quality", [
            ("MIN_CONTOUR_AREA", "Min contour area", "float", 10, 500),
            ("MIN_CIRCULARITY", "Min circularity", "float", 0.1, 1.0),
            ("DIAMETER_MM_MIN", "Diameter mm min", "float", 5, 25),
            ("DIAMETER_MM_MAX", "Diameter mm max", "float", 25, 50),
        ]),
        ("Tracking", [
            ("CENTER_MATCH_PX", "Center match (px)", "int", 5, 50),
            ("DIAMETER_HISTORY_LEN", "Diameter history len", "int", 10, 200),
            ("MAX_FRAMES_MISSING", "Max frames missing", "int", 1, 30),
            ("MAX_DIAMETER_DEVIATION_MM", "Max diameter deviation mm", "float", 0.5, 10.0),
            ("MIN_SAMPLES_FOR_STABLE", "Min samples stable", "int", 1, 20),
        ]),
        ("Histogram", [
            ("HIST_WIDTH", "Hist width", "int", 100, 600),
            ("HIST_HEIGHT", "Hist height", "int", 100, 400),
            ("HIST_BIN_MIN", "Hist bin min", "float", 5, 20),
            ("HIST_BIN_MAX", "Hist bin max", "float", 30, 50),
            ("HIST_BIN_STEP", "Hist bin step", "float", 0.1, 1.0),
        ]),
        ("LAB", [
            ("inner_radius_frac", "Inner radius fraction", "float", 0.3, 1.0),
        ]),
    ]:
        f = add_section(parent, f_title)
        f.pack(fill=tk.X, padx=2, pady=2)
        for key, name, typ, low, high in items:
            if typ == "int":
                add_param_int(f, name, key, low, high, params, vars_dict)
            else:
                add_param_float(f, name, key, float(low), float(high), params, vars_dict)

    f = add_section(parent, "Paths & classifier")
    f.pack(fill=tk.X, padx=2, pady=2)
    row = ttk.Frame(f)
    row.pack(fill=tk.X, pady=1)
    ttk.Label(row, text="Build dir (classifiers)", width=28, anchor=tk.W).pack(side=tk.LEFT)
    build_var = tk.StringVar(value=params["build_dir"])
    vars_dict["build_dir"] = build_var
    ttk.Entry(row, textvariable=build_var, width=24).pack(side=tk.RIGHT, fill=tk.X, expand=True)
    row = ttk.Frame(f)
    row.pack(fill=tk.X, pady=1)
    ttk.Label(row, text="Calibration (YAML or pkl)", width=28, anchor=tk.W).pack(side=tk.LEFT)
    calibration_path_var.set(params.get("calibration_path") or "")
    ttk.Entry(row, textvariable=calibration_path_var, width=24).pack(side=tk.RIGHT, fill=tk.X, expand=True)
    row = ttk.Frame(f)
    row.pack(fill=tk.X, pady=2)
    ttk.Label(row, text="Classifier 1-4", width=28, anchor=tk.W).pack(side=tk.LEFT)
    clf_var = tk.IntVar(value=params["classifier_index"])
    vars_dict["classifier_index"] = clf_var
    for i in range(4):
        ttk.Radiobutton(row, text=str(i + 1), variable=clf_var, value=i, command=lambda i=i: set_param("classifier_index", i)).pack(side=tk.LEFT)
    row = ttk.Frame(f)
    row.pack(fill=tk.X, pady=1)
    show_debug_var = tk.BooleanVar(value=params["show_debug"])
    vars_dict["show_debug"] = show_debug_var
    ttk.Checkbutton(row, text="Show debug windows", variable=show_debug_var, command=lambda: set_param("show_debug", show_debug_var.get())).pack(anchor=tk.W)
    return vars_dict


def main_gui() -> None:
    root = tk.Tk()
    root.title("Coin counter — tune parameters (same as C++)")
    root.geometry("420x820")
    calibration_path_var = tk.StringVar()
    for path in [str(BUILD_DIR / "coin_calibration_robust.yaml"), "coin_calibration_robust.pkl", "coin_calibration_robust.yaml"]:
        if os.path.isfile(path):
            calibration_path_var.set(path)
            set_param("calibration_path", path)
            break

    classifier_holder = ClassifierHolder()
    classifier_holder.load(get_params()["build_dir"], get_params()["classifier_index"])

    canvas = tk.Canvas(root)
    scroll = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
    frame = ttk.Frame(canvas)
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=frame, anchor=tk.NW)
    canvas.configure(yscrollcommand=scroll.set)
    vars_dict = build_param_panel(frame, calibration_path_var)

    def on_start():
        for k, v in vars_dict.items():
            if isinstance(v, (tk.IntVar, tk.DoubleVar, tk.StringVar)):
                try:
                    set_param(k, v.get())
                except Exception:
                    pass
        set_param("calibration_path", calibration_path_var.get().strip())
        classifier_holder.load(get_params()["build_dir"], get_params()["classifier_index"])
        stop = threading.Event()
        t = threading.Thread(target=run_camera_thread, args=(stop, classifier_holder, calibration_path_var), daemon=True)
        t.start()
        start_btn.config(state=tk.DISABLED)

    start_btn = ttk.Button(frame, text="Start camera (q in video to quit)", command=on_start)
    start_btn.pack(pady=8)
    def browse_calibration():
        path = filedialog.askopenfilename(title="Calibration file", filetypes=[("YAML/PKL", "*.yaml *.yml *.pkl"), ("All", "*")])
        if path:
            calibration_path_var.set(path)
            set_param("calibration_path", path)

    ttk.Button(frame, text="Browse calibration…", command=browse_calibration).pack(pady=2)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    root.mainloop()


if __name__ == "__main__":
    main_gui()
