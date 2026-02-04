"""
Debug tool for coin detection using watershed segmentation.
Live camera: paper detection + warp, then watershed on the warped image.
Visualizes all steps; adaptive + Otsu thresholding; many tunable parameters.
Press 'q' to quit.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple

import cv2
import numpy as np

# Watershed + preprocessing defaults (from tuned GUI)
DEFAULTS = {
    "CHANNEL_MODE": 2,  # 0=Gray, 1=H, 2=S, 3=V, 4=L, 5=A, 6=B
    "CLAHE_CLIP": 9,
    "CLAHE_GRID": 1,  # GUI showed 0; grid must be >= 1
    "BLUR_KSIZE": 9,
    "use_adaptive": 0,  # 0=Otsu, 1=Adaptive
    "adaptive_block": 23,
    "adaptive_C": 10,  # slider 60 → stored 10 (trackbar: v - 50)
    "invert_binary": 1,
    "morph_open_size": 1,
    "morph_close_size": 3,
    "morph_open_iters": 2,
    "morph_close_iters": 4,
    "bg_dilate_size": 4,
    "dist_mask_size": 5,  # 3 or 5 (slider 1 → 5)
    "watershed_fg_frac": 0.45,  # FG frac x100 = 57
    "MIN_CONTOUR_AREA": 464,
    "MIN_CIRCULARITY": 0.56,
    "DIAMETER_MM_MIN": 10.0,
    "DIAMETER_MM_MAX": 40.0,
    "CENTER_MATCH_PX": 20,
    "SCALE_FACTOR": 5.0,  # Scale x10 = 50
    "paper_line_min_length": 100,
    "paper_morph_kernel": 3,
    "paper_approx_eps_factor": 0.02,
    "paper_min_area": 10000,
    "PAPER_WIDTH_MM": 330.0,
    "PAPER_HEIGHT_MM": 216.0,
    "stabilizer_window": 10,
}


@dataclass
class WatershedSteps:
    """All intermediate images for visualization."""
    blurred: Optional[np.ndarray] = None
    gradient_for_watershed: Optional[np.ndarray] = None  # image actually passed to watershed
    binary: Optional[np.ndarray] = None
    after_open: Optional[np.ndarray] = None
    after_close: Optional[np.ndarray] = None
    sure_bg: Optional[np.ndarray] = None
    sure_fg: Optional[np.ndarray] = None
    dist_vis: Optional[np.ndarray] = None
    unknown: Optional[np.ndarray] = None
    markers_vis: Optional[np.ndarray] = None
    watershed_vis: Optional[np.ndarray] = None
    detections: List[Tuple[Tuple[int, int], float]] = field(default_factory=list)

_state: dict = {}


def _ensure_odd(k: int) -> int:
    return max(1, k | 1)


def get_params() -> dict:
    s = _state
    blur_k = s.get("BLUR_KSIZE", DEFAULTS["BLUR_KSIZE"])
    return {
        "CLAHE_CLIP": max(1, s.get("CLAHE_CLIP", DEFAULTS["CLAHE_CLIP"])),
        "CLAHE_GRID": max(1, s.get("CLAHE_GRID", DEFAULTS["CLAHE_GRID"])),
        "BLUR_KSIZE": _ensure_odd(blur_k),
        "CHANNEL_MODE": min(6, max(0, int(s.get("CHANNEL_MODE", DEFAULTS["CHANNEL_MODE"])))),
        "use_adaptive": int(s.get("use_adaptive", DEFAULTS["use_adaptive"])),
        "adaptive_block": _ensure_odd(min(51, max(3, s.get("adaptive_block", DEFAULTS["adaptive_block"])))),
        "adaptive_C": int(s.get("adaptive_C", DEFAULTS["adaptive_C"])),
        "invert_binary": int(s.get("invert_binary", DEFAULTS["invert_binary"])),
        "morph_open_size": _ensure_odd(max(1, s.get("morph_open_size", DEFAULTS["morph_open_size"]))),
        "morph_close_size": _ensure_odd(max(1, s.get("morph_close_size", DEFAULTS["morph_close_size"]))),
        "morph_open_iters": max(0, s.get("morph_open_iters", DEFAULTS["morph_open_iters"])),
        "morph_close_iters": max(0, s.get("morph_close_iters", DEFAULTS["morph_close_iters"])),
        "bg_dilate_size": _ensure_odd(max(1, s.get("bg_dilate_size", DEFAULTS["bg_dilate_size"]))),
        "dist_mask_size": 5 if s.get("dist_mask_size", DEFAULTS["dist_mask_size"]) >= 4 else 3,
        "watershed_fg_frac": float(s.get("watershed_fg_frac", DEFAULTS["watershed_fg_frac"])),
        "MIN_CONTOUR_AREA": float(s.get("MIN_CONTOUR_AREA", DEFAULTS["MIN_CONTOUR_AREA"])),
        "MIN_CIRCULARITY": float(s.get("MIN_CIRCULARITY", DEFAULTS["MIN_CIRCULARITY"])),
        "DIAMETER_MM_MIN": float(s.get("DIAMETER_MM_MIN", DEFAULTS["DIAMETER_MM_MIN"])),
        "DIAMETER_MM_MAX": float(s.get("DIAMETER_MM_MAX", DEFAULTS["DIAMETER_MM_MAX"])),
        "CENTER_MATCH_PX": int(s.get("CENTER_MATCH_PX", DEFAULTS["CENTER_MATCH_PX"])),
        "SCALE_FACTOR": float(s.get("SCALE_FACTOR", DEFAULTS["SCALE_FACTOR"])),
        "paper_line_min_length": s.get("paper_line_min_length", DEFAULTS["paper_line_min_length"]),
        "paper_morph_kernel": _ensure_odd(s.get("paper_morph_kernel", DEFAULTS["paper_morph_kernel"])),
        "paper_approx_eps_factor": s.get("paper_approx_eps_factor", DEFAULTS["paper_approx_eps_factor"]),
        "paper_min_area": s.get("paper_min_area", DEFAULTS["paper_min_area"]),
        "PAPER_WIDTH_MM": float(s.get("PAPER_WIDTH_MM", DEFAULTS["PAPER_WIDTH_MM"])),
        "PAPER_HEIGHT_MM": float(s.get("PAPER_HEIGHT_MM", DEFAULTS["PAPER_HEIGHT_MM"])),
        "stabilizer_window": s.get("stabilizer_window", DEFAULTS["stabilizer_window"]),
    }


def find_paper_corners(frame: np.ndarray, p: dict) -> Optional[np.ndarray]:
    """Same pipeline as debug_paper_detection.py."""
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
    k = p["paper_morph_kernel"]
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


class CornerStabilizer:
    def __init__(self, window_size: int = 10):
        self._history: Deque[np.ndarray] = collections.deque(maxlen=max(1, window_size))

    def update(self, corners: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if corners is None:
            return None
        self._history.append(corners.copy())
        return np.mean(self._history, axis=0).astype(np.float32)


def preprocess_for_circles(frame: np.ndarray, p: dict) -> np.ndarray:
    # 0=Gray, 1=H, 2=S, 3=V, 4=L, 5=A, 6=B
    mode = p.get("CHANNEL_MODE", 0)
    if mode == 0:
        ch = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif mode in (1, 2, 3):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ch = hsv[:, :, mode - 1]
    else:  # 4=L, 5=A, 6=B
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        ch = lab[:, :, mode - 4]
    clahe = cv2.createCLAHE(clipLimit=p["CLAHE_CLIP"], tileGridSize=(p["CLAHE_GRID"], p["CLAHE_GRID"]))
    enhanced = clahe.apply(ch)
    k = _ensure_odd(p["BLUR_KSIZE"])
    return cv2.medianBlur(enhanced, k)


def _show_step(img: Optional[np.ndarray], scale: float = 0.5, title: str = "") -> np.ndarray:
    """Resize and optionally add title; return 3-channel for imshow."""
    if img is None or img.size == 0:
        out = np.zeros((200, 280, 3), dtype=np.uint8)
        cv2.putText(out, "N/A", (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        if len(img.shape) == 2:
            out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            out = img.copy()
        out = cv2.resize(out, (0, 0), fx=scale, fy=scale)
        if title:
            cv2.putText(out, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def detect_coins_watershed(
    frame: np.ndarray,
    ratio_px_to_mm: float,
    p: dict,
) -> WatershedSteps:
    """
    Watershed segmentation with adaptive or Otsu thresholding.
    Returns WatershedSteps with all intermediate images for visualization.
    """
    steps = WatershedSteps()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = preprocess_for_circles(frame, p)
    steps.blurred = blurred

    # 1. Threshold: Adaptive or Otsu
    if p.get("use_adaptive", 1):
        block = p["adaptive_block"]
        C = p["adaptive_C"]
        # Adaptive: local mean/gaussian threshold (coins often darker -> THRESH_BINARY_INV for foreground = coins)
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block, C,
        )
    else:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if p.get("invert_binary", 1):
        binary = cv2.bitwise_not(binary)
    steps.binary = binary.copy()

    # 2. Morphological open (remove noise)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["morph_open_size"], p["morph_open_size"]))
    after_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=p["morph_open_iters"])
    steps.after_open = after_open.copy()

    # 3. Morphological close (fill holes inside coins)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["morph_close_size"], p["morph_close_size"]))
    after_close = cv2.morphologyEx(after_open, cv2.MORPH_CLOSE, k_close, iterations=p["morph_close_iters"])
    steps.after_close = after_close.copy()
    binary = after_close

    # 4. Sure background
    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["bg_dilate_size"], p["bg_dilate_size"]))
    sure_bg = cv2.dilate(binary, k_bg)
    steps.sure_bg = sure_bg.copy()

    # 5. Distance transform -> sure foreground
    dsize = p.get("dist_mask_size", 5)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, dsize)
    dist_max = dist.max()
    steps.dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if dist_max > 0 else binary

    if dist_max <= 0:
        return steps
    frac = max(0.2, min(0.6, p.get("watershed_fg_frac", 0.4)))
    _, sure_fg = cv2.threshold(dist, frac * dist_max, 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    steps.sure_fg = sure_fg.copy()

    # 6. Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    steps.unknown = unknown.copy()

    # 7. Markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers.astype(np.int32)
    markers = markers + 1
    markers[unknown == 255] = 0
    # Visualize markers (color by label)
    np.random.seed(42)
    markers_vis = np.zeros((*markers.shape, 3), dtype=np.uint8)
    for lid in range(1, num_labels + 1):
        c = np.random.randint(50, 255, 3).tolist()
        markers_vis[markers == lid] = c
    steps.markers_vis = markers_vis

    # 8. Watershed
    watershed_input = cv2.cvtColor(steps.after_close, cv2.COLOR_GRAY2BGR)
    markers_out = markers.copy()
    cv2.imshow("Watershed input", watershed_input)
    cv2.watershed(watershed_input, markers_out)


    # 9. Extract regions and measure; build watershed vis
    detections: List[Tuple[Tuple[int, int], float]] = []
    watershed_vis = watershed_input
    colors = [np.array(np.random.randint(50, 255, 3), dtype=np.float64) for _ in range(num_labels + 1)]

    for label in range(2, num_labels + 1):
        mask = np.uint8(markers_out == label)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < p["MIN_CONTOUR_AREA"]:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < p["MIN_CIRCULARITY"]:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < 1:
            continue
        diameter_px = 2.0 * radius
        diameter_mm = diameter_px * ratio_px_to_mm
        if not (p["DIAMETER_MM_MIN"] <= diameter_mm <= p["DIAMETER_MM_MAX"]):
            continue
        detections.append(((int(cx), int(cy)), float(diameter_mm)))
        cvec = colors[label]
        watershed_vis[mask.astype(bool)] = (watershed_vis[mask.astype(bool)] * 0.4 + cvec * 0.6).clip(0, 255).astype(np.uint8)

    kept: List[Tuple[Tuple[int, int], float]] = []
    for (c, d) in detections:
        if not any(np.hypot(c[0] - k[0][0], c[1] - k[0][1]) < p["CENTER_MATCH_PX"] for k in kept):
            kept.append((c, d))

    steps.detections = kept
    steps.watershed_vis = watershed_vis
    return steps


def detect_and_measure_coins(frame: np.ndarray, ratio_px_to_mm: float, p: dict) -> List[Tuple[Tuple[int, int], float]]:
    """Use watershed segmentation for coin detection."""
    steps = detect_coins_watershed(frame, ratio_px_to_mm, p)
    return steps.detections


def main() -> None:
    global _state
    _state = dict(DEFAULTS)

    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Could not open camera (index 2). Try changing camera_index in the script.")
        return

    def set_state(key: str, val: Any) -> None:
        _state[key] = val

    win = "Coin detection"
    cv2.namedWindow(win)
    scale_vis = 0.45

    # Preprocessing (channel: 0=Gray, 1=H, 2=S, 3=V, 4=L, 5=A, 6=B)
    cv2.createTrackbar("Channel 0-6", win, DEFAULTS["CHANNEL_MODE"], 6, lambda v: set_state("CHANNEL_MODE", v))
    cv2.createTrackbar("CLAHE clip", win, DEFAULTS["CLAHE_CLIP"], 20, lambda v: set_state("CLAHE_CLIP", max(1, v)))
    cv2.createTrackbar("CLAHE grid", win, DEFAULTS["CLAHE_GRID"], 8, lambda v: set_state("CLAHE_GRID", max(1, v)))
    cv2.createTrackbar("Blur ksize", win, DEFAULTS["BLUR_KSIZE"], 15, lambda v: set_state("BLUR_KSIZE", _ensure_odd(max(1, v))))
    # Threshold
    cv2.createTrackbar("0=Otsu 1=Adapt", win, DEFAULTS["use_adaptive"], 1, lambda v: set_state("use_adaptive", v))
    cv2.createTrackbar("Adapt block", win, DEFAULTS["adaptive_block"], 51, lambda v: set_state("adaptive_block", _ensure_odd(max(3, v))))
    cv2.createTrackbar("Adapt C", win, DEFAULTS["adaptive_C"] + 50, 100, lambda v: set_state("adaptive_C", v - 50))
    cv2.createTrackbar("Invert bin", win, DEFAULTS["invert_binary"], 1, lambda v: set_state("invert_binary", v))
    # Morph
    cv2.createTrackbar("Open size", win, DEFAULTS["morph_open_size"], 15, lambda v: set_state("morph_open_size", _ensure_odd(max(1, v))))
    cv2.createTrackbar("Close size", win, DEFAULTS["morph_close_size"], 21, lambda v: set_state("morph_close_size", _ensure_odd(max(1, v))))
    cv2.createTrackbar("Open iters", win, DEFAULTS["morph_open_iters"], 5, lambda v: set_state("morph_open_iters", v))
    cv2.createTrackbar("Close iters", win, DEFAULTS["morph_close_iters"], 5, lambda v: set_state("morph_close_iters", v))
    cv2.createTrackbar("BG dilate", win, DEFAULTS["bg_dilate_size"], 21, lambda v: set_state("bg_dilate_size", _ensure_odd(max(1, v))))
    cv2.createTrackbar("Dist mask 3/5", win, 1, 1, lambda v: set_state("dist_mask_size", 5 if v else 3))
    # Watershed + filters
    cv2.createTrackbar("FG frac x100", win, int(DEFAULTS["watershed_fg_frac"] * 100), 60, lambda v: set_state("watershed_fg_frac", max(20, v) / 100.0))
    cv2.createTrackbar("Min area", win, int(DEFAULTS["MIN_CONTOUR_AREA"]), 500, lambda v: set_state("MIN_CONTOUR_AREA", float(max(1, v))))
    cv2.createTrackbar("Circ x100", win, int(DEFAULTS["MIN_CIRCULARITY"] * 100), 100, lambda v: set_state("MIN_CIRCULARITY", v / 100.0))
    cv2.createTrackbar("D mm min", win, int(DEFAULTS["DIAMETER_MM_MIN"]), 25, lambda v: set_state("DIAMETER_MM_MIN", float(v)))
    cv2.createTrackbar("D mm max", win, int(DEFAULTS["DIAMETER_MM_MAX"]), 50, lambda v: set_state("DIAMETER_MM_MAX", float(v)))
    cv2.createTrackbar("Center match", win, DEFAULTS["CENTER_MATCH_PX"], 50, lambda v: set_state("CENTER_MATCH_PX", max(1, v)))
    cv2.createTrackbar("Scale x10", win, int(DEFAULTS["SCALE_FACTOR"] * 10), 50, lambda v: set_state("SCALE_FACTOR", v / 10.0))

    stabilizer: Optional[CornerStabilizer] = None
    placeholder = np.zeros((220, 280, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Align paper", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    while True:
        p = get_params()
        if stabilizer is None or stabilizer._history.maxlen != p["stabilizer_window"]:
            stabilizer = CornerStabilizer(p["stabilizer_window"])

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        raw_corners = find_paper_corners(frame, p)
        stable_corners = stabilizer.update(raw_corners)
        warped = None
        if stable_corners is not None and len(stable_corners) == 4:
            rect = order_corners(stable_corners)
            w_px = int(p["PAPER_WIDTH_MM"] * p["SCALE_FACTOR"])
            h_px = int(p["PAPER_HEIGHT_MM"] * p["SCALE_FACTOR"])
            dst = np.array([[0, 0], [w_px - 1, 0], [w_px - 1, h_px - 1], [0, h_px - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(rect, dst)
            if abs(np.linalg.det(M)) > 1e-6:
                warped = cv2.warpPerspective(frame, M, (w_px, h_px))

        if warped is not None:
            ratio_px_to_mm = 1.0 / p["SCALE_FACTOR"]
            steps = detect_coins_watershed(warped, ratio_px_to_mm, p)

            result = warped.copy()
            for (cx, cy), diameter_mm in steps.detections:
                r = max(2, int((diameter_mm / ratio_px_to_mm) / 2.0))
                cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
                cv2.putText(result, f"{diameter_mm:.1f}mm", (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(result, f"Coins: {len(steps.detections)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(win, _show_step(result, scale_vis, "Result"))
            cv2.imshow("1 Blurred", _show_step(steps.blurred, scale_vis, "1 Blurred"))
            cv2.imshow("1b Gradient", _show_step(steps.gradient_for_watershed, scale_vis, "1b Grad (watershed input)"))
            cv2.imshow("2 Binary", _show_step(steps.binary, scale_vis, "2 Binary (Adapt/Otsu)"))
            cv2.imshow("3 After Open", _show_step(steps.after_open, scale_vis, "3 Morph Open"))
            cv2.imshow("4 After Close", _show_step(steps.after_close, scale_vis, "4 Morph Close"))
            cv2.imshow("5 Sure BG", _show_step(steps.sure_bg, scale_vis, "5 Sure background"))
            cv2.imshow("6 Sure FG", _show_step(steps.sure_fg, scale_vis, "6 Sure foreground"))
            cv2.imshow("7 Dist Transform", _show_step(steps.dist_vis, scale_vis, "7 Distance"))
            cv2.imshow("8 Unknown", _show_step(steps.unknown, scale_vis, "8 Unknown"))
            cv2.imshow("9 Markers", _show_step(steps.markers_vis, scale_vis, "9 Markers"))
            cv2.imshow("10 Watershed", _show_step(steps.watershed_vis, scale_vis, "10 Segments"))
        else:
            cv2.imshow(win, placeholder)
            for name in ["1 Blurred", "1b Gradient", "2 Binary", "3 After Open", "4 After Close", "5 Sure BG", "6 Sure FG",
                        "7 Dist Transform", "8 Unknown", "9 Markers", "10 Watershed"]:
                cv2.imshow(name, placeholder)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
