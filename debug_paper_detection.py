"""
Debug tool for paper detection and warping.
Live camera feed with sliders for all paper/warp parameters; updates in real time.
Press 'q' to quit.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np

# Defaults match Config
DEFAULTS = {
    "paper_line_min_length": 100,
    "paper_morph_kernel": 3,
    "paper_approx_eps_factor": 0.02,
    "paper_min_area": 10000,
    "PAPER_WIDTH_MM": 330.0,
    "PAPER_HEIGHT_MM": 216.0,
    "SCALE_FACTOR": 3.0,
    "stabilizer_window": 10,
}

# Global state for trackbar callbacks (OpenCV requires a single ref)
_state: dict = {}


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


class CornerStabilizer:
    def __init__(self, window_size: int = 5):
        self._history: Deque[np.ndarray] = collections.deque(maxlen=max(1, window_size))

    def set_window(self, n: int) -> None:
        self._history = collections.deque(self._history, maxlen=max(1, n))

    def update(self, corners: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if corners is None:
            return None
        self._history.append(corners.copy())
        return np.mean(self._history, axis=0).astype(np.float32)


def get_params() -> dict:
    s = _state
    return {
        "paper_line_min_length": s.get("paper_line_min_length", DEFAULTS["paper_line_min_length"]),
        "paper_morph_kernel": _ensure_odd(s.get("paper_morph_kernel", DEFAULTS["paper_morph_kernel"])),
        "paper_approx_eps_factor": s.get("paper_approx_eps_factor", DEFAULTS["paper_approx_eps_factor"]),
        "paper_min_area": s.get("paper_min_area", DEFAULTS["paper_min_area"]),
        "PAPER_WIDTH_MM": float(s.get("PAPER_WIDTH_MM", DEFAULTS["PAPER_WIDTH_MM"])),
        "PAPER_HEIGHT_MM": float(s.get("PAPER_HEIGHT_MM", DEFAULTS["PAPER_HEIGHT_MM"])),
        "SCALE_FACTOR": float(s.get("SCALE_FACTOR", DEFAULTS["SCALE_FACTOR"])),
        "stabilizer_window": s.get("stabilizer_window", DEFAULTS["stabilizer_window"]),
    }


def main() -> None:
    global _state
    _state = dict(DEFAULTS)

    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Could not open camera (index 2). Try changing camera_index in the script.")
        return

    # SCALE_FACTOR 1.0-5.0: slider 10-50 -> value = x/10
    cv2.namedWindow("Paper detection")
    cv2.createTrackbar("line_min_len", "Paper detection", DEFAULTS["paper_line_min_length"], 300, lambda v: _state.__setitem__("paper_line_min_length", v))
    cv2.createTrackbar("morph_kernel", "Paper detection", DEFAULTS["paper_morph_kernel"], 15, lambda v: _state.__setitem__("paper_morph_kernel", max(1, (v | 1))))
    cv2.createTrackbar("approx_eps (1-100)", "Paper detection", 12, 100, lambda v: _state.__setitem__("paper_approx_eps_factor", 0.01 + (v / 100.0) * 0.09))
    cv2.createTrackbar("min_area/1000", "Paper detection", DEFAULTS["paper_min_area"] // 1000, 50, lambda v: _state.__setitem__("paper_min_area", max(1000, v * 1000)))
    cv2.createTrackbar("paper_w_mm", "Paper detection", int(DEFAULTS["PAPER_WIDTH_MM"]), 500, lambda v: _state.__setitem__("PAPER_WIDTH_MM", float(v)))
    cv2.createTrackbar("paper_h_mm", "Paper detection", int(DEFAULTS["PAPER_HEIGHT_MM"]), 400, lambda v: _state.__setitem__("PAPER_HEIGHT_MM", float(v)))
    cv2.createTrackbar("scale x10", "Paper detection", int(DEFAULTS["SCALE_FACTOR"] * 10), 50, lambda v: _state.__setitem__("SCALE_FACTOR", v / 10.0))
    cv2.createTrackbar("stabilizer", "Paper detection", DEFAULTS["stabilizer_window"], 30, lambda v: _state.__setitem__("stabilizer_window", max(1, v)))

    stabilizer: Optional[CornerStabilizer] = None

    while True:
        p = get_params()
        if stabilizer is None or stabilizer._history.maxlen != p["stabilizer_window"]:
            stabilizer = CornerStabilizer(p["stabilizer_window"])

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        raw_corners = find_paper_corners(frame, p)
        stable_corners = stabilizer.update(raw_corners)

        display = frame.copy()
        warped_display = None

        if raw_corners is not None:
            cv2.drawContours(display, [raw_corners.astype(np.int32)], -1, (0, 255, 0), 2)
            cv2.putText(display, "Paper found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if stable_corners is not None and len(stable_corners) == 4:
            rect = order_corners(stable_corners)
            w_px = int(p["PAPER_WIDTH_MM"] * p["SCALE_FACTOR"])
            h_px = int(p["PAPER_HEIGHT_MM"] * p["SCALE_FACTOR"])
            dst = np.array([[0, 0], [w_px - 1, 0], [w_px - 1, h_px - 1], [0, h_px - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(rect, dst)
            if abs(np.linalg.det(M)) > 1e-6:
                warped = cv2.warpPerspective(frame, M, (w_px, h_px))
                warped_display = warped

        # Show camera + contour (resized to fit)
        scale_show = 0.6
        small = cv2.resize(display, (0, 0), fx=scale_show, fy=scale_show)
        cv2.imshow("Paper detection", small)

        if warped_display is not None:
            small_warped = cv2.resize(warped_display, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("Warped", small_warped)
        else:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blank, "No stable paper", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Warped", blank)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
