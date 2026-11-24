"""OpenCV camera wrapper with Qt signal output.

Cameras are opened at their highest supported resolution (OpenCV's default
is 640×480 whatever the camera can do). On Windows the DirectShow backend is
used — it negotiates resolution and MJPG reliably — for both listing and
opening, so device indices match between the two.
"""

import os
import sys
from typing import Tuple

import cv2
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QImage

_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

# Tried largest first; the first one the camera really delivers is used.
# 1920 by default: the overhead camera shows the same field of view at 1080p
# and 4K, but all processing runs on the UI thread and 4K frames take ~100 ms
# each (sluggish UI) against ~35 ms at 1080p. Raise to 3840 only if markers
# get too small to detect.
MAX_LONG_SIDE = 1920
_RESOLUTIONS = [
    (3840, 2160), (2592, 1944), (2560, 1440), (2048, 1536), (1920, 1080),
    (1600, 1200), (1280, 960), (1280, 720), (1024, 768), (800, 600), (640, 480),
]


def _open(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, _BACKEND)
    if not cap.isOpened() and _BACKEND != cv2.CAP_ANY:
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap


def _use_max_resolution(cap: cv2.VideoCapture, verify: bool = True) -> Tuple[int, int]:
    """Switch the camera to its largest supported resolution. MJPG first:
    uncompressed USB video at high resolution drops to a few fps."""
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    for w, h in _RESOLUTIONS:
        if max(w, h) > MAX_LONG_SIDE:
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (w, h):
            continue
        if not verify:
            return w, h
        ok, frame = cap.read()                 # some drivers accept, then don't deliver
        if ok and frame is not None and frame.shape[:2] == (h, w):
            return w, h
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


def list_available_cameras(max_check: int = 8) -> list[tuple[int, str]]:
    """Return [(index, label)] for each camera that opens successfully,
    labelled with its maximum resolution.

    Probing indices that don't exist generates noisy OpenCV warnings;
    stderr is temporarily redirected to suppress them.
    """
    results = []
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved   = os.dup(2)
    os.dup2(devnull, 2)
    try:
        for i in range(max_check):
            cap = _open(i)
            if cap.isOpened():
                w, h = _use_max_resolution(cap, verify=False)
                cap.release()
                results.append((i, f"Camera {i}  (up to {w}×{h})"))
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)
    return results


class CameraSource(QObject):
    frame_ready = pyqtSignal(QImage)   # emits an RGB888 QImage each capture

    def __init__(self, device_index: int, fps: int = 30, parent=None):
        super().__init__(parent)
        self._cap = _open(device_index)
        if self._cap.isOpened():
            _use_max_resolution(self._cap)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._capture)
        self._interval = max(10, 1000 // fps)

    @property
    def resolution(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def is_open(self) -> bool:
        return self._cap.isOpened()

    def start(self) -> None:
        if self._cap.isOpened():
            self._timer.start(self._interval)

    def stop(self) -> None:
        self._timer.stop()

    def release(self) -> None:
        self.stop()
        if self._cap.isOpened():
            self._cap.release()

    def _capture(self) -> None:
        ok, frame = self._cap.read()
        if not ok:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self.frame_ready.emit(img.copy())
