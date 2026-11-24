"""
YOLO-based person detector for the production camera feed.

Runs YOLOv11-nano (yolo11n.pt) on BGR frames asynchronously — one inference
at a time in a daemon thread so it never blocks the camera loop. The caller
reads `.latest()` each frame; it returns the most recent completed results.

Only COCO class 0 (person) is detected. The model is downloaded automatically
by ultralytics on first use (~6 MB for the nano variant).
"""

import threading
from dataclasses import dataclass

import numpy as np

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO = None
    _YOLO_AVAILABLE = False

DEFAULT_MODEL  = "yolo11n.pt"
DEFAULT_CONF   = 0.45   # minimum confidence to report a detection
PERSON_CLASS   = 0      # COCO class index for "person"


@dataclass
class PersonDetection:
    x1: float          # pixel bounding box (top-left)
    y1: float
    x2: float          # pixel bounding box (bottom-right)
    y2: float
    confidence: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class PersonDetector:
    """Thread-safe YOLO person detector.  Call detect_async() each frame;
    read latest() to get the most recent completed detection list."""

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 conf_threshold: float = DEFAULT_CONF):
        self._conf   = conf_threshold
        self._lock   = threading.Lock()
        self._busy   = False
        self._results: list[PersonDetection] = []
        self._enabled = True
        self._model  = None

        if _YOLO_AVAILABLE:
            # Load in a background thread so startup isn't blocked
            threading.Thread(target=self._load_model,
                             args=(model_name,), daemon=True).start()

    @property
    def available(self) -> bool:
        return _YOLO_AVAILABLE and self._model is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if not value:
            with self._lock:
                self._results = []

    def _load_model(self, model_name: str) -> None:
        try:
            model = _YOLO(model_name)
            model.fuse()   # merge BN into conv for faster inference
            self._model = model
        except Exception:
            self._model = None

    def detect_async(self, frame_bgr: np.ndarray) -> None:
        """Submit a frame for detection. Skipped if a previous inference is
        still running or the detector is disabled / not yet loaded."""
        if not self._enabled or not self.available or self._busy:
            return
        self._busy = True
        threading.Thread(target=self._infer,
                         args=(frame_bgr.copy(),), daemon=True).start()

    def _infer(self, frame_bgr: np.ndarray) -> None:
        try:
            results = self._model(
                frame_bgr,
                classes=[PERSON_CLASS],
                conf=self._conf,
                verbose=False,
                imgsz=640,
            )
            dets: list[PersonDetection] = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    dets.append(PersonDetection(x1, y1, x2, y2, conf))
            with self._lock:
                self._results = dets
        except Exception:
            pass
        finally:
            self._busy = False

    def latest(self) -> list[PersonDetection]:
        """Return the most recent detection results (thread-safe)."""
        with self._lock:
            return list(self._results)
