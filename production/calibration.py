"""
Camera-to-grid floor calibration via homography.

Setup (one-time):
  1. Place physical markers at 4+ known grid positions on the factory floor.
  2. In the app, enter calibration mode and click each marker in the camera
     image, then type the corresponding grid (row, col).
  3. Calibration is computed and saved to data/calibration.json.

After calibration, pixel_to_grid() converts any camera pixel into a
(row, col) float for use by the production engine.

grid_to_pixel() is the inverse — used to draw planned paths on the camera.
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

CALIB_FILE = Path("data/calibration.json")

MIN_PIXEL_SPREAD = 10.0   # px: closer clicks count as the same image point
MIN_GRID_AREA    = 4.0    # cells²: smaller grid hulls are (nearly) collinear
MAX_POINT_ERROR  = 3.0    # cells: worse fit than this → probably a mistyped point


def _hull_area(pts: np.ndarray) -> float:
    return float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))


class FloorCalibration:
    def __init__(self, path: Path = CALIB_FILE):
        self._path = Path(path)
        self._pixel_pts: list[list[float]] = []   # [[px, py], ...]
        self._grid_pts:  list[list[float]] = []   # [[col, row], ...] (x=col for cv2)
        self._H:  Optional[np.ndarray] = None     # pixel → grid homography (3×3)
        self._Hi: Optional[np.ndarray] = None     # grid → pixel (inverse)
        self.problem: str = ""                    # why the points give no calibration
        self.warning: str = ""                    # calibrated, but a point fits badly
        self.max_error_cells: float = 0.0         # worst point's fit error
        self.image_size: Optional[tuple[int, int]] = None   # (w, h) the points belong to

    def match_image_size(self, w: int, h: int) -> str:
        """Called with each frame's size. Pixel points belong to one camera
        resolution: on a change with the same aspect ratio they are rescaled;
        otherwise the calibration is unusable until redone. Returns a message
        when something changed, '' otherwise."""
        if self.image_size == (w, h):
            return ""
        old, self.image_size = self.image_size, (w, h)
        if old is None or not self._pixel_pts:
            return ""                               # first frame: just record it
        ow, oh = old
        if abs(w / h - ow / oh) < 0.01:
            sx, sy = w / ow, h / oh
            self._pixel_pts = [[x * sx, y * sy] for x, y in self._pixel_pts]
            if len(self._pixel_pts) >= 4:
                self._recompute()
            self.save()
            return (f"Camera resolution changed {ow}×{oh} → {w}×{h} — "
                    f"calibration points rescaled")
        # Different framing: no safe way to move the points. Not saved, so the
        # old set stays on disk for the old resolution.
        self._clear_H()
        self.problem = (f"camera resolution changed {ow}×{oh} → {w}×{h} (different "
                        f"framing) — clear with ↺ Calib and click the points again")
        return f"Calibration unusable: {self.problem}"

    @property
    def is_calibrated(self) -> bool:
        return self._H is not None

    @property
    def point_count(self) -> int:
        return len(self._pixel_pts)

    # ── Point collection ──────────────────────────────────────────────────────

    def add_point(self, px: float, py: float, row: int, col: int) -> None:
        self._pixel_pts.append([px, py])
        self._grid_pts.append([float(col), float(row)])  # cv2 uses (x, y) = (col, row)
        if len(self._pixel_pts) >= 4:
            self._recompute()

    def remove_last(self) -> None:
        if self._pixel_pts:
            self._pixel_pts.pop()
            self._grid_pts.pop()
            self._recompute() if len(self._pixel_pts) >= 4 else self._clear_H()

    def clear(self) -> None:
        self._pixel_pts.clear()
        self._grid_pts.clear()
        self._clear_H()

    def _clear_H(self) -> None:
        self._H = None
        self._Hi = None
        self.problem = ""
        self.warning = ""
        self.max_error_cells = 0.0

    def _recompute(self) -> None:
        """Fit the homography and check that it makes sense. On failure the
        calibration is cleared and self.problem says why."""
        self._clear_H()
        src = np.array(self._pixel_pts, dtype=np.float32)
        dst = np.array(self._grid_pts,  dtype=np.float32)

        # Degenerate input: findHomography would fail or return nonsense
        for i in range(len(src)):
            for j in range(i):
                if np.linalg.norm(src[i] - src[j]) < MIN_PIXEL_SPREAD:
                    self.problem = f"points {j + 1} and {i + 1} are the same spot in the image"
                    return
                if np.array_equal(dst[i], dst[j]):
                    r, c = int(dst[i][1]), int(dst[i][0])
                    self.problem = (f"points {j + 1} and {i + 1} both have grid ({r},{c}) — "
                                    f"was the default 0 accepted?")
                    return
        if _hull_area(dst) < MIN_GRID_AREA:
            self.problem = "the grid coordinates lie on one line — spread them over the floor"
            return
        if _hull_area(src) < MIN_PIXEL_SPREAD ** 2:
            self.problem = "the clicked points lie on one line in the image"
            return

        # Least squares over all points (RANSAC would silently drop a typo)
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
            self.problem = "no consistent mapping — check the typed grid coordinates"
            return
        # All points must be on the same side of the horizon (w > 0 or all < 0);
        # otherwise the mapping folds the floor through infinity
        w = (np.hstack([src, np.ones((len(src), 1), np.float32)]) @ H.T)[:, 2]
        if not (np.all(w > 0) or np.all(w < 0)):
            self.problem = "points don't fit one floor plane — a grid coordinate is mistyped"
            return

        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        err  = np.linalg.norm(proj - dst, axis=1)
        self._H  = H
        self._Hi = np.linalg.inv(H)
        self.max_error_cells = float(err.max())
        if self.max_error_cells > MAX_POINT_ERROR:
            worst = int(err.argmax())
            self.warning = (f"point {worst + 1} is off by {err[worst]:.1f} cells — "
                            f"mistyped? right-click removes the last point")

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def pixel_to_grid(self, px: float, py: float) -> tuple[float, float]:
        """Returns (row, col) floats. Returns (0, 0) if not calibrated."""
        if self._H is None:
            return (0.0, 0.0)
        pt = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H)
        col, row = result[0, 0]
        return (float(row), float(col))

    def pixels_to_grid_xy(self, pts: np.ndarray) -> np.ndarray:
        """Vectorised pixel_to_grid: (N, 2) pixel (x, y) → (N, 2) grid (col, row)."""
        if self._H is None:
            return np.zeros((len(pts), 2), np.float32)
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self._H).reshape(-1, 2)

    def floor_polygon_px(self, rows: int, cols: int,
                         row0: int = 0, col0: int = 0) -> Optional[np.ndarray]:
        """Grid rectangle [row0, rows) × [col0, cols) projected into the camera
        image, (4, 2) pixels."""
        if self._Hi is None:
            return None
        corners = np.array([[[col0, row0]], [[cols, row0]], [[cols, rows]], [[col0, rows]]],
                           np.float32)
        return cv2.perspectiveTransform(corners, self._Hi).reshape(-1, 2)

    def grid_to_pixel(self, row: float, col: float) -> tuple[float, float]:
        """Returns (px, py) floats. Returns (0, 0) if not calibrated."""
        if self._Hi is None:
            return (0.0, 0.0)
        pt = np.array([[[col, row]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._Hi)
        px, py = result[0, 0]
        return (float(px), float(py))

    def floor_mask(self, rows: int, cols: int) -> Optional[np.ndarray]:
        """(rows, cols) bool: grid cells inside the floor area the calibration
        points enclose — the part of the grid the camera actually covers."""
        if self._H is None:
            return None
        hull = cv2.convexHull(np.array(self._grid_pts, np.float32))   # (col, row)
        mask = np.zeros((rows, cols), np.uint8)
        cv2.fillPoly(mask, [np.round(hull).astype(np.int32)], 1)
        return mask.astype(bool)

    def calibration_points(self) -> list[tuple[float, float, int, int]]:
        """Returns list of (px, py, row, col) for all registered points."""
        return [
            (p[0], p[1], int(g[1]), int(g[0]))
            for p, g in zip(self._pixel_pts, self._grid_pts)
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({
            "pixel_pts": self._pixel_pts,
            "grid_pts":  self._grid_pts,
            "H":         self._H.tolist() if self._H is not None else None,
            "image_size": list(self.image_size) if self.image_size else None,
        }, indent=2))

    def load(self) -> bool:
        """Load saved points (partial sets included) and refit from them."""
        if not self._path.exists():
            return False
        try:
            data = json.loads(self._path.read_text())
            self._pixel_pts = [list(map(float, p)) for p in data["pixel_pts"]]
            self._grid_pts  = [list(map(float, g)) for g in data["grid_pts"]]
            size = data.get("image_size")
            self.image_size = (int(size[0]), int(size[1])) if size else None
        except Exception:
            self._pixel_pts, self._grid_pts = [], []
            self._clear_H()
            return False
        if len(self._pixel_pts) >= 4:
            self._recompute()
        else:
            self._clear_H()
        return self._H is not None
