"""
ArUco-based cart detection and tracking.

Each physical cart has a unique ArUco marker (DICT_4X4_50) mounted on top,
visible to the overhead camera.

CartDetector — single-frame marker detection, tuned for the real floor:
  • grayscale input, wider adaptive-threshold range, sub-pixel corners
  • geometric validation (square-ish, convex, plausible size) to reject
    large dark rectangles (toolboxes, cards) that decode as random IDs
  • detect_roi(): re-search a small window with local contrast normalisation,
    which recovers markers washed out by specular glare (glossy paper turns
    its black cells light grey under the lamp hotspot)

CartTracker — temporal layer on top of the detector, one track per marker:
  1. full-frame detection, gated against each track's predicted position
  2. ROI re-search around the prediction when the marker is missed
  3. foreground-blob fallback (needs a mask from ObstacleDetector): cart body
     centroid + body axis, heading disambiguated with the last known heading
  4. constant-velocity prediction for a few frames, then the track is dropped
Every output carries `source` so the controller can decide how much to trust it.

Angle convention: 0° = marker top edge pointing right in camera image;
positive = counterclockwise (standard math convention, not screen).
The angle is derived from the top-left→top-right corner vector of the marker.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np


ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

# Marker sources, from most to least trustworthy
SRC_MARKER    = "marker"      # full-frame ArUco detection
SRC_ROI       = "marker_roi"  # ArUco found by the local re-search
SRC_BLOB      = "blob"        # cart body from the foreground mask
SRC_PREDICTED = "predicted"   # no measurement — extrapolated
SRC_MEMORY    = "memory"      # lost mid-floor: still there, position = last seen

ACQUIRE_EVERY = 3             # frames between glare searches for untracked markers

# Full-frame marker search runs on a copy downscaled to this long side, then
# the corners are refined on the full-resolution image. ArUco's cost grows
# with the pixel count (≈200 ms on a 1080p frame vs ≈50 ms at 960 px), and it
# runs on the UI thread. ROI re-searches around tracked carts stay full-res.
DETECT_LONG_SIDE = 960
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


@dataclass
class Detection:
    marker_id: int
    cx: float        # pixel x of marker centre
    cy: float        # pixel y of marker centre
    angle_deg: float # heading in degrees (0=right, 90=up in camera coords)
    corners: np.ndarray  # shape (4, 2) pixel corners, clockwise from top-left
    source: str = SRC_MARKER
    frames_lost: int = 0     # consecutive frames without a marker measurement
    frames_unmeasured: int = 0  # consecutive frames without any measurement
    vx: float = 0.0          # tracked velocity, px/frame
    vy: float = 0.0

    def footprint(self, scale: float, trail_frames: float = 2.0) -> np.ndarray:
        """Marker square scaled about its centre — approximates the cart body
        (marker side × scale) — swept back along the velocity over
        `trail_frames`, since the smoothed foreground mask lags a moving cart.
        Convex pixel polygon."""
        c   = np.array([self.cx, self.cy], np.float32)
        box = (self.corners - c) * scale + c
        if trail_frames <= 0 or (self.vx == 0 and self.vy == 0):
            return box
        back = box - np.array([self.vx, self.vy], np.float32) * trail_frames
        return cv2.convexHull(np.vstack([box, back])).reshape(-1, 2)

    @property
    def side_px(self) -> float:
        return float(np.mean(np.linalg.norm(
            self.corners - np.roll(self.corners, -1, axis=0), axis=1)))


def _make_params(subpix: bool = True) -> cv2.aruco.DetectorParameters:
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin  = 5
    p.adaptiveThreshWinSizeMax  = 53
    p.adaptiveThreshWinSizeStep = 8
    p.adaptiveThreshConstant    = 5
    p.minMarkerPerimeterRate    = 0.02
    p.polygonalApproxAccuracyRate = 0.05
    if subpix:
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


# Gamma > 1 stretches the bright tones: a marker washed out by glare keeps its
# contrast there (black cells turn light grey, white stays white)
_GLARE_LUT = np.clip(255.0 * (np.arange(256) / 255.0) ** 2.5, 0, 255).astype(np.uint8)


def glare_stretch(gray: np.ndarray) -> np.ndarray:
    return cv2.LUT(gray, _GLARE_LUT)


def local_normalize(gray: np.ndarray, k: int = 31) -> np.ndarray:
    """Per-pixel (x - local mean) / local std — flattens glare gradients."""
    f  = gray.astype(np.float32)
    mu = cv2.blur(f, (k, k))
    sd = np.sqrt(np.maximum(cv2.blur((f - mu) ** 2, (k, k)), 0)) + 4.0
    return np.clip((f - mu) / sd * 50.0 + 128.0, 0, 255).astype(np.uint8)


class CartDetector:
    def __init__(self, min_side_px: float = 12.0, max_side_frac: float = 0.25):
        aruco_dict     = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        self._det      = cv2.aruco.ArucoDetector(aruco_dict, _make_params())
        self._min_side = min_side_px
        self._max_side_frac = max_side_frac

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect all ArUco markers in a BGR (or grayscale) frame."""
        return self._detect_scaled(_gray(frame_bgr))

    def detect_glare(self, gray: np.ndarray) -> list[Detection]:
        """Full-frame search on the glare-stretched image (for acquisition)."""
        return self._detect_scaled(glare_stretch(gray), source=SRC_ROI)

    def _detect_scaled(self, img: np.ndarray, source: str = SRC_MARKER) -> list[Detection]:
        """detectMarkers on a downscaled copy; corners mapped back to full
        resolution and sub-pixel refined there."""
        s = min(1.0, DETECT_LONG_SIDE / max(img.shape[:2]))
        if s >= 1.0:
            corners, ids, _ = self._det.detectMarkers(img)
            return self._to_detections(corners, ids, img.shape, source=source)
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        corners, ids, _ = self._det.detectMarkers(small)
        if ids is None:
            return []
        corners = [self._refine(img, c.reshape(4, 2) / s, s) for c in corners]
        return self._to_detections(corners, ids, img.shape, source=source)

    @staticmethod
    def _refine(img: np.ndarray, pts: np.ndarray, s: float) -> np.ndarray:
        """Sub-pixel corners on the full-res image, searching the area one
        downscaled pixel could hide; keep the coarse corner if refinement
        wanders off (blur, glare)."""
        win  = max(2, int(round(1.5 / s)))
        fine = pts.astype(np.float32).reshape(-1, 1, 2).copy()
        try:
            cv2.cornerSubPix(img, fine, (win, win), (-1, -1), _SUBPIX_CRITERIA)
        except cv2.error:
            return pts.astype(np.float32).reshape(1, 4, 2)
        fine = fine.reshape(4, 2)
        moved = np.linalg.norm(fine - pts, axis=1)
        out = np.where((moved <= 1.5 / s)[:, None], fine, pts)
        return out.astype(np.float32).reshape(1, 4, 2)

    def detect_roi(self, gray: np.ndarray, cx: float, cy: float,
                   half: float, side_hint: float = 0.0) -> list[Detection]:
        """Re-search of a window centred on (cx, cy): local contrast
        normalisation first, then the glare stretch."""
        h, w = gray.shape[:2]
        x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
        x1, y1 = min(w, int(cx + half)), min(h, int(cy + half))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return []
        scale = 2.0 if 0 < side_hint < 40 else 1.0
        for prep in (local_normalize, glare_stretch):
            crop = prep(gray[y0:y1, x0:x1])
            if scale != 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            corners, ids, _ = self._det.detectMarkers(crop)
            if ids is not None:
                corners = [c / scale + np.array([x0, y0], np.float32) for c in corners]
                found = self._to_detections(corners, ids, gray.shape, source=SRC_ROI)
                if found:
                    return found
        return []

    def _to_detections(self, corners, ids, shape,
                       source: str = SRC_MARKER) -> list[Detection]:
        if ids is None:
            return []
        max_side = self._max_side_frac * max(shape[:2])
        results  = []
        for corner, mid in zip(corners, ids.flatten()):
            pts  = corner.reshape(4, 2).astype(np.float32)
            side = self._quad_side(pts)
            if side is None or not (self._min_side <= side <= max_side):
                continue
            cx, cy = pts.mean(axis=0)
            # Angle from top-left to top-right edge
            dx = pts[1, 0] - pts[0, 0]
            dy = pts[1, 1] - pts[0, 1]
            angle = float(np.degrees(np.arctan2(-dy, dx)))  # flip y for screen coords
            results.append(Detection(int(mid), float(cx), float(cy), angle, pts,
                                     source=source))
        return results

    @staticmethod
    def _quad_side(pts: np.ndarray) -> Optional[float]:
        """Mean side length if the quad looks like a square seen from above."""
        sides = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
        if sides.min() <= 0 or sides.max() / sides.min() > 1.5:
            return None
        d1 = np.linalg.norm(pts[0] - pts[2])
        d2 = np.linalg.norm(pts[1] - pts[3])
        if max(d1, d2) / max(min(d1, d2), 1e-6) > 1.3:
            return None
        if not cv2.isContourConvex(pts.reshape(-1, 1, 2)):
            return None
        return float(sides.mean())

    def annotate(self, frame_bgr: np.ndarray,
                 detections: list[Detection]) -> np.ndarray:
        """Return a copy of the frame with detection overlays drawn."""
        out = frame_bgr.copy()
        colours = {SRC_MARKER: (0, 255, 0), SRC_ROI: (0, 200, 255),
                   SRC_BLOB: (255, 160, 0), SRC_PREDICTED: (160, 160, 160)}
        for d in detections:
            colour = colours.get(d.source, (0, 255, 0))
            pts = d.corners.astype(np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=colour, thickness=2)
            cx, cy = int(d.cx), int(d.cy)
            # Orientation arrow
            r = np.radians(d.angle_deg)
            ex = int(cx + 40 * np.cos(r))
            ey = int(cy - 40 * np.sin(r))
            cv2.arrowedLine(out, (cx, cy), (ex, ey), colour, 2, tipLength=0.3)
            cv2.putText(out, f"{d.marker_id} {d.source}", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        return out


# ── Tracking ──────────────────────────────────────────────────────────────────

@dataclass
class _Track:
    marker_id: int
    x: float
    y: float
    angle: float
    corners_rel: np.ndarray            # corners relative to centre, at angle
    vx: float = 0.0
    vy: float = 0.0
    side: float = 0.0
    frames_lost: int = 0          # frames since the last marker measurement
    frames_unmeasured: int = 0    # frames since any measurement (marker or blob)
    source: str = SRC_MARKER
    body_offset: Optional[float] = None  # heading − body long axis, (-90, 90]
    blob_area: Optional[float] = None
    pending: List[Detection] = field(default_factory=list)  # far re-acquisitions
    mx: float = float("nan")      # last *measured* pose (marker or blob)
    my: float = float("nan")
    ma: float = 0.0

    def __post_init__(self):
        self.mark_measured()

    def mark_measured(self) -> None:
        self.mx, self.my, self.ma = self.x, self.y, self.angle

    def predict(self) -> tuple[float, float]:
        return self.x + self.vx, self.y + self.vy

    def as_memory(self) -> Detection:
        """Where the cart was last measured — it can't have gone anywhere else
        without being seen."""
        rot     = _rot(self.ma - _corner_angle(self.corners_rel))
        corners = self.corners_rel @ rot.T + np.array([self.mx, self.my], np.float32)
        return Detection(self.marker_id, self.mx, self.my, self.ma,
                         corners.astype(np.float32), source=SRC_MEMORY,
                         frames_lost=self.frames_lost,
                         frames_unmeasured=self.frames_unmeasured)

    def as_detection(self) -> Detection:
        rot     = _rot(self.angle - _corner_angle(self.corners_rel))
        corners = self.corners_rel @ rot.T + np.array([self.x, self.y], np.float32)
        return Detection(self.marker_id, self.x, self.y, self.angle,
                         corners.astype(np.float32), source=self.source,
                         frames_lost=self.frames_lost,
                         frames_unmeasured=self.frames_unmeasured,
                         vx=self.vx, vy=self.vy)


class CartTracker:
    """Keeps a smoothed pose per marker ID across frames (see module doc)."""

    POS_GAIN_MARKER = 0.85    # alpha-beta gains: trust markers more than blobs
    POS_GAIN_BLOB   = 0.6
    VEL_GAIN        = 0.3
    ANGLE_GAIN      = 0.7

    def __init__(self, detector: Optional[CartDetector] = None,
                 max_predict_frames: int = 15, max_blob_frames: int = 90,
                 max_dormant_frames: int = 300,
                 gate_sides: float = 3.0, reacquire_frames: int = 3):
        self.detector          = detector or CartDetector()
        self._tracks: Dict[int, _Track] = {}
        self._max_predict      = max_predict_frames  # reported with no measurement
        self._max_blob         = max_blob_frames     # body-blob tracking without a marker
        # Carts can't vanish mid-floor: a tracked (assigned) cart lost away from
        # the edge of the view stays reported as SRC_MEMORY at its last measured
        # position, indefinitely. Only one lost at the edge (it left the view)
        # is dropped, after max_dormant_frames of searching.
        self._max_dormant      = max_dormant_frames
        self._gate_sides       = gate_sides        # max jump, in marker sides
        self._reacquire_frames = reacquire_frames  # hits needed to accept a far jump
        self.raw: List[Detection] = []             # last full-frame detections
        self._frame_no = 0

    def reset(self) -> None:
        self._tracks.clear()

    def forget_memory(self) -> int:
        """Drop carts that are only remembered (not currently measured)."""
        gone = [m for m, t in self._tracks.items() if t.frames_unmeasured > self._max_predict]
        for m in gone:
            del self._tracks[m]
        return len(gone)

    def update(self, frame_bgr: np.ndarray,
               ids: Optional[Iterable[int]] = None,
               fg_mask: Optional[np.ndarray] = None,
               fg_scale: float = 1.0) -> List[Detection]:
        """Process one frame.

        ids      — marker IDs to track (None = every valid marker seen)
        fg_mask  — optional uint8 foreground mask (non-zero = not floor)
        fg_scale — fg_mask pixel size relative to the frame (0.5 = half res)
        """
        gray     = _gray(frame_bgr)
        wanted   = set(ids) if ids is not None else None
        self.raw = self.detector.detect(gray)
        by_id: Dict[int, Detection] = {}
        for d in self.raw:
            if wanted is None or d.marker_id in wanted:
                by_id.setdefault(d.marker_id, d)

        for mid, det in by_id.items():
            tr = self._tracks.get(mid)
            if tr is None:
                self._tracks[mid] = self._new_track(det)
            elif not self._size_ok(tr, det):
                by_id[mid] = None          # wrong-sized look-alike: treat as a miss
            elif self._gate_ok(tr, det):
                self._correct(tr, det, gray, fg_mask, fg_scale)
            else:
                self._handle_far(tr, det, gray, fg_mask, fg_scale)

        # Acquisition: assigned markers with no live track (e.g. a cart parked
        # under the lamp since start-up) get a periodic glare-stretched search
        self._frame_no += 1
        if wanted and self._frame_no % ACQUIRE_EVERY == 0:
            missing = {m for m in wanted
                       if by_id.get(m) is None and
                       (m not in self._tracks or self._tracks[m].frames_lost > 0)}
            if missing:
                for det in self.detector.detect_glare(gray):
                    if det.marker_id not in missing or by_id.get(det.marker_id) is not None:
                        continue
                    tr = self._tracks.get(det.marker_id)
                    if tr is None:
                        self._tracks[det.marker_id] = self._new_track(det)
                    elif not self._size_ok(tr, det):
                        continue
                    elif self._gate_ok(tr, det):
                        self._correct(tr, det, gray, fg_mask, fg_scale)
                    else:
                        self._handle_far(tr, det, gray, fg_mask, fg_scale)
                    by_id[det.marker_id] = det

        for mid, tr in list(self._tracks.items()):
            if wanted is not None and mid not in wanted:
                del self._tracks[mid]
                continue
            if by_id.get(mid) is not None:
                continue                       # measured, or handled by _handle_far
            self._recover(tr, gray, fg_mask, fg_scale)
            left_view = wanted is None or self._near_edge(tr, gray.shape)
            if tr.frames_unmeasured > self._max_dormant and left_view:
                del self._tracks[mid]

        out = []
        for tr in self._tracks.values():
            if tr.frames_unmeasured <= self._max_predict:
                out.append(tr.as_detection())
            elif wanted is not None and not self._near_edge(tr, gray.shape):
                out.append(tr.as_memory())
        return out

    @staticmethod
    def _near_edge(tr: "_Track", shape) -> bool:
        """Was the cart last measured close enough to the image border that it
        may have driven (or been carried) out of view?"""
        h, w   = shape[:2]
        margin = 1.5 * max(tr.side, 10.0)
        return not (margin <= tr.mx <= w - margin and margin <= tr.my <= h - margin)

    # ── Track maintenance ─────────────────────────────────────────────────────

    def _new_track(self, det: Detection) -> _Track:
        return _Track(det.marker_id, det.cx, det.cy, det.angle_deg,
                      det.corners - np.array([det.cx, det.cy], np.float32),
                      side=det.side_px, source=det.source)

    @staticmethod
    def _size_ok(tr: _Track, det: Detection) -> bool:
        return 0.6 * tr.side <= det.side_px <= 1.6 * tr.side

    def _gate_ok(self, tr: _Track, det: Detection) -> bool:
        px, py = tr.predict()
        gate   = self._gate_sides * max(tr.side, 10.0) * (1 + 0.2 * min(tr.frames_lost, 30))
        return math.hypot(det.cx - px, det.cy - py) <= gate

    def _handle_far(self, tr: _Track, det: Detection, gray, fg_mask, fg_scale) -> None:
        """A marker with this ID appeared far from the prediction. Accept it only
        after it repeats consistently (cart was moved, or the old track was wrong);
        meanwhile keep following the existing track."""
        if tr.pending and math.hypot(det.cx - tr.pending[-1].cx,
                                     det.cy - tr.pending[-1].cy) > self._gate_sides * tr.side:
            tr.pending.clear()
        tr.pending.append(det)
        if len(tr.pending) >= self._reacquire_frames:
            self._tracks[tr.marker_id] = self._new_track(det)
            return
        self._recover(tr, gray, fg_mask, fg_scale)

    def _correct(self, tr: _Track, det: Detection, gray, fg_mask, fg_scale) -> None:
        px, py = tr.predict()
        a      = self.POS_GAIN_MARKER
        rx, ry = det.cx - px, det.cy - py
        tr.x, tr.y   = px + a * rx, py + a * ry
        tr.vx       += self.VEL_GAIN * rx
        tr.vy       += self.VEL_GAIN * ry
        tr.angle     = _wrap(tr.angle + self.ANGLE_GAIN * _angle_diff(det.angle_deg, tr.angle))
        tr.corners_rel = det.corners - np.array([det.cx, det.cy], np.float32)
        tr.side      = 0.8 * tr.side + 0.2 * det.side_px if tr.side else det.side_px
        tr.frames_lost = 0
        tr.frames_unmeasured = 0
        tr.source    = det.source
        tr.mark_measured()
        tr.pending.clear()
        if fg_mask is not None:
            self._learn_body(tr, det, fg_mask, fg_scale)

    def _recover(self, tr: _Track, gray, fg_mask, fg_scale) -> None:
        px, py = tr.predict()
        half   = max(2.5 * tr.side, 60.0) * (1 + 0.15 * min(tr.frames_lost, 20))
        for det in self.detector.detect_roi(gray, px, py, half, tr.side):
            if det.marker_id == tr.marker_id and self._size_ok(tr, det) \
                    and self._gate_ok(tr, det):
                self._correct(tr, det, gray, fg_mask, fg_scale)
                return

        # The body blob only bridges short marker gaps (e.g. crossing the glare).
        # A cart already lost for longer — or remembered — is re-acquired by its
        # marker only: any cart-sized blob (a person, an object) would do
        # otherwise, and the cart would then be driven on a wrong position.
        blob = self._blob_measure(tr, px, py, fg_mask, fg_scale) \
            if fg_mask is not None and tr.frames_lost <= self._max_blob \
            and tr.frames_unmeasured <= self._max_predict else None
        tr.frames_lost += 1
        if blob is not None:
            bx, by, heading = blob
            a      = self.POS_GAIN_BLOB
            rx, ry = bx - px, by - py
            tr.x, tr.y = px + a * rx, py + a * ry
            tr.vx += self.VEL_GAIN * rx
            tr.vy += self.VEL_GAIN * ry
            if heading is not None:
                tr.angle = _wrap(tr.angle + 0.5 * _angle_diff(heading, tr.angle))
            tr.source = SRC_BLOB
            tr.frames_unmeasured = 0
            tr.mark_measured()
        else:
            tr.frames_unmeasured += 1
            tr.x, tr.y = px, py
            tr.vx *= 0.8          # decay — a lost cart is more likely stopping
            tr.vy *= 0.8
            tr.source = SRC_PREDICTED

    # ── Foreground-blob fallback ──────────────────────────────────────────────

    def _component_at(self, fg_mask, fg_scale, x, y, half):
        """Return (contour in frame px, area in frame px²) of the foreground
        component nearest (x, y) within a window, or None."""
        h, w = fg_mask.shape[:2]
        s    = fg_scale
        x0, y0 = max(0, int((x - half) * s)), max(0, int((y - half) * s))
        x1, y1 = min(w, int((x + half) * s)), min(h, int((y + half) * s))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        crop = (fg_mask[y0:y1, x0:x1] > 0).astype(np.uint8)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_d = None, None
        cxl, cyl = x * s - x0, y * s - y0
        for c in contours:
            d = -cv2.pointPolygonTest(c, (float(cxl), float(cyl)), True)  # <0 inside
            if best_d is None or d < best_d:
                best, best_d = c, d
        if best is None or best_d > 0.5 * half * s:
            return None
        cnt = (best.astype(np.float32) + np.array([x0, y0], np.float32)) / s
        return cnt, cv2.contourArea(cnt)

    def _learn_body(self, tr: _Track, det: Detection, fg_mask, fg_scale) -> None:
        comp = self._component_at(fg_mask, fg_scale, det.cx, det.cy, 2.5 * tr.side)
        if comp is None:
            return
        cnt, area = comp
        # The component may be merged with an obstacle the cart touches —
        # only learn from plausible cart-sized blobs.
        if tr.blob_area and not (0.5 * tr.blob_area <= area <= 2.0 * tr.blob_area):
            return
        if area < 1.5 * tr.side ** 2 or area > 12 * tr.side ** 2:
            return
        tr.blob_area = area if tr.blob_area is None else 0.9 * tr.blob_area + 0.1 * area
        axis, elongation = _body_axis(cnt)
        if elongation > 1.15:
            off = _fold90(det.angle_deg - axis)
            tr.body_offset = off if tr.body_offset is None else \
                _fold90(tr.body_offset + 0.2 * _fold90(off - tr.body_offset))

    def _blob_measure(self, tr: _Track, px, py, fg_mask, fg_scale):
        if tr.blob_area is None:
            return None
        comp = self._component_at(fg_mask, fg_scale, px, py, 2.5 * tr.side)
        if comp is None:
            return None
        cnt, area = comp
        if not (0.4 * tr.blob_area <= area <= 2.5 * tr.blob_area):
            return None
        m = cv2.moments(cnt)
        if m["m00"] <= 0:
            return None
        bx, by = m["m10"] / m["m00"], m["m01"] / m["m00"]
        heading = None
        axis, elongation = _body_axis(cnt)
        if tr.body_offset is not None and elongation > 1.15:
            c1 = _wrap(axis + tr.body_offset)
            c2 = _wrap(c1 + 180.0)
            heading = c1 if abs(_angle_diff(c1, tr.angle)) <= abs(_angle_diff(c2, tr.angle)) else c2
        return bx, by, heading


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _angle_diff(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def _fold90(a: float) -> float:
    """Fold an angle into (-90, 90] (axis directions are defined mod 180°)."""
    return (a + 90.0) % 180.0 - 90.0


def _rot(deg: float) -> np.ndarray:
    # Image y points down, so a math-CCW rotation by θ uses -θ in pixel space
    r = math.radians(-deg)
    return np.array([[math.cos(r), -math.sin(r)],
                     [math.sin(r),  math.cos(r)]], np.float32)


def _corner_angle(rel: np.ndarray) -> float:
    dx, dy = rel[1] - rel[0]
    return math.degrees(math.atan2(-dy, dx))


def _body_axis(cnt: np.ndarray) -> tuple[float, float]:
    """Long-axis angle (math convention, mod 180) and elongation of a contour."""
    (_, _), (w, h), ang = cv2.minAreaRect(cnt.reshape(-1, 1, 2).astype(np.float32))
    if w < h:
        w, h, ang = h, w, ang + 90.0
    return _fold90(-ang), w / max(h, 1e-6)
