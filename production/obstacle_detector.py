"""
Camera-based obstacle detection by comparison with an empty-floor background.

Workflow:
  1. With the floor clear, call capture_frame() on ~20 frames (the engine does
     this when the user clicks "Capture Floor"); finish_capture() builds a
     per-pixel median background plus a per-pixel noise estimate. Pixels inside
     cart footprints are excluded from the median, so carts may stay in view.
  2. Every frame, update() aligns the frame to the background (ORB features +
     RANSAC similarity — compensates camera drift / vibration), then flags
     pixels whose LAB colour differs from the floor by more than its noise.

Tuned on the overhead test videos (light wood floor, lamp hotspot):
  • lightness differences are down-weighted relative to chroma, and lightness
    is gain-compensated, so shadows and exposure changes cause fewer false hits
  • hysteresis: weak-difference regions survive only if they touch a strong
    difference — keeps faint objects whole without picking up wood texture
  • a "new edges" channel (edges absent from the background) closes the outline
    of floor-coloured objects such as cardboard, and holes are then filled
  • the final mask is smoothed over time to suppress single-frame flicker

The foreground mask (`foreground`, includes carts) feeds CartTracker's blob
fallback; obstacles() returns the components that are not carts.

ObstacleTracker adds object permanence on top: objects don't vanish in the
middle of the floor — they leave across the edge of the view or get carried
off by something that moves. So an obstacle the detector stops seeing stays
*remembered* unless it was at the edge, or a moving object (person, cart)
passed over it and the spot then stayed clear. Moving objects (people) are
tracked with a velocity and reported with the area they will sweep shortly.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import math

import cv2
import numpy as np

BACKGROUND_FILE = Path("data/floor_background.npz")

WORK_LONG_SIDE   = 960     # processing resolution (long side, px)
STRONG_SCORE     = 3.0     # noise-normalised colour distance for a sure hit
WEAK_SCORE       = 1.6     # ...for pixels that may extend a sure hit
MIN_AREA_FRAC    = 2.5e-4  # minimum obstacle area, fraction of the work image (≈2.5 cm)
ALIGN_EVERY      = 3       # re-estimate camera drift every N frames
MIN_ALIGN_INLIERS = 15
MAX_ALIGN_FAILS  = 30      # consecutive failed alignments (~3 s) → background unusable
TEMPORAL_ALPHA   = 0.5     # EMA weight of the newest mask
BORDER_FRAC      = 0.01    # ignore this fraction of the image at each edge
HEAL_FRAMES      = 30      # stable, uncovered frames before a never-seen pixel is learned
HEAL_MAX_STD_L   = 4.0     # max lightness std over those frames ("stable")
HULL_FILL_SOLIDITY = 0.5   # components filling less of their hull are outlines
HULL_FILL_MAX_FRAC = 0.02  # ...but never fill hulls larger than this (image fraction)
HULL_FILL_MAX_STRONG = 0.15  # ...nor ones with a solid strong-difference core

_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


# Obstacle states (ObstacleTracker)
OB_VISIBLE    = "visible"      # seen this frame, not moving
OB_MOVING     = "moving"       # a moving object (person…), seen recently
OB_REMEMBERED = "remembered"   # not seen, but can't have left — still there


@dataclass
class Obstacle:
    contour: np.ndarray   # (N, 2) float32 polygon in frame pixels
    area_px: float        # area in frame pixels²
    state: str = OB_VISIBLE
    track_id: int = -1
    swept: Optional[np.ndarray] = None   # moving objects: area covered shortly (frame px)

    @property
    def blocking_polygon(self) -> np.ndarray:
        return self.swept if self.swept is not None else self.contour

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        x0, y0 = self.contour.min(axis=0)
        x1, y1 = self.contour.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)

    @property
    def center(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2, (y0 + y1) / 2


class ObstacleDetector:
    def __init__(self, path: Path = BACKGROUND_FILE):
        self._path    = Path(path)                   # where save()/load() keep the floor
        self._orb     = cv2.ORB_create(3000)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self._scale: Optional[float] = None          # frame px → work px
        self._bg_lab: Optional[np.ndarray] = None    # (h, w, 3) float32
        self._bg_noise: Optional[np.ndarray] = None  # (h, w, 3) float32
        self._bg_edges: Optional[np.ndarray] = None  # dilated background edges
        self._bg_kp = None
        self._bg_desc = None
        self._bg_L_med: float = 0.0

        self._capture: List[Tuple[np.ndarray, np.ndarray]] = []
        self._unknown: Optional[np.ndarray] = None   # floor never seen during capture
        self._heal_n = self._heal_mean = self._heal_m2 = None
        self._roi_mask: Optional[np.ndarray] = None  # work-res, background coords
        self._roi_poly: Optional[np.ndarray] = None  # frame px

        self._M = np.eye(2, 3, dtype=np.float32)     # current work → background work
        self._align_fails = 0
        self._frame_no = 0
        self._ema: Optional[np.ndarray] = None
        self._fg_bg: Optional[np.ndarray] = None     # final mask, background coords
        self.foreground: Optional[np.ndarray] = None # final mask, current coords (work res)
        self.enabled = True

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def has_background(self) -> bool:
        return self._bg_lab is not None

    @property
    def alignment_ok(self) -> bool:
        """False when the camera view no longer matches the background
        (camera moved / bumped) — recapture the floor."""
        return self._align_fails < MAX_ALIGN_FAILS

    @property
    def capturing(self) -> bool:
        return bool(self._capture)

    @property
    def scale(self) -> float:
        """Work-image pixels per frame pixel (for the foreground mask)."""
        return self._scale or 1.0

    def set_roi(self, polygon_px: Optional[Sequence[Tuple[float, float]]]) -> None:
        """Only report obstacles inside this frame-pixel polygon (e.g. the
        calibrated floor area). None = whole image."""
        self._roi_poly = None if polygon_px is None else \
            np.asarray(polygon_px, np.float32).reshape(-1, 2)
        self._roi_mask = None   # rebuilt lazily at work resolution

    # ── Background capture ────────────────────────────────────────────────────

    def capture_frame(self, frame_bgr: np.ndarray,
                      exclude: Sequence[np.ndarray] = ()) -> int:
        """Add one frame to the pending background. `exclude` = polygons
        (frame px) to leave out, e.g. cart footprints. Returns frames so far."""
        work  = self._to_work(frame_bgr)
        valid = np.full(work.shape[:2], 255, np.uint8)
        for poly in exclude:
            cv2.fillPoly(valid, [np.round(poly * self._scale).astype(np.int32)], 0)
        self._capture.append((work, valid))
        return len(self._capture)

    def finish_capture(self) -> bool:
        if len(self._capture) < 3:
            self._capture.clear()
            return False
        frames = np.stack([f for f, _ in self._capture]).astype(np.float32)
        valid  = np.stack([v for _, v in self._capture]) > 0
        self._capture.clear()

        labs  = np.stack([cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2LAB)
                          for f in frames]).astype(np.float32)
        masked = np.where(valid[..., None], labs, np.nan)
        lab    = np.nanmedian(masked, axis=0)
        noise  = np.nanstd(masked, axis=0)
        hole   = np.isnan(lab[..., 0])
        lab    = np.nan_to_num(lab)
        noise  = np.nan_to_num(noise)
        if hole.any():   # never-seen pixels (cart parked all the time) → inpaint
            lab8 = cv2.inpaint(np.clip(lab, 0, 255).astype(np.uint8),
                               hole.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
            lab[hole] = lab8[hole]
        self._set_background(lab, noise)
        # The inpainted guess is poor (e.g. inside the lamp hotspot): learn these
        # pixels properly once they are seen uncovered (see _heal)
        self._unknown = hole if hole.any() else None
        return True

    def _set_background(self, lab: np.ndarray, noise: np.ndarray) -> None:
        # Floor the noise so a short capture can't make the model over-sensitive
        noise = np.maximum(cv2.GaussianBlur(noise, (5, 5), 0), [1.5, 0.6, 0.6])
        self._bg_lab   = lab.astype(np.float32)
        self._bg_noise = noise.astype(np.float32)
        self._heal_n = self._heal_mean = self._heal_m2 = None
        self._bg_L_med = float(np.median(lab[..., 0]))
        bgr  = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self._bg_edges = cv2.dilate(cv2.Canny(gray, 30, 90), np.ones((7, 7), np.uint8))
        self._bg_kp, self._bg_desc = self._orb.detectAndCompute(self._clahe.apply(gray), None)
        self._M     = np.eye(2, 3, dtype=np.float32)
        self._align_fails = 0
        self._ema   = None
        self._roi_mask = None

    def clear_background(self) -> None:
        self._unknown = None
        self._scale = None          # re-derived from the next frame's resolution
        self._roi_mask = None
        self._bg_lab = self._bg_noise = self._bg_edges = None
        self._bg_kp = self._bg_desc = None
        self._ema = self._fg_bg = self.foreground = None

    def save(self, path: Optional[Path] = None) -> None:
        if self._bg_lab is None:
            return
        path = Path(path or self._path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, lab=self._bg_lab.astype(np.float16),
                            noise=self._bg_noise.astype(np.float16),
                            scale=self._scale)

    def load(self, path: Optional[Path] = None) -> bool:
        path = Path(path or self._path)
        if not path.exists():
            return False
        try:
            data = np.load(path)
            self._scale = float(data["scale"])
            self._set_background(data["lab"].astype(np.float32),
                                 data["noise"].astype(np.float32))
            return True
        except Exception:
            return False

    # ── Per-frame detection ───────────────────────────────────────────────────

    def update(self, frame_bgr: np.ndarray,
               exclude: Sequence[np.ndarray] = ()) -> Optional[np.ndarray]:
        """Segment one frame. Returns the foreground mask (work resolution,
        current-frame coordinates) or None without a background.
        `exclude` — cart footprints (frame px), used only to heal the parts of
        the floor that were hidden under carts during the capture."""
        if not self.enabled or self._bg_lab is None:
            self.foreground = None
            return None
        work = self._to_work(frame_bgr)
        h, w = self._bg_lab.shape[:2]
        if work.shape[:2] != (h, w):           # camera resolution changed
            self.clear_background()
            return None

        if self._frame_no % ALIGN_EVERY == 0:
            if self._estimate_alignment(work):
                self._align_fails = 0
            else:
                self._align_fails += 1
        self._frame_no += 1
        if not self.alignment_ok:
            # Comparing against a misaligned background yields garbage
            self._ema = self._fg_bg = self.foreground = None
            return None

        warped = cv2.warpAffine(work, self._M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        valid  = cv2.warpAffine(np.full((h, w), 255, np.uint8), self._M, (w, h))
        mask   = self._segment(warped) & (valid == 255) & self._get_roi_mask()
        if self._unknown is not None:
            self._heal(warped, exclude)

        m = mask.astype(np.float32)
        self._ema = m if self._ema is None else \
            TEMPORAL_ALPHA * m + (1 - TEMPORAL_ALPHA) * self._ema
        fg = (self._ema > 0.5).astype(np.uint8) * 255
        fg = self._drop_small(fg)
        self._fg_bg = fg
        Mi = cv2.invertAffineTransform(self._M)
        self.foreground = cv2.warpAffine(fg, Mi, (work.shape[1], work.shape[0]),
                                         flags=cv2.INTER_NEAREST)
        return self.foreground

    def obstacles(self, exclude: Sequence[np.ndarray] = ()) -> List[Obstacle]:
        """Foreground components (frame px) after erasing the `exclude`
        polygons (cart footprints, frame px). Only the footprint is erased, so
        an object touching a cart is still reported."""
        if self.foreground is None:
            return []
        s  = self._scale
        fg = self.foreground
        if len(exclude):
            fg = fg.copy()
            cv2.fillPoly(fg, [np.round(np.asarray(p) * s).astype(np.int32)
                              for p in exclude], 0)
            fg = self._drop_small(cv2.morphologyEx(fg, cv2.MORPH_OPEN, _OPEN_KERNEL))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            poly = c.reshape(-1, 2).astype(np.float32) / s
            out.append(Obstacle(poly, cv2.contourArea(c) / (s * s)))
        return out

    # ── Internals ─────────────────────────────────────────────────────────────

    def _to_work(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._scale is None:
            self._scale = min(1.0, WORK_LONG_SIDE / max(frame_bgr.shape[:2]))
        if self._scale == 1.0:
            return frame_bgr
        return cv2.resize(frame_bgr, None, fx=self._scale, fy=self._scale,
                          interpolation=cv2.INTER_AREA)

    def _estimate_alignment(self, work: np.ndarray) -> bool:
        if self._bg_desc is None:
            return False
        gray = self._clahe.apply(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY))
        # Match on floor only: people and objects (last foreground) carry no
        # information about camera motion and a person near the camera can
        # cover much of the view
        feat_mask = None
        if self.foreground is not None and self.foreground.shape == gray.shape:
            feat_mask = cv2.bitwise_not(cv2.dilate(self.foreground, np.ones((15, 15), np.uint8)))
        kp, desc = self._orb.detectAndCompute(gray, feat_mask)
        if desc is None or len(kp) < MIN_ALIGN_INLIERS:
            return False
        pairs   = self._matcher.knnMatch(desc, self._bg_desc, k=2)
        matches = [p[0] for p in pairs
                   if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(matches) < MIN_ALIGN_INLIERS:
            return False
        src = np.float32([kp[m.queryIdx].pt for m in matches])
        dst = np.float32([self._bg_kp[m.trainIdx].pt for m in matches])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
        if M is None or inl is None or int(inl.sum()) < MIN_ALIGN_INLIERS:
            return False
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        if not 0.9 < scale < 1.1:       # implausible for a fixed camera
            return False
        self._M = M.astype(np.float32)
        return True

    def _segment(self, warped: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Global exposure compensation on lightness
        L_med = float(np.median(lab[..., 0]))
        if L_med > 0:
            lab[..., 0] *= self._bg_L_med / L_med
        d = lab - self._bg_lab
        n = self._bg_noise
        score = np.sqrt(0.5 * (d[..., 0] / (2 * n[..., 0] + 6)) ** 2 +
                        (d[..., 1] / (2 * n[..., 1] + 3)) ** 2 +
                        (d[..., 2] / (2 * n[..., 2] + 3)) ** 2)

        strong = (score > STRONG_SCORE).astype(np.uint8)
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        new_edges = cv2.Canny(gray, 30, 90) & ~self._bg_edges
        weak   = ((score > WEAK_SCORE) | (new_edges > 0)).astype(np.uint8)

        strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        weak   = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # Hysteresis: keep weak components that contain strong pixels, or whose
        # outline is long enough to be a real object edge (floor-coloured items)
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
        keep = np.zeros(n_lbl, bool)
        keep[np.unique(lbl[strong > 0])] = True
        min_area = MIN_AREA_FRAC * weak.size
        bw = stats[:, cv2.CC_STAT_WIDTH]
        bh = stats[:, cv2.CC_STAT_HEIGHT]
        keep |= (bw * bh > 4 * min_area) & (stats[:, cv2.CC_STAT_AREA] > 0.15 * bw * bh)
        keep[0] = False
        mask = keep[lbl].astype(np.uint8) * 255

        # Fill holes so outlined objects become solid, then open to drop thin
        # line-like residue (floor seams / wood grain under slight misalignment)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
        filled = self._fill_outlines(filled, contours, strong)
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, _OPEN_KERNEL)
        return filled > 0

    def _fill_outlines(self, mask: np.ndarray, contours, strong: np.ndarray) -> np.ndarray:
        """An outline that doesn't fully close (floor-coloured cardboard) fills
        to almost nothing — replace such hollow components by their convex hull.
        Skipped when the component has a solid strongly-different core (e.g. a
        cart that merged with a floor seam), which would make a phantom hull."""
        min_area = MIN_AREA_FRAC * mask.size
        for c in contours:
            hull      = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if not (4 * min_area <= hull_area <= HULL_FILL_MAX_FRAC * mask.size) or \
                    cv2.contourArea(c) >= HULL_FILL_SOLIDITY * hull_area:
                continue
            x, y, w, h = cv2.boundingRect(hull)
            comp = np.zeros((h, w), np.uint8)
            cv2.drawContours(comp, [c - [x, y]], -1, 1, cv2.FILLED)
            if (strong[y:y + h, x:x + w] & comp).sum() > HULL_FILL_MAX_STRONG * hull_area:
                continue
            cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def _heal(self, warped: np.ndarray, exclude: Sequence[np.ndarray]) -> None:
        """Learn floor pixels that were never seen during the capture (a cart
        was parked on them): once a pixel has been uncovered and stable for
        HEAL_FRAMES consecutive frames, its running mean becomes the floor.
        Caveat: an object set down exactly there before that would be learned
        as floor — the hole is a cart-sized patch and heals within ~1 s."""
        h, w = self._unknown.shape
        if self._heal_n is None:
            self._heal_n    = np.zeros((h, w), np.int32)
            self._heal_mean = np.zeros((h, w, 3), np.float32)
            self._heal_m2   = np.zeros((h, w, 3), np.float32)
        ys, xs = np.nonzero(self._unknown)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        cover = np.zeros((h, w), np.uint8)
        if len(exclude):
            polys = [cv2.transform((np.asarray(p, np.float32) * self._scale).reshape(-1, 1, 2),
                                   self._M).reshape(-1, 2) for p in exclude]
            cv2.fillPoly(cover, [np.round(p).astype(np.int32) for p in polys], 1)
            cover = cv2.dilate(cover, np.ones((9, 9), np.uint8))
        u   = self._unknown[y0:y1, x0:x1]
        obs = u & (cover[y0:y1, x0:x1] == 0)
        lab = cv2.cvtColor(warped[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
        n, mean, m2 = (a[y0:y1, x0:x1] for a in (self._heal_n, self._heal_mean, self._heal_m2))
        n[~obs] = 0                                         # must be consecutive
        n[obs] += 1
        k = np.maximum(n, 1)[..., None].astype(np.float32)
        first = (n == 1)[..., None]
        delta = lab - mean
        mean[:] = np.where(first, lab, mean + delta / k)
        m2[:]   = np.where(first, 0, m2 + delta * (lab - mean))
        done = obs & (n >= HEAL_FRAMES)
        if not done.any():
            return
        std  = np.sqrt(m2 / np.maximum(n - 1, 1)[..., None])
        good = done & (std[..., 0] < HEAL_MAX_STD_L)
        n[done & ~good] = 0                                 # something moved there: retry
        if good.any():
            bg, nz = self._bg_lab[y0:y1, x0:x1], self._bg_noise[y0:y1, x0:x1]
            bg[good] = mean[good]
            nz[good] = np.maximum(std[good], [1.5, 0.6, 0.6])
            u[good]  = False
            bgr  = cv2.cvtColor(np.clip(self._bg_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            self._bg_edges = cv2.dilate(cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 30, 90),
                                        np.ones((7, 7), np.uint8))
            if not self._unknown.any():
                self._unknown = None

    def _drop_small(self, mask: np.ndarray) -> np.ndarray:
        min_area = MIN_AREA_FRAC * mask.size
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False
        return keep[lbl].astype(np.uint8) * 255

    def _get_roi_mask(self) -> np.ndarray:
        if self._roi_mask is None:
            h, w = self._bg_lab.shape[:2]
            m = np.zeros((h, w), np.uint8)
            bx, by = int(BORDER_FRAC * w), int(BORDER_FRAC * h)
            m[by:h - by, bx:w - bx] = 1
            if self._roi_poly is not None:
                roi = np.zeros_like(m)
                cv2.fillPoly(roi, [np.round(self._roi_poly * self._scale).astype(np.int32)], 1)
                m &= roi
            self._roi_mask = m.astype(bool)
        return self._roi_mask


# ── Object permanence ─────────────────────────────────────────────────────────

CONFIRM_HITS         = 12     # sightings before an obstacle is "real" (can be remembered)…
CONFIRM_RATIO        = 0.75   # …seen in at least this share of the frames since it appeared
REMEMBER_MIN_FRAC    = 1e-3   # smaller things (specks) block while seen, but aren't remembered
UNCONFIRMED_DROP     = 10     # unseen frames before an unconfirmed track is dropped
EDGE_RELEASE_FRAMES  = 10     # unseen frames before an obstacle at the edge counts as gone
CLEAR_RELEASE_FRAMES = 15     # clear frames, after something passed over it, to release
MOBILE_MEMORY_FRAMES = 60     # how long a lost moving object is still assumed there
MOTION_WINDOW        = 10     # frames over which net displacement is measured
MOTION_MIN_FRAC      = 0.3    # net displacement (× object size) that marks a mover…
MOTION_MIN_DIAG      = 0.012  # …and at least this fraction of the image diagonal
MOBILE_MIN_FRAC      = 2e-3   # movers are big (people); specks never count as moving
PREDICT_FRAMES       = 30     # moving objects block the area they sweep in this time
MATCH_MIN_IOU        = 0.1
REAPPEAR_MIN_IOU     = 0.4    # after being passed over, a sighting must match this well…
REAPPEAR_FRAMES      = 5      # …this many frames in a row to count as the same object
COVER_MIN_FRAC       = 0.3    # a static neighbour hides an obstacle only above this overlap
MERGE_GROW           = 1.6    # a static obstacle's blob growing this much at once = touched
# Shape anchoring: a confirmed static obstacle keeps the outline it was
# confirmed with while new sightings still match it — edges that breathe with
# lighting / segmentation noise, or swell with a passing cart's shadow, must
# not creep onto planned paths. A real change is adopted once it persists.
STABLE_IOU           = 0.8    # sightings matching the outline this well = same shape
ADOPT_FRAMES         = 5      # consecutive differing sightings before adopting the new shape
MOVER_NEAR_PX        = 40     # never adopt while a cart / person is this close (frame px):
                              # its shadow or body merges into the blob
REACH_FRAC           = 0.5    # a person within this × their size can take an object
EDGE_MARGIN_FRAC     = 0.03   # "at the edge": within this fraction of the image size
MASK_SCALE           = 0.25   # rasterisation scale for overlap tests


@dataclass
class _ObTrack:
    id: int
    contour: np.ndarray
    area: float
    hits: int = 1
    age: int = 1
    unseen: int = 0
    vx: float = 0.0
    vy: float = 0.0
    mobile: bool = False          # has been seen moving — a person or similar
    confirmed: bool = False       # solid, persistent object (can be remembered)
    disturbed: bool = False       # something moving passed over it while unseen
    clear: int = 0                # frames verified clear since disturbed
    reseen: int = 0               # consistent sightings since disturbed
    changed: int = 0              # consecutive sightings differing from the anchored outline
    history: list = None          # recent centroids

    def __post_init__(self):
        self.history = [_centroid(self.contour)]

    @property
    def center(self) -> Tuple[float, float]:
        return _centroid(self.contour)


def _centroid(poly: np.ndarray) -> Tuple[float, float]:
    m = cv2.moments(poly.reshape(-1, 1, 2).astype(np.float32))
    if m["m00"] > 1e-6:
        return m["m10"] / m["m00"], m["m01"] / m["m00"]
    c = poly.mean(axis=0)
    return float(c[0]), float(c[1])


class ObstacleTracker:
    """Gives obstacles identity over time and enforces object permanence
    (see module docstring). Feed it ObstacleDetector.obstacles() each frame."""

    def __init__(self):
        self._tracks: Dict[int, _ObTrack] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()

    def forget_region(self, polygon: np.ndarray) -> int:
        """Operator removed something here: drop tracks whose centre lies in
        `polygon` (frame px)."""
        poly = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
        gone = [i for i, t in self._tracks.items()
                if cv2.pointPolygonTest(poly, tuple(map(float, t.center)), False) >= 0]
        for i in gone:
            del self._tracks[i]
        return len(gone)

    def forget_remembered(self) -> int:
        """Operator override: drop every obstacle that isn't currently seen."""
        gone = [i for i, t in self._tracks.items() if t.unseen > 0]
        for i in gone:
            del self._tracks[i]
        return len(gone)

    def update(self, detections: List[Obstacle], frame_shape: Tuple[int, int],
               movers: Sequence[np.ndarray] = (),
               roi: Optional[np.ndarray] = None) -> List[Obstacle]:
        """detections — this frame's obstacles (frame px)
        movers     — polygons of other things known to move (cart footprints)
        roi        — floor polygon (frame px); its border is the 'edge' of the view
        Returns visible, moving and remembered obstacles."""
        h, w   = frame_shape[:2]
        shape  = (max(1, int(h * MASK_SCALE)), max(1, int(w * MASK_SCALE)))
        diag   = math.hypot(w, h)
        min_remember = REMEMBER_MIN_FRAC * h * w
        det_m  = [self._mask(d.contour, shape) for d in detections]
        det_a  = [int(m.sum()) for m in det_m]
        mover_m = self._mask_many(movers, shape)

        # Predicted masks: moving tracks are shifted by their velocity
        tr_ids = list(self._tracks)
        tr_m   = {i: self._mask(self._tracks[i].contour +
                                np.float32([self._tracks[i].vx, self._tracks[i].vy]), shape)
                  for i in tr_ids}
        tr_a   = {i: int(tr_m[i].sum()) for i in tr_ids}

        # Global matching by IoU (best pairs first). A detection overlapping
        # other tracks covers them (occluded), it doesn't make them vanish.
        pairs = []
        overlaps: Dict[int, List[Tuple[int, int]]] = {i: [] for i in tr_ids}
        for j, dm in enumerate(det_m):
            for i in tr_ids:
                inter = int((dm & tr_m[i]).sum())
                if inter:
                    overlaps[i].append((j, inter))
                    pairs.append((inter / max(det_a[j] + tr_a[i] - inter, 1), i, j))
        pairs.sort(reverse=True)
        primary: Dict[int, int] = {}            # track id → detection index
        owner: Dict[int, int] = {}              # detection index → track id
        match_iou: Dict[int, float] = {}
        for iou, i, j in pairs:
            if iou < MATCH_MIN_IOU:
                break
            if i not in primary and j not in owner:
                primary[i], owner[j] = j, i
                match_iou[i] = iou

        out: List[Obstacle] = []
        for i in tr_ids:
            t = self._tracks[i]
            t.age += 1
            if i in primary:
                d = detections[primary[i]]
                if t.confirmed and not t.mobile and d.area_px > MERGE_GROW * t.area:
                    # Its blob suddenly grew: something (a hand, an arm) is
                    # touching it — it may be picked up, so from now on only a
                    # clear sighting of the object itself proves it's still there
                    t.disturbed = True
                if t.disturbed and not t.mobile:
                    # After something passed over it, only a sighting that looks
                    # like the same object (size + place), several frames in a
                    # row, proves it is still there — not a leftover piece of
                    # the person walking away
                    same = match_iou[i] >= REAPPEAR_MIN_IOU and \
                        0.6 < d.area_px / max(t.area, 1.0) < 1.6
                    t.unseen += 1
                    if not same:
                        t.reseen = 0
                        t.clear = 0
                        continue
                    t.reseen += 1
                    if t.reseen < REAPPEAR_FRAMES:
                        continue
                    t.disturbed = False
                if self._keeps_shape(t, match_iou[i], tr_m[i], mover_m):
                    pass                         # same object, noisy edge: outline anchored
                else:
                    t.contour, t.area = d.contour, d.area_px
                t.history = (t.history + [_centroid(d.contour)])[-MOTION_WINDOW:]
                self._update_motion(t, diag)
                t.hits += 1
                t.unseen = 0
                t.disturbed = False
                t.clear = 0
                t.reseen = 0
                if not t.mobile and t.hits >= CONFIRM_HITS and \
                        t.hits >= CONFIRM_RATIO * t.age and t.area >= min_remember:
                    t.confirmed = True
                continue

            # Not seen as itself this frame
            t.unseen += 1
            if t.mobile:
                # A person can't hide behind an object: coast briefly, then let
                # go (they walked off or merged into something else)
                t.contour = t.contour + np.float32([t.vx, t.vy])
                t.vx *= 0.9
                t.vy *= 0.9
                if t.unseen > MOBILE_MEMORY_FRAMES or self._at_edge(t.contour, w, h, roi):
                    del self._tracks[i]
                continue
            # Occluded by / merged into something? Anything moving counts; a
            # static neighbour only if it really covers it (a picked-up object's
            # old outline may graze the object next to it)
            def is_mobile(j):
                o = self._tracks.get(owner.get(j, -1))
                return o is not None and o.mobile

            def is_newcomer(j):
                # Not an established static object: something that just
                # appeared over it — typically a hand reaching in to take it
                o = self._tracks.get(owner.get(j, -1))
                return o is None or not o.confirmed
            covering = [j for j, inter in overlaps[i]
                        if is_mobile(j) or inter >= COVER_MIN_FRAC * max(tr_a[i], 1)]
            if covering:
                if any(is_mobile(j) or is_newcomer(j) for j in covering):
                    t.disturbed = True
                t.clear = 0
                continue
            if (tr_m[i] & mover_m).any():
                t.disturbed = True               # a cart drove over / pushed it
                t.clear = 0
                continue
            if not t.disturbed and self._within_reach(t, w, h):
                t.disturbed = True               # a person right beside it can take it
            if t.disturbed:
                t.clear += 1
            if t.confirmed and t.area < min_remember:
                t.confirmed = False              # shrank to a speck: not worth remembering
            if not t.confirmed:
                if t.unseen > UNCONFIRMED_DROP:
                    del self._tracks[i]
            elif self._at_edge(t.contour, w, h, roi):
                if t.unseen > EDGE_RELEASE_FRAMES:
                    del self._tracks[i]          # partly out of view: it left
            elif t.disturbed and t.clear >= CLEAR_RELEASE_FRAMES:
                del self._tracks[i]              # carried off
            # else: remembered — it's still there, the detector just misses it

        for j, d in enumerate(detections):
            if j not in owner:
                self._tracks[self._next_id] = _ObTrack(self._next_id, d.contour, d.area_px)
                self._next_id += 1

        for t in self._tracks.values():
            if t.mobile:
                state = OB_MOVING
            elif t.unseen == 0:
                state = OB_VISIBLE
            elif t.confirmed:
                state = OB_REMEMBERED
            else:
                continue                         # unconfirmed and not seen: noise
            swept = None
            if t.mobile:
                ahead = t.contour + np.float32([t.vx, t.vy]) * PREDICT_FRAMES
                swept = cv2.convexHull(np.vstack([t.contour, ahead]).astype(np.float32)).reshape(-1, 2)
            out.append(Obstacle(t.contour, t.area, state, t.id, swept))
        return out

    @staticmethod
    def _keeps_shape(t: _ObTrack, iou: float, track_mask: np.ndarray,
                     mover_mask: np.ndarray) -> bool:
        """Should a confirmed static obstacle keep its anchored outline for
        this sighting? Yes while the sighting matches it (edge noise), and
        while a cart or person is close (their shadow / body merges into the
        blob); a different shape is adopted only after ADOPT_FRAMES in a row."""
        if not t.confirmed or t.mobile:
            t.changed = 0
            return False
        if iou >= STABLE_IOU:
            t.changed = 0
            return True
        if mover_mask.any():
            k = 2 * max(1, int(MOVER_NEAR_PX * MASK_SCALE)) + 1
            near = cv2.dilate(track_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            if (near & mover_mask).any():
                t.changed = 0
                return True
        t.changed += 1
        if t.changed < ADOPT_FRAMES:
            return True
        t.changed = 0
        return False

    def _within_reach(self, t: _ObTrack, w: int, h: int) -> bool:
        """Is a moving object (a person) close enough to have picked this up?
        Reach scales with the person's size in the image (arm ≈ shoulder width)."""
        x0, y0 = t.contour.min(axis=0)
        x1, y1 = t.contour.max(axis=0)
        for o in self._tracks.values():
            if not o.mobile or o.unseen > 2:
                continue
            reach = REACH_FRAC * math.sqrt(max(o.area, 1.0))
            ox0, oy0 = o.contour.min(axis=0)
            ox1, oy1 = o.contour.max(axis=0)
            gap_x = max(0.0, max(ox0 - x1, x0 - ox1))
            gap_y = max(0.0, max(oy0 - y1, y0 - oy1))
            if math.hypot(gap_x, gap_y) < reach:
                return True
        return False

    @staticmethod
    def _update_motion(t: _ObTrack, diag: float) -> None:
        """Velocity from the centroid history; 'mobile' only after a sustained
        net displacement (shape jitter and blob merges don't qualify)."""
        if len(t.history) < 3:
            return
        (x0, y0), (x1, y1) = t.history[0], t.history[-1]
        n = len(t.history) - 1
        t.vx, t.vy = (x1 - x0) / n, (y1 - y0) / n
        if len(t.history) >= MOTION_WINDOW and not t.mobile and \
                t.area >= MOBILE_MIN_FRAC * diag * diag / 2:
            net = math.hypot(x1 - x0, y1 - y0)
            if net > max(MOTION_MIN_FRAC * math.sqrt(max(t.area, 1.0)), MOTION_MIN_DIAG * diag):
                t.mobile = True
                t.confirmed = False
        if not t.mobile:
            t.vx = t.vy = 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mask(poly: np.ndarray, shape) -> np.ndarray:
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.round(np.asarray(poly) * MASK_SCALE).astype(np.int32)], 1)
        return m

    @staticmethod
    def _mask_many(polys, shape) -> np.ndarray:
        m = np.zeros(shape, np.uint8)
        if len(polys):
            cv2.fillPoly(m, [np.round(np.asarray(p) * MASK_SCALE).astype(np.int32)
                             for p in polys], 1)
        return m

    @staticmethod
    def _at_edge(poly: np.ndarray, w: int, h: int, roi: Optional[np.ndarray]) -> bool:
        margin = EDGE_MARGIN_FRAC * max(w, h)
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        if x0 < margin or y0 < margin or x1 > w - margin or y1 > h - margin:
            return True
        if roi is not None:
            r = roi.reshape(-1, 1, 2).astype(np.float32)
            return any(cv2.pointPolygonTest(r, (float(x), float(y)), True) < margin
                       for x, y in poly[::max(1, len(poly) // 16)])
        return False
