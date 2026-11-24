"""
Photo-realistic rendering from the recorded test videos.

Shared by tools/synth_videos.py (offline test videos) and
simulation/digital_twin.py (closed-loop virtual production mode).

Assets are cut from the real recordings once and cached:
  • floor     — every frame of "Background 1" (JPEG in memory), played back
                and forth so sensor noise, the lamp hotspot and drift are real
  • cart      — sprite from "Just cart 1" with its marker corners; extra carts
                get their marker re-rendered with another ArUco ID (remark)
  • obstacles — toolbox, phones, wallets, card, cardboard from "Setup 1" and
                "Background and objects 1"

Rendering (place + composite) models what the recordings show: weak re-lighting
of matte objects, glossy marker washout inside the hotspot, motion blur.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from production.cart_detector import ARUCO_DICT_ID, CartDetector
from production.obstacle_detector import ObstacleDetector

ROOT       = Path(__file__).resolve().parent.parent
VIDEOS     = ROOT / "videos"
FLOOR_CLIP = VIDEOS / "Background 1.mp4"
CART_SRC   = (VIDEOS / "Just cart 1.mp4", 980)        # sharp, outside the glare
OBST_SRCS  = [(VIDEOS / "Setup 1.mp4", 300),
              (VIDEOS / "Background and objects 1.mp4", 100)]
ASSET_CACHE  = VIDEOS / "synthetic" / "assets"
FLOOR_BOTTOM = 0.84          # below this fraction of the height is off-floor clutter
# On the real cart the marker is mounted rotated: in the recordings the cart
# drives toward (marker angle − 90°).
MARKER_MINUS_TRAVEL = 90.0


# ── Assets ────────────────────────────────────────────────────────────────────

@dataclass
class Sprite:
    rgba: np.ndarray                     # (h, w, 4) uint8
    anchor: np.ndarray                   # (2,) rotation centre in sprite px
    illum: float                         # floor illumination where it was cut
    marker: Optional[np.ndarray] = None  # (4, 2) marker corners in sprite px
    marker_angle: float = 0.0            # marker heading when cut


def _read(path: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {idx} of {path}")
    return frame


def _floor_detector() -> ObstacleDetector:
    od  = ObstacleDetector()
    cap = cv2.VideoCapture(str(FLOOR_CLIP))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in np.linspace(0, n - 1, 20).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            od.capture_frame(frame)
    od.finish_capture()
    return od


def _cutout(frame: np.ndarray, fg: np.ndarray, hint: np.ndarray, margin: int = 25):
    """RGBA crop of the foreground object around `hint` (polygon), refined
    with GrabCut. Returns (rgba, (x0, y0))."""
    h, w = frame.shape[:2]
    x0, y0 = np.maximum(hint.min(0).astype(int) - margin, 0)
    x1, y1 = np.minimum(hint.max(0).astype(int) + margin, [w, h])
    crop, m = frame[y0:y1, x0:x1].copy(), fg[y0:y1, x0:x1]
    gc = np.full(m.shape, cv2.GC_PR_BGD, np.uint8)
    gc[cv2.dilate(m, np.ones((15, 15), np.uint8)) == 0] = cv2.GC_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(m, np.ones((9, 9), np.uint8)) > 0] = cv2.GC_FGD
    cv2.grabCut(crop, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4,
                cv2.GC_INIT_WITH_MASK)
    a = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(a)
    if n > 1:
        a = np.where(lbl == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
    a = cv2.GaussianBlur(a, (5, 5), 0)
    return np.dstack([crop, a]), (int(x0), int(y0))


def _foreground(od: ObstacleDetector, frame: np.ndarray) -> np.ndarray:
    for _ in range(3):                       # let the temporal filter settle
        fg = od.update(frame)
    return cv2.resize(fg, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)


def build_assets(cache: Path) -> Tuple[Sprite, List[Sprite], np.ndarray, List[np.ndarray]]:
    """Cut the sprites (cached as PNG/JSON in `cache`). Returns the cart
    sprite, obstacle sprites, the floor illumination map and the floor frames."""
    od      = _floor_detector()
    floor_h = od._bg_lab.shape[0] / od.scale
    illum   = _illumination_map(od)
    floors  = _floor_frames()

    cache.mkdir(parents=True, exist_ok=True)
    meta_f  = cache / "assets.json"
    if meta_f.exists():
        meta = json.loads(meta_f.read_text())
        def load(m):
            return Sprite(cv2.imread(str(cache / m["file"]), cv2.IMREAD_UNCHANGED),
                          np.array(m["anchor"], np.float32), m["illum"],
                          None if m.get("marker") is None else np.array(m["marker"], np.float32),
                          m.get("marker_angle", 0.0))
        return load(meta["cart"]), [load(m) for m in meta["obstacles"]], illum, floors

    det = CartDetector()

    # Cart: the foreground component under the marker
    frame = _read(*CART_SRC)
    mk    = [d for d in det.detect(frame) if d.marker_id == 1][0]
    fg    = _foreground(od, frame)
    rgba, (x0, y0) = _cutout(frame, fg, mk.footprint(2.6))
    cart = Sprite(rgba, np.array([mk.cx - x0, mk.cy - y0], np.float32),
                  _illum_at(illum, mk.cx, mk.cy),
                  mk.corners - np.array([x0, y0], np.float32), mk.angle_deg)

    # Obstacles: every on-floor component that is not a cart
    obstacles: List[Sprite] = []
    for path, idx in OBST_SRCS:
        frame  = _read(path, idx)
        fg     = _foreground(od, frame)
        carts  = [d.footprint(2.6) for d in det.detect(frame)]
        for ob in od.obstacles(carts):
            cx, cy = ob.center
            x_0, y_0, x_1, y_1 = ob.bbox
            if y_1 > FLOOR_BOTTOM * floor_h or min(x_1 - x_0, y_1 - y_0) < 25:
                continue
            rgba, (x0, y0) = _cutout(frame, fg, ob.contour)
            obstacles.append(Sprite(rgba, np.array([cx - x0, cy - y0], np.float32),
                                    _illum_at(illum, cx, cy)))

    def dump(name, s: Sprite):
        cv2.imwrite(str(cache / name), s.rgba)
        return {"file": name, "anchor": s.anchor.tolist(), "illum": s.illum,
                "marker": None if s.marker is None else s.marker.tolist(),
                "marker_angle": s.marker_angle}
    meta_f.write_text(json.dumps({
        "cart": dump("cart.png", cart),
        "obstacles": [dump(f"obstacle_{i}.png", s) for i, s in enumerate(obstacles)],
    }, indent=1))
    return cart, obstacles, illum, floors


def _illumination_map(od: ObstacleDetector) -> np.ndarray:
    """Smooth floor brightness at full resolution (L channel, 0–255)."""
    L = od._bg_lab[..., 0]
    h, w = int(round(L.shape[0] / od.scale)), int(round(L.shape[1] / od.scale))
    return cv2.resize(cv2.GaussianBlur(L, (0, 0), 12), (w, h)).astype(np.float32)


def _illum_at(illum: np.ndarray, x: float, y: float) -> float:
    h, w = illum.shape
    return float(illum[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))])


def _floor_frames() -> List[np.ndarray]:
    """All floor frames, JPEG-encoded in memory (full-res raw would be ~2 GB)."""
    cap, frames = cv2.VideoCapture(str(FLOOR_CLIP)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1])
    return frames


def remark(cart: Sprite, marker_id: int, quarter_turns: int = 0) -> Sprite:
    """Copy of the cart sprite with its marker re-rendered as `marker_id`,
    matched to the sprite's own black/white levels. `quarter_turns` mounts the
    marker rotated on the cart by that many 90° turns CCW (the body — and so
    the direction the cart drives — stays the same)."""
    quarter_turns %= 4
    if marker_id == 1 and quarter_turns == 0:
        return cart
    rgba = cart.rgba.copy()
    quad = cart.marker.astype(np.float32)
    mask = np.zeros(rgba.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    vals = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2GRAY)[mask > 0]
    black, white = np.percentile(vals, 5), np.percentile(vals, 95)
    size = 120
    img  = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID),
                                         marker_id, size)
    img  = np.rot90(img, quarter_turns).copy()      # CCW as displayed
    src  = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    H    = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(img, H, rgba.shape[1::-1], flags=cv2.INTER_AREA)
    level  = black + (white - black) * (warped.astype(np.float32) / 255.0)
    tint   = rgba[..., :3][mask > 0].mean(axis=0) / max(vals.mean(), 1)   # keep colour cast
    new    = np.clip(level[..., None] * tint, 0, 255).astype(np.uint8)
    inner  = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
    rgba[..., :3][inner] = new[inner]
    # Corners re-ordered so corner 0 is still the marker's top-left
    marker = np.roll(cart.marker, -quarter_turns, axis=0)
    return Sprite(rgba, cart.anchor, cart.illum, marker,
                  (cart.marker_angle + 90.0 * quarter_turns + 180) % 360 - 180)


# ── Rendering ─────────────────────────────────────────────────────────────────

def place(sprite: Sprite, x: float, y: float, angle_deg: float, shape=None,
          scale: float = 1.0) -> Tuple:
    """Rotate the sprite by angle_deg (CCW) and scale it about its anchor, and
    put the anchor at (x, y). Returns (patch_rgba, x0, y0, M) with
    M = sprite px → frame px."""
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    M = cv2.getRotationMatrix2D(tuple(map(float, sprite.anchor)), angle_deg, scale)
    M[:, 2] += np.array([x, y]) - sprite.anchor
    h, w = sprite.rgba.shape[:2]
    box  = cv2.transform(np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]), M).reshape(-1, 2)
    x0, y0 = np.floor(box.min(0)).astype(int) - 2
    x1, y1 = np.ceil(box.max(0)).astype(int) + 2
    Mp = M.copy()
    Mp[:, 2] -= [x0, y0]
    patch = cv2.warpAffine(sprite.rgba, Mp, (x1 - x0, y1 - y0), flags=interp,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return patch, int(x0), int(y0), M


def composite(frame: np.ndarray, patch: np.ndarray, x0: int, y0: int,
              illum: np.ndarray, src_illum: float, gloss: Optional[np.ndarray] = None,
              gloss_strength: float = 0.0, blur: Tuple[float, float] = (0.0, 0.0)) -> None:
    """Alpha-blend a placed sprite patch into `frame` in place, re-lit for its
    new position. `gloss` (patch-sized 0–1) marks the glossy marker paper."""
    H, W = frame.shape[:2]
    ph, pw = patch.shape[:2]
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + pw, W), min(y0 + ph, H)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    p = patch[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].astype(np.float32)
    rgb, a = p[..., :3], p[..., 3:] / 255.0

    # The hotspot is specular reflection on the glossy floor, not stronger
    # light: matte objects barely change, so re-light only weakly...
    light = illum[fy0:fy1, fx0:fx1][..., None]
    rgb  *= np.clip((light / max(src_illum, 1.0)) ** 0.3, 0.8, 1.2)
    # ...while glossy marker paper reflects it: dark cells fade to mid grey
    if gloss is not None and gloss_strength > 0:
        g = gloss[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0][..., None]
        hot = np.clip((light - 150.0) / 55.0, 0, 1) ** 1.5      # 0 off-spot → 1 at peak
        w = np.clip(gloss_strength * hot * g, 0, 0.88)
        rgb = np.maximum(rgb, rgb * (1 - w) + 175.0 * w)

    length, angle = blur
    if length >= 1.5:
        k = _motion_kernel(length, angle)
        rgb = cv2.filter2D(rgb * a, -1, k)
        a   = cv2.filter2D(a[..., 0], -1, k)[..., None]
        rgb = np.where(a > 1e-3, rgb / np.maximum(a, 1e-3), rgb)

    dst = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = np.clip(rgb * a + dst * (1 - a), 0, 255).astype(np.uint8)


def _motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    n = int(math.ceil(length)) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    dx, dy = math.cos(math.radians(angle_deg)), -math.sin(math.radians(angle_deg))
    for t in np.linspace(-length / 2, length / 2, n * 3):
        k[int(round(c + dy * t)), int(round(c + dx * t))] = 1
    return k / k.sum()


def _alpha_polygon(patch: np.ndarray, x0: int, y0: int) -> np.ndarray:
    cs, _ = cv2.findContours((patch[..., 3] > 127).astype(np.uint8),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).astype(np.float32)
    return c + [x0, y0]



# ── Camera ─────────────────────────────────────────────────────────────────────

class CameraDrift:
    """Smooth random similarity transform (Ornstein–Uhlenbeck on tx, ty, θ)."""

    def __init__(self, rng, amp_px: float, amp_deg: float):
        self.rng, self.amp_px, self.amp_deg = rng, amp_px, amp_deg
        self.s = np.zeros(3)

    def step(self, center) -> np.ndarray:
        if self.amp_px <= 0 and self.amp_deg <= 0:
            return np.float32([[1, 0, 0], [0, 1, 0]])
        sig = np.array([self.amp_px, self.amp_px, self.amp_deg]) * 0.08
        self.s += -0.02 * self.s + self.rng.normal(0, 1, 3) * sig
        M = cv2.getRotationMatrix2D(center, float(self.s[2]), 1.0)
        M[:, 2] += self.s[:2]
        return M.astype(np.float32)
