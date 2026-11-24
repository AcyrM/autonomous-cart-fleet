"""
Digital twin of the production setup, built from the recorded test footage.

Closes the loop around the *real* production stack: ProdEngine (tracking,
obstacle detection, Theta* planning, steering) receives camera frames from
VirtualCamera and sends DRIVE/STOP commands to VirtualBTController, exactly
as it would to a webcam and the ESP32 carts. The commands drive simulated
differential-drive carts whose new poses appear in the next rendered frame.

  VirtualWorld        — real floor frames + cart/obstacle sprites
                        (simulation/photoreal.py), cart physics, collisions,
                        ground truth
  VirtualCart         — differential drive: command latency, motor dead
                        band, first-order wheel response, per-wheel gain error
  VirtualCamera       — CameraSource look-alike (frame_ready QImage signal)
  VirtualBTController — BTController look-alike (same signals and methods)

Everything is in *render* pixels unless noted; the world is modelled at the
recordings' full resolution and scaled down for rendering (TwinConfig.scale).

The control code never reads VirtualWorld state: ProdEngine only receives the
rendered images and only emits BT commands, so cart positions come from the
image exactly as in production. World state is used only by the test harness
(placing obstacles, choosing mission destinations, scoring against ground
truth), and calibration is given the hand-click error an operator would make.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QImage

from production.bt_controller import BTDevice
from simulation.photoreal import (
    ASSET_CACHE, FLOOR_BOTTOM, MARKER_MINUS_TRAVEL, CameraDrift, Sprite,
    _alpha_polygon, build_assets, composite, place, remark,
)

CART_COLLIDE_SIDES = 1.0      # cart body radius for collisions, in marker sides
TRACK_SIDES        = 1.9      # wheel track width, in marker sides
OBSTACLE_GAP_PX    = 60       # min gap between placed obstacles (full res)


@dataclass
class TwinConfig:
    n_carts: int = 2
    n_obstacles: int = 5
    scale: float = 0.5            # render resolution relative to the recordings
    fps: int = 30
    # Marker heading minus travel direction (multiples of 90°). The recorded
    # cart has its marker mounted at +90° — the default. Other values re-mount
    # the marker image on the cart; the body always drives along its wheels.
    marker_offset: float = MARKER_MINUS_TRAVEL
    px_per_unit: float = 1.1      # full-res px/s per DRIVE unit (180 ≈ recorded speed)
    deadband: int = 40            # |DRIVE| below this doesn't move the wheel
    motor_tau: float = 0.12       # s, wheel speed response time constant
    latency: float = 0.06         # s, BT + firmware command delay
    wheel_mismatch: float = 0.05  # ± per-wheel gain error (makes carts drift)
    drift_px: float = 0.0         # camera drift amplitude (full-res px)
    calib_noise_px: float = 2.0   # σ of the operator's calibration clicks (render px)
    walkers_per_min: float = 0.0  # people walking through (spawned once enabled)
    # "landscape": the recorded (portrait) camera turned 90° → 1920×1080, 16:9
    # like a computer screen; "portrait": as recorded
    orientation: str = "landscape"
    seed: int = 0


# ── Carts ─────────────────────────────────────────────────────────────────────

@dataclass
class VirtualCart:
    marker_id: int
    sprite: Sprite
    gloss: np.ndarray
    x: float                      # full-res px
    y: float
    heading: float                # travel direction, deg (math convention)
    gain_l: float = 1.0
    gain_r: float = 1.0
    # Per-cart physics (set from TwinConfig defaults by VirtualWorld.add_cart)
    marker_offset: float = MARKER_MINUS_TRAVEL   # marker heading − travel direction
    px_per_unit: float = 1.1      # full-res px/s per DRIVE unit
    motor_tau: float = 0.12       # s, wheel speed response
    deadband: int = 40            # |DRIVE| below this doesn't move the wheel
    vl: float = 0.0               # wheel speeds, full-res px/s
    vr: float = 0.0
    cmd: Tuple[int, int] = (0, 0)
    queue: List[Tuple[float, Tuple[int, int]]] = field(default_factory=list)
    collisions: int = 0
    hidden: bool = False          # test hook: not drawn (total detection failure), still physical
    _touching: bool = False
    _free_since: float = -1.0

    def __post_init__(self):
        self.gloss_sprite = Sprite(np.dstack([self.gloss] * 4), self.sprite.anchor, 0)

    def command(self, t: float, left: int, right: int, latency: float) -> None:
        self.queue.append((t + latency, (int(left), int(right))))

    @property
    def speed(self) -> float:
        return (self.vl + self.vr) / 2


# ── World ─────────────────────────────────────────────────────────────────────

@dataclass(eq=False)          # identity, not field-by-field (holds image arrays)
class WorldObstacle:
    patch: np.ndarray             # rendered once, at render resolution
    x0: int
    y0: int
    polygon: np.ndarray           # full-res px
    src_illum: float
    sprite_idx: int = 0           # which obstacle sprite, and its pose (templates)
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0


class VirtualWorld:
    def __init__(self, cfg: TwinConfig = TwinConfig()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._cart_sprite, self._obstacle_sprites, self._illum, self._floors = \
            build_assets(ASSET_CACHE)
        landscape = cfg.orientation == "landscape"
        if landscape:
            # Camera turned 90° CCW: the recording's bottom edge (off-floor
            # clutter) ends up on the right
            self._illum = cv2.rotate(self._illum, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.H, self.W = self._illum.shape                 # full-res world size
        # Floor area (full-res px): the view minus the off-floor clutter strip
        if landscape:
            self.floor_rect = (0.0, 0.0, FLOOR_BOTTOM * self.W, float(self.H))
        else:
            self.floor_rect = (0.0, 0.0, float(self.W), FLOOR_BOTTOM * self.H)
        self.render_size = (int(round(self.W * cfg.scale)), int(round(self.H * cfg.scale)))
        # Everything is drawn directly at render resolution: floor frames
        # (turned, scaled) and the illumination map are prepared once here
        if landscape or cfg.scale != 1.0:
            def prep(buf):
                f = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if landscape:
                    f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if cfg.scale != 1.0:
                    f = cv2.resize(f, self.render_size, interpolation=cv2.INTER_AREA)
                return cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
            self._floors = [prep(f) for f in self._floors]
        self._illum_r = cv2.resize(self._illum, self.render_size) \
            if cfg.scale != 1.0 else self._illum
        side = float(np.mean(np.linalg.norm(
            self._cart_sprite.marker - np.roll(self._cart_sprite.marker, -1, axis=0), axis=1)))
        self.marker_side = side                             # full-res px
        self.cart_radius = CART_COLLIDE_SIDES * side
        self.track       = TRACK_SIDES * side

        self.t = 0.0
        self.frame_no = 0
        self.carts: Dict[int, VirtualCart] = {}
        self.obstacles: List[WorldObstacle] = []
        self.collision_log: List[Tuple[float, int, str]] = []   # (t, marker id, what)
        self.walkers: List["Walker"] = []
        self.walkers_enabled = False       # set once the floor has been captured
        self.walker_log: List[Tuple[float, str]] = []           # (t, event)
        self._next_walker = 0.0
        self._occ = np.zeros((self.H, self.W), np.uint8)       # obstacle footprint
        self._inflated = np.zeros_like(self._occ)              # + cart radius
        self._drift = CameraDrift(self.rng, cfg.drift_px, 0.0)
        self._Mc = np.float32([[1, 0, 0], [0, 1, 0]])
        for i in range(cfg.n_carts):
            self.add_cart(i + 1)

    # ── Population ────────────────────────────────────────────────────────────

    def free_point(self, clearance: float, avoid: List[Tuple[float, float]] = (),
                   tries: int = 500) -> Optional[Tuple[float, float]]:
        """Random full-res point on the floor at least `clearance` px from
        obstacles, the floor edge and the `avoid` points."""
        m = clearance
        for _ in range(tries):
            fx0, fy0, fx1, fy1 = self.floor_rect
            x = float(self.rng.uniform(fx0 + m, fx1 - m))
            y = float(self.rng.uniform(fy0 + m, fy1 - m))
            if self._clear_of_obstacles(x, y, clearance) and \
                    all(math.hypot(x - ax, y - ay) > 2 * clearance for ax, ay in avoid):
                return x, y
        return None

    def add_cart(self, marker_id: int, at: Optional[Tuple[float, float]] = None,
                 heading: Optional[float] = None,
                 params: Optional[dict] = None) -> VirtualCart:
        """Put a cart on the floor. `at` (full-res px) and `heading` (travel
        direction, deg) default to a free spot / random; `params` may set
        marker_offset, px_per_unit, motor_tau, deadband, gain_l, gain_r."""
        params = dict(params or {})
        if at is None:
            # Room around it, as an operator leaves for the 🧭 heading calibration
            at = self.free_point(4 * self.cart_radius,
                                 [(c.x, c.y) for c in self.carts.values()]) \
                or self.free_point(2 * self.cart_radius,
                                   [(c.x, c.y) for c in self.carts.values()]) \
                or ((self.floor_rect[0] + self.floor_rect[2]) / 2,
                    (self.floor_rect[1] + self.floor_rect[3]) / 2)
        mm   = self.cfg.wheel_mismatch
        cart = VirtualCart(marker_id, self._cart_sprite, np.zeros((1, 1), np.float32),
                           float(at[0]), float(at[1]),
                           float(self.rng.uniform(-180, 180) if heading is None else heading),
                           gain_l=float(params.get("gain_l", 1 + self.rng.uniform(-mm, mm))),
                           gain_r=float(params.get("gain_r", 1 + self.rng.uniform(-mm, mm))),
                           px_per_unit=float(params.get("px_per_unit", self.cfg.px_per_unit)),
                           motor_tau=float(params.get("motor_tau", self.cfg.motor_tau)),
                           deadband=int(params.get("deadband", self.cfg.deadband)))
        self.carts[marker_id] = cart
        self.set_mounting(marker_id, float(params.get("marker_offset", self.cfg.marker_offset)))
        return cart

    def set_mounting(self, marker_id: int, marker_offset: float) -> None:
        """Mount the cart's marker at `marker_offset` (marker − travel, multiple
        of 90°). The photographed body drives toward (marker − MARKER_MINUS_TRAVEL),
        so a different mounting turns the marker on the body, never the body."""
        c     = self.carts[marker_id]
        turns = round((marker_offset - MARKER_MINUS_TRAVEL) / 90.0)
        c.marker_offset = (MARKER_MINUS_TRAVEL + 90.0 * turns + 180) % 360 - 180
        c.sprite = remark(self._cart_sprite, marker_id, turns)
        gloss = np.zeros(c.sprite.rgba.shape[:2], np.float32)
        cv2.fillConvexPoly(gloss, c.sprite.marker.astype(np.int32), 1.0)
        c.gloss = cv2.GaussianBlur(gloss, (7, 7), 0)
        c.gloss_sprite = Sprite(np.dstack([c.gloss] * 4), c.sprite.anchor, 0)

    def remove_cart(self, marker_id: int) -> None:
        self.carts.pop(marker_id, None)

    def cart_params(self, marker_id: int) -> dict:
        c = self.carts[marker_id]
        return {"marker_offset": c.marker_offset, "px_per_unit": c.px_per_unit,
                "motor_tau": c.motor_tau, "deadband": c.deadband,
                "gain_l": c.gain_l, "gain_r": c.gain_r}

    def can_place_cart(self, x: float, y: float) -> bool:
        """Free floor for a cart body at (x, y): inside the floor, off
        obstacles and away from other carts."""
        fx0, fy0, fx1, fy1 = self.floor_rect
        r = self.cart_radius
        return fx0 + r <= x <= fx1 - r and fy0 + r <= y <= fy1 - r and \
            self._clear_of_obstacles(x, y, 1.2 * r) and \
            all(math.hypot(x - c.x, y - c.y) > 2.2 * r for c in self.carts.values())

    def cart_at(self, x: float, y: float) -> Optional[int]:
        """Marker id of the cart covering full-res point (x, y), if any."""
        for mid, c in self.carts.items():
            if math.hypot(x - c.x, y - c.y) < 1.3 * self.cart_radius:
                return mid
        return None

    # ── Floor-normalised coordinates (templates survive orientation changes) ──

    def to_norm(self, x: float, y: float) -> Tuple[float, float]:
        fx0, fy0, fx1, fy1 = self.floor_rect
        return (x - fx0) / (fx1 - fx0), (y - fy0) / (fy1 - fy0)

    def from_norm(self, u: float, v: float) -> Tuple[float, float]:
        fx0, fy0, fx1, fy1 = self.floor_rect
        return fx0 + u * (fx1 - fx0), fy0 + v * (fy1 - fy0)

    def add_obstacle(self, at: Optional[Tuple[float, float]] = None,
                     spread: float = 60.0) -> bool:
        """Put an obstacle on free floor (not on a cart): anywhere, or near `at`
        (full-res px) — where a person sets it down."""
        for _ in range(60):
            idx = int(self.rng.integers(len(self._obstacle_sprites)))
            if at is None:
                fx0, fy0, fx1, fy1 = self.floor_rect
                x = float(self.rng.uniform(fx0 + 80, fx1 - 80))
                y = float(self.rng.uniform(fy0 + 80, fy1 - 80))
            else:
                x, y = (float(v + self.rng.normal(0, spread)) for v in at)
            if self.place_obstacle(idx, x, y, float(self.rng.uniform(-180, 180))):
                return True
        return False

    @property
    def n_obstacle_kinds(self) -> int:
        return len(self._obstacle_sprites)

    def place_obstacle(self, sprite_idx: int, x: float, y: float, angle: float,
                       check: bool = True) -> bool:
        """Put obstacle sprite `sprite_idx` exactly at (x, y) full-res px, rotated
        by `angle`. With `check`, refuse spots off the floor, touching another
        obstacle or right against a cart."""
        spr = self._obstacle_sprites[sprite_idx % len(self._obstacle_sprites)]
        patch, x0, y0, _ = place(spr, x, y, angle)
        poly = _alpha_polygon(patch, x0, y0).astype(np.float32)
        if check:
            fx0, fy0, fx1, fy1 = self.floor_rect
            if poly[:, 0].min() < fx0 + 10 or poly[:, 1].min() < fy0 + 10 or \
                    poly[:, 0].max() > fx1 - 20 or poly[:, 1].max() > fy1 - 20:
                return False
            m = np.zeros_like(self._occ)
            cv2.fillPoly(m, [poly.astype(np.int32)], 1)
            near = cv2.dilate(m, np.ones((OBSTACLE_GAP_PX, OBSTACLE_GAP_PX), np.uint8))
            if (near & self._occ).any():
                return False
            clear = 3.0 * self.cart_radius          # never drop it right against a cart
            if any(cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (c.x, c.y), True) > -clear
                   for c in self.carts.values()):
                return False
        rp, rx0, ry0, _ = place(spr, x * self.cfg.scale, y * self.cfg.scale,
                                angle, scale=self.cfg.scale)
        self.obstacles.append(WorldObstacle(rp, rx0, ry0, poly, spr.illum,
                                            sprite_idx % len(self._obstacle_sprites),
                                            x, y, angle))
        self._rebuild_occupancy()
        return True

    def obstacle_at(self, x: float, y: float) -> Optional[int]:
        """Index of the obstacle under full-res point (x, y), if any."""
        for i, ob in enumerate(self.obstacles):
            if cv2.pointPolygonTest(ob.polygon.reshape(-1, 1, 2), (x, y), True) > -15:
                return i
        return None

    def clear_obstacles(self) -> None:
        self.obstacles.clear()
        self._rebuild_occupancy()

    def remove_obstacle(self, index: Optional[int] = None) -> bool:
        if not self.obstacles:
            return False
        if index is None:
            index = int(self.rng.integers(len(self.obstacles)))
        self.obstacles.pop(index)
        self._rebuild_occupancy()
        return True

    def _rebuild_occupancy(self) -> None:
        self._occ[:] = 0
        for ob in self.obstacles:
            cv2.fillPoly(self._occ, [ob.polygon.astype(np.int32)], 1)
        k = 2 * int(self.cart_radius) + 1
        self._inflated = cv2.dilate(self._occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    def _clear_of_obstacles(self, x: float, y: float, clearance: float) -> bool:
        xi, yi = int(x), int(y)
        if not (0 <= xi < self.W and 0 <= yi < self.H):
            return False
        r = int(clearance)
        win = self._occ[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1]
        if not win.any():
            return True
        ys, xs = np.nonzero(win)
        d = np.hypot(xs + max(0, xi - r) - x, ys + max(0, yi - r) - y)
        return bool(d.min() > clearance)

    # ── People ────────────────────────────────────────────────────────────────

    def spawn_walker(self, kind: str = "pass") -> bool:
        """A person walking through: 'pass' (straight across), 'visit' (walks
        to a spot, stands a while, leaves), 'drop' (visit + sets an object
        down), 'pickup' (walks to an object, picks it up, leaves)."""
        r_c   = self.cart_radius
        speed = float(self.rng.uniform(1000, 1400)) * PX_PER_MM        # px/s
        for _ in range(150):
            entry = self._border_point()
            if kind == "pass":
                exit_ = self._border_point(not_side_of=entry)
                path, pause_at = [entry, exit_], None
                if not self._walk_clear(path):
                    # Go around things: via a free point somewhere on the floor
                    mid = self.free_point(1.5 * r_c, [(c.x, c.y) for c in self.carts.values()])
                    if mid is not None:
                        path = [entry, mid, exit_]
            else:
                if kind == "pickup":
                    if not self.obstacles:
                        return False
                    idx    = int(self.rng.integers(len(self.obstacles)))
                    ox, oy = self.obstacles[idx].polygon.mean(axis=0)
                    d      = np.array([ox - entry[0], oy - entry[1]])
                    stand  = np.array([ox, oy]) - d / np.linalg.norm(d) * 330 * PX_PER_MM
                    visit  = (float(stand[0]), float(stand[1]))
                else:
                    visit = self.free_point(1.5 * r_c, [(c.x, c.y) for c in self.carts.values()])
                    if visit is None:
                        continue
                exit_ = self._border_point()
                path, pause_at = [entry, visit, exit_], 1
            if not self._walk_clear(path, None if kind != "pickup" else self.obstacles[idx]):
                continue                                   # people walk around things
            w = Walker(path, speed, pause_at, float(self.rng.uniform(1.5, 4.0)),
                       None if kind in ("pass", "visit") else kind,
                       trousers=TROUSERS[int(self.rng.integers(len(TROUSERS)))],
                       shirt=SHIRTS[int(self.rng.integers(len(SHIRTS)))],
                       skin=SKIN[int(self.rng.integers(len(SKIN)))],
                       hair=HAIR[int(self.rng.integers(len(HAIR)))])
            if kind == "pickup":
                w.target = self.obstacles[idx]
            self.walkers.append(w)
            self.walker_log.append((self.t, f"person enters ({kind})"))
            return True
        return False

    def _walk_clear(self, path, target=None) -> bool:
        """Does a walking route keep the feet off obstacles and the body clear
        of carts? (The object being picked up is allowed to be close.)"""
        feet_clear = 220 * PX_PER_MM                       # half the stance + margin
        # The feet must clear a cart; the torso may pass above it (carts are low)
        body_clear = feet_clear + self.cart_radius
        for k in range(len(path) - 1):
            (ax, ay), (bx, by) = path[k], path[k + 1]
            n = max(2, int(math.hypot(bx - ax, by - ay) / 40))
            for t in np.linspace(0, 1, n):
                x, y = ax + (bx - ax) * t, ay + (by - ay) * t
                if any(math.hypot(x - c.x, y - c.y) < body_clear for c in self.carts.values()):
                    return False
                for ob in self.obstacles:
                    if ob is target:
                        continue
                    if cv2.pointPolygonTest(ob.polygon.reshape(-1, 1, 2),
                                            (float(x), float(y)), True) > -feet_clear:
                        return False
        return True

    def _border_point(self, not_side_of: Optional[Tuple[float, float]] = None):
        """Start/end point just outside the floor area, on a random side (other
        than the side of `not_side_of`, for a walk across)."""
        out = 350 * PX_PER_MM
        sides = ["left", "right", "top", "bottom"]
        if not_side_of is not None:
            x, y = not_side_of
            fx0, fy0, fx1, fy1 = self.floor_rect
            sides.remove("left" if x < fx0 else "right" if x > fx1 else
                         "top" if y < fy0 else "bottom")
        side = sides[int(self.rng.integers(len(sides)))]
        u = float(self.rng.uniform(0.15, 0.85))
        fx0, fy0, fx1, fy1 = self.floor_rect
        return {"left":   (fx0 - out, fy0 + u * (fy1 - fy0)),
                "right":  (fx1 + out, fy0 + u * (fy1 - fy0)),
                "top":    (fx0 + u * (fx1 - fx0), fy0 - out),
                "bottom": (fx0 + u * (fx1 - fx0), fy1 + out)}[side]

    def _step_walkers(self, dt: float) -> None:
        rate = self.cfg.walkers_per_min
        if self.walkers_enabled and rate > 0 and self.t >= self._next_walker:
            if self._next_walker > 0:
                kinds = ["pass", "pass", "visit", "drop"] + (["pickup"] if self.obstacles else [])
                self.spawn_walker(kinds[int(self.rng.integers(len(kinds)))])
            self._next_walker = self.t + float(self.rng.exponential(60.0 / rate))
        for w in list(self.walkers):
            event = w.step(dt)
            if event == "action" and w.action == "drop":
                fx, fy = w.front(380 * PX_PER_MM)
                ok = any(self.add_obstacle(at=(fx, fy), spread=s_)
                         for s_ in (40, 120, 220))
                self.walker_log.append((self.t, "person sets an object down" if ok
                                        else "person couldn't set the object down"))
            elif event == "action" and w.action == "pickup":
                if w.target in self.obstacles:
                    self.obstacles.remove(w.target)
                    self._rebuild_occupancy()
                    self.walker_log.append((self.t, "person picks an object up"))
            elif event == "done":
                self.walkers.remove(w)
                self.walker_log.append((self.t, "person leaves"))

    # ── Physics ───────────────────────────────────────────────────────────────

    def step(self, dt: float) -> None:
        self.t += dt
        self._step_walkers(dt)
        for c in self.carts.values():
            while c.queue and c.queue[0][0] <= self.t:
                c.cmd = c.queue.pop(0)[1]
            tl = self._wheel_target(c, c.cmd[0]) * c.gain_l
            tr = self._wheel_target(c, c.cmd[1]) * c.gain_r
            a  = 1 - math.exp(-dt / max(c.motor_tau, 1e-3))
            c.vl += (tl - c.vl) * a
            c.vr += (tr - c.vr) * a

            v     = (c.vl + c.vr) / 2
            omega = (c.vr - c.vl) / self.track              # rad/s, CCW positive
            h     = math.radians(c.heading) + omega * dt / 2
            nx    = c.x + v * math.cos(h) * dt
            ny    = c.y - v * math.sin(h) * dt
            new_heading = (c.heading + math.degrees(omega * dt) + 180) % 360 - 180

            hit = self._blocked(c, nx, ny)
            if hit:
                # Bumped into something: wheels slip, no translation — it can still turn.
                # One collision per contact (a cart pushing on and off counts once).
                if not c._touching and self.t - c._free_since > 0.5:
                    if hit == "person" and c.cmd == (0, 0):
                        # Already told to stop, still braking: the person walked in
                        self.collision_log.append((self.t, c.marker_id, "person (cart braking)"))
                    else:
                        c.collisions += 1
                        self.collision_log.append((self.t, c.marker_id, hit))
                c._touching = True
                c.heading = new_heading
            else:
                if c._touching:
                    c._free_since = self.t
                c._touching = False
                c.x, c.y, c.heading = nx, ny, new_heading

    @staticmethod
    def _wheel_target(c: VirtualCart, cmd: int) -> float:
        if abs(cmd) < c.deadband:
            return 0.0
        return cmd * c.px_per_unit

    def _blocked(self, c: VirtualCart, x: float, y: float) -> Optional[str]:
        """What the cart would hit at (x, y): 'edge', 'obstacle', 'cart' or None."""
        r = self.cart_radius
        fx0, fy0, fx1, fy1 = self.floor_rect
        if not (fx0 + r <= x <= fx1 - r and fy0 + r <= y <= fy1 - r):
            return "edge"
        xi, yi = int(x), int(y)
        if self._inflated[yi, xi]:
            return "obstacle"
        if any(o is not c and math.hypot(x - o.x, y - o.y) < 2 * r and
               math.hypot(x - o.x, y - o.y) < math.hypot(c.x - o.x, c.y - o.y)
               for o in self.carts.values()):
            return "cart"
        for w in self.walkers:
            for shoe in w.shoes():
                d_new = -cv2.pointPolygonTest(shoe.reshape(-1, 1, 2), (x, y), True)
                if d_new < r:
                    d_old = -cv2.pointPolygonTest(shoe.reshape(-1, 1, 2), (c.x, c.y), True)
                    if d_new < d_old:          # the cart moved into the person
                        return "person"
        return None

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self) -> np.ndarray:
        """Current view as a BGR frame at render resolution."""
        n = len(self._floors)
        j = self.frame_no % (2 * n - 2)
        frame = cv2.imdecode(self._floors[j if j < n else 2 * n - 2 - j], cv2.IMREAD_COLOR)
        self.frame_no += 1

        s = self.cfg.scale
        for ob in self.obstacles:
            composite(frame, ob.patch, ob.x0, ob.y0, self._illum_r, ob.src_illum)
        for c in self.carts.values():
            if c.hidden:
                continue
            rot = c.heading + c.marker_offset - c.sprite.marker_angle
            patch, x0, y0, _ = place(c.sprite, c.x * s, c.y * s, rot, scale=s)
            gl, _, _, _ = place(c.gloss_sprite, c.x * s, c.y * s, rot, scale=s)
            composite(frame, patch, x0, y0, self._illum_r, c.sprite.illum, gl[..., 0],
                      0.8, (abs(c.speed) * s / self.cfg.fps * 0.5, c.heading))
        for w in self.walkers:
            w.render(frame, s, (self.W / 2, self.H / 2))

        self._Mc = self._drift.step((self.W / 2, self.H / 2))
        if self.cfg.drift_px > 0:
            Mr = self._Mc.copy()
            Mr[:, 2] *= s                                   # same motion, render px
            frame = cv2.warpAffine(frame, Mr, self.render_size,
                                   borderMode=cv2.BORDER_REFLECT)
        return frame

    # ── Ground truth / calibration (render px) ────────────────────────────────

    def truth(self) -> Dict[int, Tuple[float, float, float]]:
        """marker_id → (x, y, marker heading) in render px / degrees."""
        s, out = self.cfg.scale, {}
        for mid, c in self.carts.items():
            p = cv2.transform(np.float32([[[c.x, c.y]]]), self._Mc).reshape(2)
            out[mid] = (float(p[0] * s), float(p[1] * s),
                        (c.heading + c.marker_offset + 180) % 360 - 180)
        return out

    def to_render(self, x: float, y: float) -> Tuple[float, float]:
        return x * self.cfg.scale, y * self.cfg.scale

    def calibration_points(self, rows: int, cols: int) -> Tuple[List[Tuple[float, float, int, int]], int, int]:
        """Four (px, py, row, col) correspondences mapping the floor area onto the
        grid with square cells. A landscape view maps straight onto the
        landscape grid; a portrait view is rotated 90° onto it (image down →
        grid right, image right → grid up). Returns (points, used_rows, used_cols)."""
        s = self.cfg.scale
        x0, y0, x1, y1 = (v * s for v in self.floor_rect)
        fw, fh = x1 - x0, y1 - y0
        if fw >= fh:                                         # landscape: no rotation
            cell = max(fw / cols, fh / rows)                 # render px per cell
            used_r, used_c = int(fh / cell), int(fw / cell)
            pts = [(x0, y0, 0, 0), (x1, y0, 0, used_c),
                   (x1, y1, used_r, used_c), (x0, y1, used_r, 0)]
        else:
            cell = max(fh / cols, fw / rows)
            used_r, used_c = int(fw / cell), int(fh / cell)
            pts = [(x0, y0, used_r, 0), (x1, y0, 0, 0),
                   (x1, y1, 0, used_c), (x0, y1, used_r, used_c)]
        return pts, used_r, used_c


# ── People ────────────────────────────────────────────────────────────────────
#
# A person as an overhead camera sees them: head (hair) on top of a wide
# shoulder ellipse, arms swinging at the sides, shoes showing in front of and
# behind the body while walking, a soft shadow. Parts closer to the camera are
# shifted slightly away from the image centre (mild perspective) and are a bit
# larger. Scale: the marker is ≈70 mm and ≈71 px, so 1 px ≈ 1 mm.

PX_PER_MM        = 1.0
CAMERA_HEIGHT_MM = 6000.0     # only sets the (mild) parallax of the upper body
STEP_HZ          = 1.8
SKIN     = [(120, 150, 200), (80, 110, 160), (60, 80, 120), (150, 175, 215)]   # BGR
HAIR     = [(20, 25, 30), (30, 50, 75), (40, 40, 40), (60, 100, 140)]
TROUSERS = [(70, 45, 30), (35, 35, 38), (60, 85, 105), (40, 40, 60)]
SHIRTS   = [(150, 90, 40), (60, 60, 170), (200, 200, 200), (40, 120, 60), (30, 30, 30)]


@dataclass
class Walker:
    path: List[Tuple[float, float]]     # full-res px: entry, [visit], exit
    speed: float                        # px/s
    pause_at: Optional[int]             # path index where they stand a while
    pause_s: float
    action: Optional[str]               # None | "drop" | "pickup"
    trousers: Tuple[int, int, int] = (70, 45, 30)
    shirt: Tuple[int, int, int] = (150, 90, 40)
    skin: Tuple[int, int, int] = (120, 150, 200)
    hair: Tuple[int, int, int] = (20, 25, 30)
    target: Optional[object] = None     # obstacle to pick up
    seg: int = 0
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0                # rad, image convention (atan2(dy, dx))
    phase: float = 0.0                  # gait phase, strides
    stand: float = 0.0                  # 0 walking → 1 standing (feet together)
    paused: float = 0.0
    acted: bool = False

    def __post_init__(self):
        self.x, self.y = self.path[0]
        self._aim()

    def _aim(self) -> None:
        tx, ty = self.path[self.seg + 1]
        self.heading = math.atan2(ty - self.y, tx - self.x)

    def step(self, dt: float) -> Optional[str]:
        """Advance; returns 'action' when the drop/pickup happens, 'done' when gone."""
        if self.paused > 0:
            self.paused -= dt
            self.stand = min(1.0, self.stand + dt * 3)
            if not self.acted and self.paused < self.pause_s / 2:
                self.acted = True
                if self.action:
                    return "action"
            return None
        self.stand = max(0.0, self.stand - dt * 3)
        tx, ty = self.path[self.seg + 1]
        dist   = math.hypot(tx - self.x, ty - self.y)
        move   = self.speed * dt
        self.phase += STEP_HZ / 2 * dt
        if dist <= move:
            self.x, self.y = tx, ty
            self.seg += 1
            if self.seg + 1 >= len(self.path):
                return "done"
            if self.pause_at == self.seg:
                self.paused = self.pause_s
            self._aim()
        else:
            self.x += (tx - self.x) / dist * move
            self.y += (ty - self.y) / dist * move
        return None

    def front(self, dist: float) -> Tuple[float, float]:
        return self.x + math.cos(self.heading) * dist, self.y + math.sin(self.heading) * dist

    def _feet(self) -> List[Tuple[float, float]]:
        """Shoe centres: the stance foot stays planted while the other swings."""
        step   = self.speed / STEP_HZ                     # step length, px
        f      = self.phase % 1.0
        ra     = step / 2 - step * (f / 0.5) if f < 0.5 else -step / 2 + step * ((f - 0.5) / 0.5)
        rb     = -ra
        k      = 1 - self.stand
        fx, fy = math.cos(self.heading), math.sin(self.heading)
        lx, ly = -fy, fx
        lat    = 95 * PX_PER_MM
        return [(self.x + fx * ra * k + lx * lat, self.y + fy * ra * k + ly * lat),
                (self.x + fx * rb * k - lx * lat, self.y + fy * rb * k - ly * lat)]

    def shoes(self) -> List[np.ndarray]:
        """Shoe outlines on the floor, full-res px (what a cart can bump into)."""
        out = []
        fx, fy = math.cos(self.heading), math.sin(self.heading)
        lx, ly = -fy, fx
        L, Wd  = 270 * PX_PER_MM / 2, 100 * PX_PER_MM / 2
        for cx, cy in self._feet():
            out.append(np.float32([[cx + fx * L + lx * Wd, cy + fy * L + ly * Wd],
                                   [cx + fx * L - lx * Wd, cy + fy * L - ly * Wd],
                                   [cx - fx * L - lx * Wd, cy - fy * L - ly * Wd],
                                   [cx - fx * L + lx * Wd, cy - fy * L + ly * Wd]]))
        return out

    def render(self, frame: np.ndarray, s: float, nadir: Tuple[float, float]) -> None:
        H      = CAMERA_HEIGHT_MM * PX_PER_MM
        cx, cy = nadir
        fx, fy = math.cos(self.heading), math.sin(self.heading)
        lx, ly = -fy, fx
        mm     = PX_PER_MM

        def at(along: float, side: float, z: float) -> Tuple[float, float]:
            """Body-frame point (mm ahead, mm to the left, mm up) → render px."""
            px = self.x + (fx * along + lx * side) * mm
            py = self.y + (fy * along + ly * side) * mm
            k  = H / (H - z * mm)
            return (cx + (px - cx) * k) * s, (cy + (py - cy) * k) * s

        def ellipse(along, side, z, a_mm, b_mm):
            """Ellipse with a_mm along the walking direction, b_mm across."""
            k  = H / (H - z * mm)
            ex, ey = at(along, side, z)
            return _ellipse_poly(ex, ey, a_mm * mm * k * s, b_mm * mm * k * s, self.heading)

        swing = math.sin(2 * math.pi * self.phase) * (1 - self.stand)

        _paint(frame, [ellipse(0, 0, 0, 300, 330)], (0, 0, 0), 0.20, 25 * s)    # shadow
        for shoe in self.shoes():                                                # shoes
            _paint(frame, [shoe * s], (38, 38, 42), 0.95, 1.0 * s)
            # trouser cuff over the back of the shoe
            c = shoe.mean(axis=0)
            _paint(frame, [_ellipse_poly(c[0] * s - fx * 40 * mm * s, c[1] * s - fy * 40 * mm * s,
                                         70 * mm * s, 60 * mm * s, self.heading)],
                   self.trousers, 0.9, 1.5 * s)
        # Picking up / setting down: bent forward, both arms reaching out in front
        handling = self.action is not None and self.paused > 0
        for side, sgn in ((240, 1), (-240, -1)):                                 # arms + hands
            if handling:
                _paint(frame, [ellipse(230, side * 0.55, 900, 230, 55)], self.shirt, 0.97, 1.5 * s)
                _paint(frame, [ellipse(430, side * 0.35, 300, 60, 48)], self.skin, 0.97, 1.2 * s)
                continue
            reach = 140 * swing * -sgn            # arms swing opposite to the legs
            _paint(frame, [ellipse(reach * 0.5, side, 1150, 170, 55)], self.shirt, 0.97, 1.5 * s)
            _paint(frame, [ellipse(reach * 1.1 + 40, side * 1.02, 850, 50, 42)],
                   self.skin, 0.97, 1.2 * s)
        torso = ellipse(0, 0, 1350, 140, 235)                                    # shoulders
        _paint(frame, [torso], tuple(int(v * 0.75) for v in self.shirt), 0.98, 1.5 * s)
        _paint(frame, [ellipse(0, 0, 1360, 110, 200)], self.shirt, 0.98, 4 * s)  # lit top
        _paint(frame, [ellipse(25, 0, 1650, 115, 95)], self.hair, 0.98, 1.5 * s) # head
        _paint(frame, [ellipse(95, 0, 1600, 30, 55)], self.skin, 0.9, 1.5 * s)   # forehead/nose


def _ellipse_poly(x, y, a, b, ang) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ca, sa = math.cos(ang), math.sin(ang)
    return np.float32(np.stack([x + a * np.cos(t) * ca - b * np.sin(t) * sa,
                                y + a * np.cos(t) * sa + b * np.sin(t) * ca], 1))


_RNG = np.random.default_rng(12)


def _paint(frame: np.ndarray, polys, colour, alpha: float, blur: float) -> None:
    """Blend filled polygons (render px) into frame with soft (defocused) edges."""
    h, w = frame.shape[:2]
    pts  = np.vstack(polys)
    pad  = int(3 * blur) + 2
    x0, y0 = np.maximum(np.floor(pts.min(0)).astype(int) - pad, 0)
    x1, y1 = np.minimum(np.ceil(pts.max(0)).astype(int) + pad, [w, h])
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.fillPoly(m, [np.round(p - [x0, y0]).astype(np.int32) for p in polys], 1.0)
    if blur > 0.5:
        m = cv2.GaussianBlur(m, (0, 0), blur)
    a   = (m * alpha)[..., None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    # Fabric / hair texture: low-frequency shading plus fine grain
    grain = _RNG.normal(0, 7, m.shape).astype(np.float32)
    shade = cv2.GaussianBlur(_RNG.normal(0, 14, m.shape).astype(np.float32), (0, 0), 6)
    col   = np.float32(colour)[None, None, :] + (grain + shade)[..., None]
    frame[y0:y1, x0:x1] = np.clip(roi * (1 - a) + col * a, 0, 255).astype(np.uint8)


def _seg_dist(p, a, b) -> float:
    p, a, b = (np.asarray(v, np.float64) for v in (p, a, b))
    ab = b - a
    t  = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


# ── Qt adapters ───────────────────────────────────────────────────────────────

class VirtualCamera(QObject):
    """Drop-in for production.camera_source.CameraSource."""
    frame_ready = pyqtSignal(QImage)

    def __init__(self, world: VirtualWorld, parent=None):
        super().__init__(parent)
        self.world  = world
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.tick)
        self._interval = max(10, 1000 // world.cfg.fps)
        self.paused = False

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.world.render_size

    def is_open(self) -> bool:
        return True

    def start(self) -> None:
        self._timer.start(self._interval)

    def stop(self) -> None:
        self._timer.stop()

    def release(self) -> None:
        self.stop()

    def tick(self) -> None:
        """Advance the world by one frame period and emit the rendered frame.
        Simulated time advances per frame, so a slow PC slows the world down
        instead of making the carts jump."""
        if self.paused:
            return
        self.world.step(1.0 / self.world.cfg.fps)
        self.frame_ready.emit(bgr_to_qimage(self.world.render()))


class VirtualBTController(QObject):
    """Drop-in for production.bt_controller.BTController; commands go to the
    VirtualWorld cart whose marker ID is encoded in the device address."""
    connected    = pyqtSignal(str)
    disconnected = pyqtSignal(str)
    error        = pyqtSignal(str, str)
    scan_result  = pyqtSignal(list)

    def __init__(self, world: VirtualWorld, parent=None):
        super().__init__(parent)
        self.world = world
        self._devices:   Dict[str, BTDevice] = {}
        self._connected: Dict[str, bool]     = {}
        self.sent = 0

    @staticmethod
    def device_for_marker(marker_id: int) -> BTDevice:
        return BTDevice(f"VIRT:{marker_id:02d}", f"VirtualCart-{marker_id}")

    def assign(self, cart_id: str, device: Optional[BTDevice]) -> None:
        if device is None:
            self._devices.pop(cart_id, None)
        else:
            self._devices[cart_id] = device

    def unassign(self, cart_id: str) -> None:
        self.disconnect(cart_id)
        self._devices.pop(cart_id, None)

    def device_for(self, cart_id: str) -> Optional[BTDevice]:
        return self._devices.get(cart_id)

    def is_connected(self, cart_id: str) -> bool:
        return self._connected.get(cart_id, False)

    def scan_async(self) -> None:
        self.scan_result.emit([self.device_for_marker(m) for m in sorted(self.world.carts)])

    def connect_cart(self, cart_id: str) -> None:
        if self._cart(cart_id) is None:
            self.error.emit(cart_id, "No virtual cart for this device")
            return
        self._connected[cart_id] = True
        self.connected.emit(cart_id)

    def disconnect(self, cart_id: str) -> None:
        if self._connected.pop(cart_id, False):
            self.stop(cart_id)
            self.disconnected.emit(cart_id)

    def disconnect_all(self) -> None:
        for cid in list(self._connected):
            self.disconnect(cid)

    def _cart(self, cart_id: str) -> Optional[VirtualCart]:
        dev = self._devices.get(cart_id)
        if dev is None or not dev.address.startswith("VIRT:"):
            return None
        return self.world.carts.get(int(dev.address.split(":")[1]))

    def send(self, cart_id: str, command: str) -> None:
        cart = self._cart(cart_id)
        if cart is None or not self._connected.get(cart_id):
            return
        self.sent += 1
        parts = command.strip().split(":")
        if parts[0] in ("DRIVE", "RAW") and len(parts) == 3:   # no boost modelled
            cart.command(self.world.t, int(parts[1]), int(parts[2]), self.world.cfg.latency)
        elif parts[0] == "STOP":
            cart.command(self.world.t, 0, 0, self.world.cfg.latency)

    def drive(self, cart_id: str, left: int, right: int) -> None:
        left  = max(-255, min(255, int(left)))
        right = max(-255, min(255, int(right)))
        self.send(cart_id, f"DRIVE:{left}:{right}")

    def drive_raw(self, cart_id: str, left: int, right: int) -> None:
        left  = max(-255, min(255, int(left)))
        right = max(-255, min(255, int(right)))
        self.send(cart_id, f"RAW:{left}:{right}")

    def stop(self, cart_id: str) -> None:
        self.send(cart_id, "STOP")

    def stop_all(self) -> None:
        for cid in list(self._connected):
            self.stop(cid)

    def assignments(self) -> Dict[str, Optional[BTDevice]]:
        return {cid: self._devices.get(cid) for cid in self._devices}


# ── Wiring helpers (shared by the app's virtual mode and tools/run_twin.py) ────

def calibrate_to_world(world: VirtualWorld, calib, grid) -> None:
    """Camera→grid calibration as an operator would do it: the floor corners
    clicked in the image (with TwinConfig.calib_noise_px of click error) and
    their grid coordinates typed in. Grid cells beyond the camera's floor area
    become walls (unreachable)."""
    pts, used_r, used_c = world.calibration_points(grid.rows, grid.cols)
    rng = np.random.default_rng(world.cfg.seed + 7)
    calib.clear()
    for px, py, r, c in pts:
        nx, ny = rng.normal(0, world.cfg.calib_noise_px, 2)
        calib.add_point(px + nx, py + ny, r, c)
    for r in range(grid.rows):
        for c in range(grid.cols):
            grid.set_wall(r, c, r >= used_r or c >= used_c)


def mission_goal(world: VirtualWorld, calib, avoid_render: List[Tuple[float, float]] = ()):
    """Random destination for a mission, as an operator would pick one: a free
    spot on the floor, away from `avoid_render` points (render px — pass where
    the other carts are *seen* and the goals already handed out). Whether it is
    reachable is left to the engine's planner. Returns ((x, y) render px,
    (row, col)) or None."""
    s     = world.cfg.scale
    avoid = [(x / s, y / s) for x, y in avoid_render]
    goal  = world.free_point(2.2 * world.cart_radius, avoid)
    if goal is None:
        return None
    gx, gy = world.to_render(*goal)
    r, c   = calib.pixel_to_grid(gx, gy)
    return (gx, gy), (int(round(r)), int(round(c)))


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()
