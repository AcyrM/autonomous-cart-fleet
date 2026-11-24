"""
Scenario tests of the full production stack in the digital twin.

Each test is one route: a fresh virtual floor (random layout and route from the
test's seed), the operator's set-up (floor capture, 🧭 heading calibration),
then the cart(s) drive a 2-waypoint routine while the scenario's objects and
people do their thing. The real ProdEngine only sees rendered camera images
(ArUco tracking, background-subtraction obstacles, YOLO11n people) and only
sends DRIVE/STOP commands. Ground truth from the virtual world is used only
to score each test on four criteria:

  representation  every real object/person on the floor shows up in the
                  occupancy matrix (≥50 % of its cells marked obstacle / moving /
                  person). Fails if one is missing for ≥ MISS_FAIL_S in a row
                  (after APPEAR_GRACE_S from appearing).
  trajectory      the cart's blue area (radius = engine cart radius, as drawn
                  in the occupancy view) never leaves the planned route line:
                  fails if the true centre is farther than that radius from the
                  active path for ≥ OFFPATH_FRAMES frames in a row.
  collisions      no physical contact (cart body vs object / person / other
                  cart / floor edge) — real bodies, not the safety margins.
  false stops     stops the engine attributes to a person / moving object
                  when no real person or moving object is anywhere near the
                  cart's look-ahead. Reported as total stops and tests with ≥1.

Scenarios:
  1  3 static objects, 1 cart
  2  3 static objects + 1 moving object crossing the route, 1 cart
  3  3 static objects + people walking across / stopping on the route, 1 cart
  4  3 static objects, fleet of 3 carts (each its own route, simultaneously)

The engine's clock (datetime.now in prod_engine) is replaced by simulated time,
so step timing, waits and timeouts are exactly as in real time even though the
simulation runs at whatever speed the PC allows. YOLO runs every 3rd frame
(10 Hz) and its result is used from the next submission on (~100 ms latency).

Usage:
    python tools/scenario_tests.py --scenario 1 --runs 25 --out results/scenario_tests
    python tools/scenario_tests.py --summary results/scenario_tests     # table only
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import cv2                                                        # noqa: E402
import numpy as np                                                # noqa: E402
from PyQt6.QtWidgets import QApplication                          # noqa: E402

import production.prod_engine as pe                               # noqa: E402
from config import GRID_COLS, GRID_ROWS                           # noqa: E402
from core.cart import Cart                                        # noqa: E402
from core.grid import Grid                                        # noqa: E402
from core.routine import SCHEDULE_TIME, Routine                   # noqa: E402
from core.routine_manager import RoutineManager                   # noqa: E402
from production.calibration import FloorCalibration               # noqa: E402
from production.cart_detector import CartDetector                 # noqa: E402
from production.obstacle_detector import ObstacleDetector         # noqa: E402
from production.person_detector import PersonDetection            # noqa: E402
from simulation.digital_twin import (                             # noqa: E402
    CAMERA_HEIGHT_MM, PX_PER_MM, SHIRTS, SKIN, HAIR, TROUSERS, TwinConfig, VirtualBTController,
    VirtualWorld, Walker, bgr_to_qimage, calibrate_to_world,
)

SCENARIOS = {
    1: "1 – Apenas objetos estáticos",
    2: "2 – Objetos estáticos e móvel",
    3: "3 – Objetos estáticos e pessoas",
    4: "4 – Objetos estáticos e frota de 3 carrinhos",
}
N_STATIC          = 3
APPEAR_GRACE_S    = 1.0    # time an object/person gets to show up after appearing
MISS_FAIL_S       = 1.0    # missing from the matrix this long in a row = failure
REPR_COVER        = 0.5    # fraction of the object's cells that must be marked
OFFPATH_FRAMES    = 3      # frames in a row outside the route line = failure
CAL_TIMEOUT_S     = 90.0
ROUTE_TIMEOUT_S   = {1: 150.0, 2: 150.0, 3: 150.0, 4: 240.0}
YOLO_EVERY        = 3      # frames between YOLO submissions (30 fps → 10 Hz)
CART_COLORS       = ["#3b82f6", "#22c55e", "#f97316"]
OCC_THINGS        = (pe.OCC_OBSTACLE, pe.OCC_MOVER, pe.OCC_PERSON)


# ── Simulated clock for the engine ────────────────────────────────────────────

class _Clock:
    t = 0.0


_BASE = datetime(2026, 1, 1, 8, 0, 0)


class _SimDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return _BASE + timedelta(seconds=_Clock.t)


pe.datetime = _SimDatetime


# ── YOLO person detection, synchronous and paced in simulated time ────────────

class SimPersonDetector:
    """Same interface as production.person_detector.PersonDetector, but runs
    YOLO every YOLO_EVERY frames and publishes the result one period later,
    so the rate and latency are those of a real PC, not of this simulation."""
    _model = None

    def __init__(self, weights: Path, conf: float = 0.45):
        if SimPersonDetector._model is None:
            import torch
            torch.set_num_threads(int(os.environ.get("YOLO_THREADS", "2")))
            from ultralytics import YOLO
            m = YOLO(str(weights))
            m.fuse()
            SimPersonDetector._model = m
        self._conf, self.enabled = conf, True
        self._n, self._res, self._pending = 0, [], None
        self.inferences = 0

    @property
    def available(self) -> bool:
        return True

    def detect_async(self, frame_bgr: np.ndarray) -> None:
        self._n += 1
        if (self._n - 1) % YOLO_EVERY:
            return
        if self._pending is not None:
            self._res = self._pending
        self._pending = self._infer(frame_bgr)

    def _infer(self, frame_bgr):
        self.inferences += 1
        out = []
        for r in self._model(frame_bgr, classes=[0], conf=self._conf, verbose=False, imgsz=640):
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                out.append(PersonDetection(x1, y1, x2, y2, float(box.conf[0])))
        return out

    def latest(self):
        return list(self._res)


# ── World with moving objects ─────────────────────────────────────────────────

class Mover:
    """An object that moves by itself (pushed box / trolley): back and forth
    between a and b (full-res px), pausing at the ends. It never moves into a
    cart (waits, and turns back after a while)."""

    def __init__(self, ob, a, b, speed, pauses, rng, scale):
        self.ob, self.a, self.b = ob, np.float32(a), np.float32(b)
        self.speed, self.pauses, self.rng, self.scale = speed, pauses, rng, scale
        self.to_b, self.wait, self.blocked_for = True, 0.0, 0.0
        self.home = (ob.x, ob.y, ob.x0, ob.y0)

    def step(self, dt: float, world: "ScenarioWorld") -> None:
        if self.wait > 0:
            self.wait -= dt
            return
        tgt  = self.b if self.to_b else self.a
        pos  = np.float32([self.ob.x, self.ob.y])
        d    = tgt - pos
        dist = float(np.linalg.norm(d))
        if dist < 1e-3:
            self.to_b, self.wait = not self.to_b, float(self.rng.uniform(*self.pauses))
            return
        mv    = min(dist, self.speed * dt)
        delta = d / dist * mv
        poly  = self.ob.polygon + delta
        if not world.mover_clear(poly):
            self.blocked_for += dt
            if self.blocked_for > 2.0:
                self.to_b, self.blocked_for = not self.to_b, 0.0
            return
        self.blocked_for = 0.0
        self.ob.polygon = poly
        self.ob.x += float(delta[0])
        self.ob.y += float(delta[1])
        hx, hy, hx0, hy0 = self.home
        self.ob.x0 = int(round(hx0 + (self.ob.x - hx) * self.scale))
        self.ob.y0 = int(round(hy0 + (self.ob.y - hy) * self.scale))
        if mv >= dist - 1e-6:
            self.to_b, self.wait = not self.to_b, float(self.rng.uniform(*self.pauses))


class ScenarioWorld(VirtualWorld):
    def __init__(self, cfg):
        self.movers = []
        super().__init__(cfg)

    def _rebuild_occupancy(self) -> None:
        # Static obstacles only: movers are checked as polygons in _blocked
        moving = {id(m.ob) for m in getattr(self, "movers", [])}
        self._occ[:] = 0
        for ob in self.obstacles:
            if id(ob) not in moving:
                cv2.fillPoly(self._occ, [ob.polygon.astype(np.int32)], 1)
        # Same as dilating by a disc of the cart radius, ~20× faster
        dist = cv2.distanceTransform((self._occ == 0).astype(np.uint8), cv2.DIST_L2, 5)
        self._inflated = (dist <= int(self.cart_radius)).astype(np.uint8)

    def mover_clear(self, poly) -> bool:
        q = poly.reshape(-1, 1, 2)
        return all(cv2.pointPolygonTest(q, (c.x, c.y), True) < -(self.cart_radius + 25)
                   for c in self.carts.values())

    def _blocked(self, c, x, y):
        hit = super()._blocked(c, x, y)
        if hit:
            return hit
        r = self.cart_radius
        for m in self.movers:
            q = m.ob.polygon.reshape(-1, 1, 2)
            dn = -cv2.pointPolygonTest(q, (float(x), float(y)), True)
            if dn < r and dn < -cv2.pointPolygonTest(q, (c.x, c.y), True):
                return "moving object"
        return None

    def step(self, dt: float) -> None:
        for m in self.movers:
            m.step(dt, self)
        super().step(dt)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def poly_dist(poly, p) -> float:
    """Distance from point p to polygon (0 inside)."""
    return max(0.0, -cv2.pointPolygonTest(np.float32(poly).reshape(-1, 1, 2),
                                          (float(p[0]), float(p[1])), True))


def seg_point_dist(p, a, b) -> float:
    p, a, b = (np.float64(v) for v in (p, a, b))
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def polyline_dist(p, pts) -> float:
    if len(pts) == 1:
        return float(np.hypot(p[0] - pts[0][0], p[1] - pts[0][1]))
    return min(seg_point_dist(p, pts[i], pts[i + 1]) for i in range(len(pts) - 1))


class Geo:
    """Full-res world px ↔ render px ↔ grid (row, col)."""

    def __init__(self, world, calib):
        self.w, self.c, self.s = world, calib, world.cfg.scale
        p0 = np.float64(calib.grid_to_pixel(50, 50))
        p1 = np.float64(calib.grid_to_pixel(50, 60))
        self.px_per_cell_full = float(np.linalg.norm(p1 - p0)) / 10 / self.s

    def to_grid(self, x, y):
        return self.c.pixel_to_grid(x * self.s, y * self.s)

    def to_cell(self, x, y):
        r, c = self.to_grid(x, y)
        return int(round(r)), int(round(c))

    def to_full(self, row, col):
        px, py = self.c.grid_to_pixel(row, col)
        return px / self.s, py / self.s

    def poly_cells_mask(self, poly_full) -> np.ndarray:
        xy = self.c.pixels_to_grid_xy(np.float32(poly_full) * self.s)
        m = np.zeros((GRID_ROWS, GRID_COLS), np.uint8)
        cv2.fillPoly(m, [np.round(xy).astype(np.int32)], 1)
        return m


_FONT = None


def caption(img: np.ndarray, text: str) -> np.ndarray:
    """Label at the top-left (PIL: OpenCV fonts have no accented letters)."""
    global _FONT
    from PIL import Image, ImageDraw, ImageFont
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        except OSError:
            _FONT = ImageFont.load_default()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    x0, y0, x1, y1 = d.textbbox((12, 10), text, font=_FONT)
    d.rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), fill=(20, 24, 33))
    d.text((12, 10), text, font=_FONT, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def person_visual_hull(w: Walker, world) -> np.ndarray:
    """Everything of a person the camera sees (full-res px): shadow, shoes and,
    shifted outward by perspective, arms, shoulders and head — what the
    detectors can react to (the floor footprint alone misses the parallax)."""
    Hc = CAMERA_HEIGHT_MM * PX_PER_MM
    nx, ny = world.W / 2, world.H / 2
    fx, fy = math.cos(w.heading), math.sin(w.heading)
    lx, ly = -fy, fx
    pts = [cv2.ellipse2Poly((int(w.x), int(w.y)), (330, 330), 0, 0, 360, 30).astype(np.float32)]
    pts += list(w.shoes())
    for along, side, z, rad in ((0, 0, 1350, 260), (0, 240, 1150, 180), (0, -240, 1150, 180),
                                (25, 0, 1650, 130)):
        px = w.x + (fx * along + lx * side) * PX_PER_MM
        py = w.y + (fy * along + ly * side) * PX_PER_MM
        k = Hc / (Hc - z * PX_PER_MM)
        qx, qy = nx + (px - nx) * k, ny + (py - ny) * k
        pts.append(cv2.ellipse2Poly((int(qx), int(qy)), (int(rad * k), int(rad * k)),
                                    0, 0, 360, 30).astype(np.float32))
    return cv2.convexHull(np.vstack(pts)).reshape(-1, 2)


def person_footprint(w: Walker) -> np.ndarray:
    """What of a person is on the floor / blocks it seen from above: body
    ellipse (shoulders ≈ 480 × 280 mm) plus the shoes."""
    pts = cv2.ellipse2Poly((int(w.x), int(w.y)), (int(240 * PX_PER_MM), int(140 * PX_PER_MM)),
                           int(math.degrees(w.heading)) + 90, 0, 360, 30).astype(np.float32)
    allp = np.vstack([pts] + [s for s in w.shoes()])
    return cv2.convexHull(allp).reshape(-1, 2)


# ── One test ──────────────────────────────────────────────────────────────────

class SetupFailed(Exception):
    pass


class Test:
    def __init__(self, scenario: int, seed: int, args, record: Path = None,
                 shots_dir: Path = None):
        self.scn, self.seed, self.args = scenario, seed, args
        self.record, self.shots_dir = record, shots_dir
        self.rng = np.random.default_rng(seed + 7)
        self.logs = []
        self.workdir = Path(tempfile.mkdtemp(prefix="scn_"))

    # ── wiring (as MainWindow does in virtual mode) ──
    def build(self):
        cfg = TwinConfig(n_carts=0, n_obstacles=0, scale=self.args.scale, seed=self.seed)
        self.world = w = ScenarioWorld(cfg)
        n = 3 if self.scn == 4 else 1
        for i in range(n):
            w.add_cart(i + 1)
        self.bt = VirtualBTController(w)
        self.grid = Grid(GRID_ROWS, GRID_COLS)
        self.calib = FloorCalibration(self.workdir / "calibration.json")
        calibrate_to_world(w, self.calib, self.grid)
        od = ObstacleDetector(self.workdir / "floor_background.npz")
        self.pdet = SimPersonDetector(ROOT / "yolo11n.pt")
        self.eng = eng = pe.ProdEngine(self.grid, RoutineManager(None), self.bt, CartDetector(),
                                       self.calib, person_detector=self.pdet,
                                       obstacle_detector=od)
        self.carts = {}
        for mid in sorted(w.carts):
            cart = Cart.make(f"V{mid}", CART_COLORS[(mid - 1) % 3], 0, 0)
            cart.marker_id = mid
            eng.add_cart(cart)
            self.bt.assign(cart.id, self.bt.device_for_marker(mid))
            self.bt.connect_cart(cart.id)
            self.carts[mid] = cart
        eng.log_message.connect(self._on_log)
        self.geo = Geo(w, self.calib)
        self.dt = 1.0 / cfg.fps
        self.phase = "setup"
        self.false_stops, self.stops = [], []
        self.writer = None
        self.canvas = None
        self.shots_taken = set()

    def _on_log(self, m: str) -> None:
        self.logs.append((self.world.t, m))
        if self.phase == "run" and "stopped —" in m:
            reason = m.split("stopped — ")[1].split(" on path")[0]
            name = m.split()[1]
            self.stops.append((self.world.t, name, reason))
            if reason in ("person", "moving object"):
                mid = int(name[1:])
                if not self._real_mover_near(mid):
                    self.false_stops.append((self.world.t, name, reason))
                    self.logs.append((self.world.t, f"[test] FALSE STOP {name} ({reason}) — "
                                                    f"no real person / moving object near"))

    def _real_mover_near(self, mid: int) -> bool:
        """Is the stop caused by something real? The moving things the engine
        sees near the cart (moving-object blobs incl. their swept area, YOLO
        person boxes) are matched against the real people / moving objects:
        the stop is genuine if any of them is real, false if all are phantoms."""
        w, s = self.world, self.world.cfg.scale
        c = w.carts[mid]
        rc = self.eng.cart_radius_cells
        look = max(pe.PERSON_LOOKAHEAD, 2 * rc) + rc + max(pe.PERSON_MARGIN, math.ceil(rc)) + 4
        R = look * self.geo.px_per_cell_full                       # full-res px
        seen = [ob.contour / s for ob in self.eng.obstacles if ob.state == pe.OB_MOVING]
        seen += [ob.swept / s for ob in self.eng.obstacles
                 if ob.state == pe.OB_MOVING and ob.swept is not None]
        seen += [np.float32([[p.x1, p.y1], [p.x2, p.y1], [p.x2, p.y2], [p.x1, p.y2]]) / s
                 for p in self.eng.person_detections]
        near = [q for q in seen if min(poly_dist(q, (c.x, c.y)), *(
            math.dist(v, (c.x, c.y)) for v in q)) < R]
        real = [person_visual_hull(wk, w) for wk in w.walkers] + [m.ob.polygon for m in w.movers]
        tol = 80.0                                                  # px
        for q in near:
            for rp in real:
                if min(min(poly_dist(rp, v) for v in q), min(poly_dist(q, v) for v in rp)) < tol:
                    return True
        return False

    # ── simulation tick ──
    def tick(self):
        w = self.world
        w.step(self.dt)
        _Clock.t = w.t
        self.frame = w.render()
        self.eng.on_frame(bgr_to_qimage(self.frame))
        self.fno = getattr(self, "fno", 0) + 1
        if self.fno % 300 == 0:
            # Heartbeat (every 10 s simulated): lets a watcher spot hangs / long waits
            if not hasattr(self, "_wall0"):
                self._wall0 = time.perf_counter()
            states = " ".join(f"{c.name}:{c.state.value}" for c in self.carts.values())
            last = self.logs[-1][1][:70] if self.logs else ""
            print(f"  · cenário {self.scn} seed {self.seed} {self.phase:5s} sim {w.t:6.1f} s "
                  f"wall {time.perf_counter() - self._wall0:6.0f} s  {states}  | {last}",
                  flush=True)

    def log(self, m):
        self.logs.append((self.world.t, f"[test] {m}"))

    # ── set-up: floor capture, heading calibration ──
    def setup(self):
        eng, w = self.eng, self.world
        for _ in range(3):
            self.tick()
        eng.capture_floor()
        while not eng.floor_captured:
            self.tick()
            if w.t > 20:
                raise SetupFailed("floor capture")
        done, retry_at, t0 = set(), 0.0, w.t
        while len(done) < len(self.carts) or eng._calibrating:
            self.tick()
            # Done first: otherwise a finished cart is re-calibrated in the
            # same tick it finished
            for cart in self.carts.values():
                if cart.id not in done and cart.id not in eng._calibrating and \
                        any(f"✓ {cart.name}: heading offset" in m for _, m in self.logs):
                    done.add(cart.id)
            if not eng._calibrating and w.t >= retry_at:
                retry_at = w.t + 3.0
                for cart in self.carts.values():
                    if cart.id not in done:
                        eng.calibrate_heading(cart.id, quiet=True)
                        break                      # one at a time
            if w.t - t0 > CAL_TIMEOUT_S:
                raise SetupFailed("heading calibration")
        bad = [c.name for c in self.carts.values()
               if abs(((c.heading_offset - w.carts[c.marker_id].marker_offset) + 180) % 360 - 180) > 5]
        if bad:
            raise SetupFailed(f"heading calibration wrong for {bad}")
        for _ in range(15):
            self.tick()

    # ── layout: routes, static objects, mover ──
    def layout(self):
        w, rng = self.world, self.rng
        r = w.cart_radius
        fx0, fy0, fx1, fy1 = w.floor_rect
        fw = fx1 - fx0
        routes = {}
        taken = [(c.x, c.y) for c in w.carts.values()]
        for mid, c in w.carts.items():
            for _ in range(400):
                w1 = w.free_point(2.0 * r, taken)
                w2 = w.free_point(2.0 * r, taken + ([w1] if w1 else []))
                if w1 is None or w2 is None:
                    continue
                d1 = math.hypot(w1[0] - c.x, w1[1] - c.y)
                d2 = math.hypot(w2[0] - w1[0], w2[1] - w1[1])
                lo = 0.35 if self.scn != 4 else 0.25
                if d1 > lo * fw and d2 > lo * fw:
                    routes[mid] = [(c.x, c.y), w1, w2]
                    taken += [w1, w2]
                    break
            else:
                raise SetupFailed("no route")
        self.routes = routes
        self.mover_track = None
        if self.scn == 2:
            self._place_mover()
        wps = [p for rt in routes.values() for p in rt]
        mids = list(routes)
        placed = 0
        for k in range(600):
            if placed == N_STATIC:
                break
            rt = routes[mids[placed % len(mids)]] if k < 400 else None
            if rt is not None:
                # Spread along the route: one object per third of its length
                # (fleet: one per cart's route, anywhere along it)
                seg = [math.dist(rt[i], rt[i + 1]) for i in range(len(rt) - 1)]
                if len(mids) == 1:
                    s = sum(seg) * (placed + rng.uniform(0.15, 0.85)) / N_STATIC
                else:
                    s = sum(seg) * rng.uniform(0.2, 0.8)
                i = 0
                while i < len(seg) - 1 and s > seg[i]:
                    s -= seg[i]
                    i += 1
                a, b = np.float64(rt[i]), np.float64(rt[i + 1])
                p = a + min(s / max(seg[i], 1e-6), 1.0) * (b - a)
                nrm = np.array([-(b - a)[1], (b - a)[0]]) / max(np.linalg.norm(b - a), 1)
                p = p + nrm * rng.normal(0, 0.8 * r)
            else:
                p = np.array([rng.uniform(fx0 + 80, fx1 - 80), rng.uniform(fy0 + 80, fy1 - 80)])
            n0 = len(w.obstacles)
            if not w.place_obstacle(int(rng.integers(w.n_obstacle_kinds)), float(p[0]), float(p[1]),
                                    float(rng.uniform(-180, 180))):
                continue
            ob = w.obstacles[-1]
            if len(w.obstacles) == n0 or min(poly_dist(ob.polygon, q) for q in wps) < 2.6 * r or \
                    (self.mover_track is not None and min(
                        seg_point_dist(v, *self.mover_track[:2]) for v in ob.polygon[::3])
                     < self.mover_track[2] + 60):
                w.obstacles.pop()
                w._rebuild_occupancy()
                continue
            placed += 1
        if placed < N_STATIC:
            raise SetupFailed("static objects")
        self.statics = [ob for ob in w.obstacles if all(m.ob is not ob for m in w.movers)]

    def _place_mover(self):
        w, rng, r = self.world, self.rng, self.world.cart_radius
        fx0, fy0, fx1, fy1 = w.floor_rect
        rt = self.routes[1]
        wps = np.float32(rt[1:])
        carts = np.float32([(c.x, c.y) for c in w.carts.values()])
        # Distance to the static objects (1/4 res), for the sweep check
        q4 = cv2.resize(w._occ, (w.W // 4, w.H // 4), interpolation=cv2.INTER_NEAREST)
        dist_static = cv2.distanceTransform((q4 == 0).astype(np.uint8), cv2.DIST_L2, 3) * 4
        for _ in range(30):
            # Outline of one sprite / angle, rendered once and translated below
            idx, angle = int(rng.integers(w.n_obstacle_kinds)), float(rng.uniform(-180, 180))
            cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
            n0 = len(w.obstacles)
            w.place_obstacle(idx, cx, cy, angle, check=False)
            base = w.obstacles[-1].polygon - np.float32([cx, cy])
            del w.obstacles[n0:]
            for _ in range(80):
                i = int(rng.integers(len(rt) - 1))
                a, b = np.float64(rt[i]), np.float64(rt[i + 1])
                P = a + rng.uniform(0.3, 0.7) * (b - a)
                ang = math.atan2(b[1] - a[1], b[0] - a[0]) + math.pi / 2 + rng.uniform(-0.6, 0.6)
                u = np.array([math.cos(ang), math.sin(ang)])
                L = rng.uniform(220, 450)
                A, B = P - u * L, P + u * L
                ok = True
                for t in np.linspace(0, 1, 15):
                    q = base + np.float32(A + t * (B - A))
                    if q[:, 0].min() < fx0 + 15 or q[:, 1].min() < fy0 + 15 or \
                            q[:, 0].max() > fx1 - 25 or q[:, 1].max() > fy1 - 25:
                        ok = False
                        break
                    xi = np.clip((q[:, 0] / 4).astype(int), 0, dist_static.shape[1] - 1)
                    yi = np.clip((q[:, 1] / 4).astype(int), 0, dist_static.shape[0] - 1)
                    if dist_static[yi, xi].min() < 60:
                        ok = False
                        break
                    if min(poly_dist(q, v) for v in carts) < 1.2 * r or \
                            min(poly_dist(q, v) for v in wps) < 1.2 * r:
                        ok = False
                        break
                if not ok:
                    continue
                S = A + rng.uniform(0, 1) * (B - A)
                if min(poly_dist(base + np.float32(S), v) for v in carts) < 2.2 * r:
                    S = A if poly_dist(base + np.float32(A), carts[0]) > \
                        poly_dist(base + np.float32(B), carts[0]) else B
                w.place_obstacle(idx, float(S[0]), float(S[1]), angle, check=False)
                ob = w.obstacles[-1]
                mv = Mover(ob, A, B, float(rng.uniform(120, 220)), (1.0, 3.5), rng, w.cfg.scale)
                mv.to_b = bool(rng.random() < 0.5)
                w.movers.append(mv)
                w._rebuild_occupancy()
                self.mover_track = (A, B, float(np.linalg.norm(base, axis=1).max()))
                self.log(f"moving object {L * 2:.0f} px track, {mv.speed:.0f} px/s")
                return
        raise SetupFailed("moving object")

    # ── people (scenario 3) ──
    def spawn_person(self, mid: int) -> bool:
        """Someone crossing (or stopping on) the cart's route ahead of it."""
        w, eng, rng = self.world, self.eng, self.rng
        cart = self.carts[mid]
        path = eng._paths.get(cart.id)
        c = w.carts[mid]
        r = w.cart_radius
        kind = "visit" if rng.random() < 0.3 else "pass"
        ahead = None
        if path:
            idx = eng._path_idx.get(cart.id, 0)
            pts = [(c.x, c.y)] + [self.geo.to_full(*n) for n in path[idx:]]
            want, acc = rng.uniform(4.5 * r, 9 * r), 0.0
            for a, b in zip(pts, pts[1:]):
                seg = math.hypot(b[0] - a[0], b[1] - a[1])
                if acc + seg >= want:
                    t = (want - acc) / max(seg, 1e-6)
                    ahead = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                    break
                acc += seg
            if ahead is None:
                ahead = pts[-1]
        speed = float(rng.uniform(1000, 1400)) * PX_PER_MM
        for _ in range(150):
            entry = w._border_point()
            exit_ = w._border_point(not_side_of=entry)
            if ahead is not None:
                P = (ahead[0] + rng.normal(0, 0.4 * r), ahead[1] + rng.normal(0, 0.4 * r))
                route = [entry, P, exit_]
            else:
                route = [entry, exit_]
            if not w._walk_clear(route):
                continue
            wk = Walker(route, speed, 1 if (kind == "visit" and ahead is not None) else None,
                        float(rng.uniform(2.0, 5.0)), None,
                        trousers=TROUSERS[int(rng.integers(len(TROUSERS)))],
                        shirt=SHIRTS[int(rng.integers(len(SHIRTS)))],
                        skin=SKIN[int(rng.integers(len(SKIN)))],
                        hair=HAIR[int(rng.integers(len(HAIR)))])
            w.walkers.append(wk)
            w.walker_log.append((w.t, f"person enters ({kind})"))
            return True
        return w.spawn_walker("pass")

    # ── scoring helpers ──
    def _things_on_floor(self):
        """(key, kind, footprint) of every real object / person on the floor."""
        w = self.world
        fx0, fy0, fx1, fy1 = w.floor_rect
        out = [(id(ob), "objeto móvel" if any(m.ob is ob for m in w.movers) else "objeto estático",
                ob.polygon) for ob in w.obstacles]
        for wk in w.walkers:
            if fx0 + 100 < wk.x < fx1 - 100 and fy0 + 100 < wk.y < fy1 - 100:
                out.append((id(wk), "pessoa", person_footprint(wk)))
        return out

    def check_representation(self, st):
        occ = self.eng.occupancy_matrix()
        things = np.isin(occ, OCC_THINGS)
        t = self.world.t
        present = set()
        for key, kind, poly in self._things_on_floor():
            present.add(key)
            rec = st.setdefault(key, {"since": t, "miss_since": None, "kind": kind,
                                      "checks": 0, "hits": 0})
            m = self.geo.poly_cells_mask(poly).astype(bool)
            ok = m.sum() == 0 or (things & m).sum() >= REPR_COVER * m.sum()
            if t - rec["since"] < APPEAR_GRACE_S:
                continue
            rec["checks"] += 1
            rec["hits"] += int(ok)
            if ok:
                rec["miss_since"] = None
            else:
                if rec["miss_since"] is None:
                    rec["miss_since"] = t
                if t - rec["miss_since"] >= MISS_FAIL_S and not rec.get("failed"):
                    rec["failed"] = True
                    self.repr_fail.append((t, kind))
                    self.log(f"REPRESENTATION FAIL: {kind} missing from the matrix "
                             f"for {t - rec['miss_since']:.1f} s")
        for key in list(st):
            if key not in present:
                # Gone (left / picked up): a later re-entry starts over
                del st[key]

    def check_trajectory(self, offp):
        eng, w = self.eng, self.world
        rc = eng.cart_radius_cells
        for mid, cart in self.carts.items():
            path = eng._paths.get(cart.id)
            if not path:
                offp[mid] = 0
                continue
            c = w.carts[mid]
            row, col = self.geo.to_grid(c.x, c.y)
            # The engine stores a path without its start cell (where the cart
            # was when it was planned): put it back, it is part of the route
            key = self._path_start.get(mid)
            if key is None or key[0] is not path:
                start = eng._detected_grid_pos(cart.id) or (row, col)
                key = self._path_start[mid] = (path, start)
            d = polyline_dist((row, col), [key[1]] + list(path))
            self.max_dev[mid] = max(self.max_dev.get(mid, 0.0), d / rc)
            if d > rc:
                offp[mid] = offp.get(mid, 0) + 1
                if offp[mid] == OFFPATH_FRAMES:
                    self.traj_fail.append((w.t, cart.name, d / rc))
                    self.log(f"TRAJECTORY FAIL: {cart.name} {d:.1f} cells from its route "
                             f"(blue radius {rc:.1f})")
            else:
                offp[mid] = 0

    # ── rendering of the app views (screenshots / video) ──
    def app_view(self) -> np.ndarray:
        from PyQt6.QtGui import QImage
        from ui.production.occupancy_canvas import OccupancyCanvas
        from ui.production.prod_canvas import ProdCanvas
        eng = self.eng
        if self.canvas is None:
            self.canvas = ProdCanvas(self.calib)
            self.canvas.set_grid_size(GRID_ROWS, GRID_COLS)
            self.canvas.resize(960, 540)
            self.occ_canvas = OccupancyCanvas()
            self.occ_canvas.resize(960, 540)
        self.canvas.on_frame(bgr_to_qimage(self.frame))
        self.canvas.update_state(eng.carts, eng.detections, paths=dict(eng._paths),
                                 person_detections=eng.person_detections,
                                 obstacles=eng.obstacles)
        self.occ_canvas.update_state(eng.occupancy_matrix(), eng.carts, eng.cart_grid_poses(),
                                     eng.paths, eng.cart_radius_cells, eng.occupancy_notice())
        imgs = []
        for wdg in (self.canvas, self.occ_canvas):
            q = wdg.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
            a = np.frombuffer(q.constBits().asstring(q.sizeInBytes()), np.uint8)
            a = a.reshape(q.height(), q.bytesPerLine())[:, :q.width() * 3]
            imgs.append(cv2.cvtColor(a.reshape(q.height(), q.width(), 3), cv2.COLOR_RGB2BGR))
        h = max(i.shape[0] for i in imgs)
        imgs = [cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 0, cv2.BORDER_CONSTANT) for i in imgs]
        view = np.hstack([imgs[0], np.full((h, 4, 3), 40, np.uint8), imgs[1]])
        t_rel = self.world.t - getattr(self, "t_start", self.world.t)
        return caption(view, f"Cenário {SCENARIOS[self.scn]}   t = {t_rel:5.1f} s")

    def shot(self, name):
        if self.shots_dir is None or name in self.shots_taken:
            return
        self.shots_taken.add(name)
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.shots_dir / f"cenario{self.scn}_{name}.png"), self.app_view())
        cv2.imwrite(str(self.shots_dir / f"cenario{self.scn}_{name}_camera.png"), self.frame)

    # ── the test ──
    def run(self) -> dict:
        self.build()
        self.setup()
        self.layout()
        eng, w = self.eng, self.world
        # Let the detector confirm the objects, then check the routes are
        # plannable (as the routine editor's preview does for an operator)
        for _ in range(int(1.5 / self.dt)):
            self.tick()
        cells = {}
        for mid, rt in self.routes.items():
            wp = [self.geo.to_cell(*p) for p in rt[1:]]
            path, bad, start = eng.preview_route(self.carts[mid].id, wp)
            if bad is not None or start is None:
                raise SetupFailed(f"route V{mid} not plannable")
            cells[mid] = wp
        self.phase = "run"
        self.t_start = w.t
        self.repr_fail, self.traj_fail, self.max_dev = [], [], {}
        self._path_start = {}
        n_coll0 = len(w.collision_log)
        for mid, wp in cells.items():
            rt = Routine.make(self.carts[mid].id, f"test {self.seed} V{mid}", SCHEDULE_TIME)
            rt.waypoints = wp
            eng.trigger_routine(rt)
            if self.carts[mid].id not in eng._paths:
                raise SetupFailed(f"routine V{mid} refused")
        self.log("routes started: " + "; ".join(f"V{m} {c}" for m, c in cells.items()))
        reached = {mid: False for mid in self.carts}
        next_person = w.t + self.rng.uniform(1.0, 3.0)
        repr_st, offp = {}, {}
        timeout = ROUTE_TIMEOUT_S[self.scn]
        k = 0
        shot_event_at = None
        while True:
            self.tick()
            k += 1
            for mid, cart in self.carts.items():
                if not reached[mid] and cart.id not in eng._paths and not cart.waypoints:
                    reached[mid] = True
                    self.log(f"{cart.name} route complete in {w.t - self.t_start:.1f} s")
            if all(reached.values()):
                break
            if w.t - self.t_start > timeout:
                self.log("route TIMEOUT")
                break
            if self.scn == 3 and w.t >= next_person:
                active = [m for m in self.carts if not reached[m]]
                self.spawn_person(active[0])
                next_person = w.t + self.rng.uniform(6.0, 12.0)
            self.check_trajectory(offp)
            if k % 3 == 0:
                self.check_representation(repr_st)
            # Screenshots: shortly after the start, and while something is in the way
            if self.shots_dir is not None:
                if w.t - self.t_start >= 2.0:
                    self.shot("inicio")
                want = ("moving object",) if self.scn == 2 else ("person", "moving object") \
                    if self.scn == 3 else ("obstacle", "cart")
                if shot_event_at is None and any(s_[2] in want for s_ in self.stops):
                    shot_event_at = w.t + 0.3
                if shot_event_at is not None and w.t >= shot_event_at:
                    self.shot("evento")
                if w.t - self.t_start >= 0.5 * getattr(self.args, "shot_mid", 20.0):
                    self.shot("meio")
            if self.record is not None:
                v = self.app_view()
                if self.writer is None:
                    self.record.parent.mkdir(parents=True, exist_ok=True)
                    self.writer = cv2.VideoWriter(str(self.record), cv2.VideoWriter_fourcc(*"mp4v"),
                                                  w.cfg.fps, (v.shape[1], v.shape[0]))
                self.writer.write(v)
        # A few more frames so the end state is on the video
        if self.record is not None:
            for _ in range(int(1.0 / self.dt)):
                self.tick()
                self.writer.write(self.app_view())
            self.writer.release()
        coll = [(t, mid, what) for t, mid, what in w.collision_log[n_coll0:]
                if what != "person (cart braking)"]
        for t, mid, what in coll:
            self.logs.append((t, f"[test] COLLISION V{mid} with {what}"))
        persons = sum(1 for t, e in w.walker_log if e.startswith("person enters") and t >= self.t_start)
        reasons = {}
        for _, _, rsn in self.stops:
            reasons[rsn] = reasons.get(rsn, 0) + 1
        return {
            "scenario": self.scn, "seed": self.seed,
            "route_ok": all(reached.values()),
            "route_time_s": round(w.t - self.t_start, 1),
            "repr_ok": not self.repr_fail,
            "repr_fail": [k for _, k in self.repr_fail],
            "traj_ok": not self.traj_fail,
            "max_dev_radii": round(max(self.max_dev.values(), default=0.0), 2),
            "collision_free": not coll,
            "collisions": [f"V{m}:{what}" for _, m, what in coll],
            "false_stops": len(self.false_stops),
            "stops": reasons,
            "replans": sum(1 for t, m in self.logs if t >= self.t_start and "replanned" in m),
            "persons": persons,
            "yolo_inferences": self.pdet.inferences,
        }

    def write_log(self, path: Path):
        logs = list(self.logs)
        for t, e in self.world.walker_log:
            logs.append((t, f"[world] {e}"))
        logs.sort(key=lambda e: e[0])
        path.write_text("".join(f"{t:7.1f}s  {m}\n" for t, m in logs))

    def close(self):
        shutil.rmtree(self.workdir, ignore_errors=True)


# ── Runner and summary ────────────────────────────────────────────────────────

FIELDS = ["scenario", "test", "seed", "route_ok", "route_time_s", "repr_ok", "repr_fail",
          "traj_ok", "max_dev_radii", "collision_free", "collisions", "false_stops", "stops",
          "replans", "persons", "setup_retries", "wall_s"]


def run_scenario(args) -> None:
    out = args.out / f"cenario{args.scenario}"
    (out / "logs").mkdir(parents=True, exist_ok=True)
    res_path = out / "runs.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            done.add(json.loads(line)["test"])
    videos = {int(v) for v in args.video_tests.split(",") if v != ""}
    lo, hi = 1, args.runs
    if args.tests:
        lo, hi = (int(v) for v in args.tests.split("-"))
    for i in range(args.runs):
        if i + 1 in done or not lo <= i + 1 <= hi:
            continue
        retries, seed = 0, 1000 * args.scenario + i
        while True:
            t_wall = time.perf_counter()
            test = Test(args.scenario, seed, args,
                        record=(out / f"video_cenario{args.scenario}_teste{i + 1}.mp4") if i in videos else None,
                        shots_dir=(out / "prints") if i == args.shot_test else None)
            try:
                res = test.run()
            except SetupFailed as e:
                test.write_log(out / "logs" / f"setup_fail_seed{seed}.log")
                print(f"[cenário {args.scenario}] test {i + 1}: set-up failed ({e}), new layout",
                      flush=True)
                test.close()
                retries += 1
                seed += 100000
                continue
            except Exception:
                traceback.print_exc()
                test.write_log(out / "logs" / f"crash_seed{seed}.log")
                test.close()
                retries += 1
                seed += 100000
                if retries > 10:
                    raise
                continue
            res.update(test=i + 1, setup_retries=retries,
                       wall_s=round(time.perf_counter() - t_wall))
            test.write_log(out / "logs" / f"teste{i + 1:02d}_seed{seed}.log")
            test.close()
            break
        with res_path.open("a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        print(f"[cenário {args.scenario}] test {i + 1}/{args.runs}: "
              f"route {'ok' if res['route_ok'] else 'NOT done'} {res['route_time_s']} s | "
              f"repr {'ok' if res['repr_ok'] else 'FAIL ' + str(res['repr_fail'])} | "
              f"traj {'ok' if res['traj_ok'] else 'FAIL'} ({res['max_dev_radii']}) | "
              f"coll {'none' if res['collision_free'] else res['collisions']} | "
              f"false stops {res['false_stops']} | stops {res['stops']} | {res['wall_s']} s",
              flush=True)


def summary(base: Path) -> None:
    rows, all_runs = [], []
    for s in sorted(SCENARIOS):
        p = base / f"cenario{s}" / "runs.jsonl"
        if not p.exists():
            continue
        runs = [json.loads(line) for line in p.read_text().splitlines()]
        runs = list({r["test"]: r for r in reversed(runs)}.values())   # first result per test
        runs.sort(key=lambda r: r["test"])
        all_runs += runs
        n = len(runs)
        ok_times = [r["route_time_s"] for r in runs if r["route_ok"]]
        rows.append({
            "Cenário": SCENARIOS[s],
            "Número de testes": n,
            "Representação da cena": sum(r["repr_ok"] for r in runs),
            "Seguimento da trajetória": sum(r["traj_ok"] for r in runs),
            "Navegação sem colisões": sum(r["collision_free"] for r in runs),
            "Paradas por falsos objetos em movimento (total)": sum(r["false_stops"] for r in runs),
            "Testes com parada falsa": sum(r["false_stops"] > 0 for r in runs),
            "Rota concluída": sum(r["route_ok"] for r in runs),
            "Tempo médio da rota (s)": round(float(np.mean(ok_times)), 1) if ok_times else "",
        })
    if not rows:
        print("no results yet")
        return
    cols = list(rows[0])
    with (base / "resultados.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        wr.writerows(rows)
    with (base / "todos_os_testes.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        wr.writeheader()
        for r in all_runs:
            wr.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                         for k, v in r.items()})
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
        wb = Workbook()
        ws = wb.active
        ws.title = "Resultados"
        ws.append(cols)
        for r in rows:
            ws.append([r[c] for c in cols])
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        ws.column_dimensions["A"].width = 44
        for col in "BCDEFGHI":
            ws.column_dimensions[col].width = 18
        ws2 = wb.create_sheet("Testes")
        ws2.append(FIELDS)
        for r in all_runs:
            ws2.append([json.dumps(r.get(k), ensure_ascii=False) if isinstance(r.get(k), (list, dict))
                        else r.get(k) for k in FIELDS])
        wb.save(base / "resultados.xlsx")
    except ImportError:
        pass
    w0 = max(len(c) for c in (r["Cenário"] for r in rows))
    print("\n" + " | ".join(["Cenário".ljust(w0)] + cols[1:]))
    for r in rows:
        print(" | ".join([r["Cenário"].ljust(w0)] + [str(r[c]) for c in cols[1:]]))
    print(f"\n→ {base / 'resultados.csv'}, {base / 'resultados.xlsx'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", type=int, choices=sorted(SCENARIOS))
    ap.add_argument("--runs", type=int, default=25)
    ap.add_argument("--tests", help="only these test numbers, e.g. 14-25 (1-based)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "scenario_tests")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--video-tests", default="0", help="test indices (0-based) to record")
    ap.add_argument("--shot-test", type=int, default=0, help="test index for the screenshots")
    ap.add_argument("--summary", type=Path, help="only build the results table from this folder")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary)
        return
    if args.scenario is None:
        ap.error("--scenario or --summary")
    cv2.setNumThreads(2)
    _app = QApplication.instance() or QApplication(sys.argv)     # noqa: F841
    run_scenario(args)
    summary(args.out)


if __name__ == "__main__":
    main()
