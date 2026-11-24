"""
Production engine — closed-loop cart control.

Driven by camera frames (not a self-ticking timer). Each call to on_frame():
  1. Obstacle detector: compare frame with the captured empty floor → mask
  2. Cart tracker: ArUco + ROI re-search + foreground-blob fallback → poses
  3. Submit frame to YOLO person detector (async, non-blocking)
  4. Convert poses / obstacles / persons to grid coords via homography
  5. For each cart with an active path:
       a. If a person, obstacle or other cart occupies upcoming path cells → stop, wait
       b. If blocked for > BLOCK_REPLAN_SECS → replan around the blocked cells
       c. If the cart is not visible → stop until it is seen again
       d. Otherwise steer toward next waypoint via DRIVE commands
  6. Advance waypoint index when cart arrives within ARRIVAL_CELLS

Routine scheduling runs on a 1-second QTimer.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QImage

from config import EMPTY
from core.cart import Cart, CartState
from core.grid import Grid
from core.routine import Routine
from core.routine_manager import RoutineManager
from pathfinding.theta_star import theta_star
from production.bt_controller import BTController
from production.calibration import FloorCalibration
from production.cart_detector import (
    SRC_MEMORY, SRC_PREDICTED, CartDetector, CartTracker, Detection,
)
from production.obstacle_detector import (
    OB_MOVING, OB_REMEMBERED, Obstacle, ObstacleDetector, ObstacleTracker,
)
from production.person_detector import PersonDetector, PersonDetection

# Movement tunables
DRIVE_SPEED          = 110    # base forward speed (1–255, rescaled to the cart's start PWM..255)
TURN_SPEED           = 40     # rotation-in-place speed (same scale)
HEADING_TOL_DEG      = 12.0   # start driving once heading error is within this...
HEADING_DRIVE_MAX_DEG = 35.0  # ...and (continuous mode) keep driving until it exceeds this
ARRIVAL_CELLS        = 1.8    # grid cells — close enough to consider waypoint reached
STEER_GAIN           = 0.9    # proportional gain for heading correction while driving
# The carts are bidirectional: when the next node is behind the cart, it
# drives backwards instead of spinning 180° first. Whichever end needs less
# turning leads; the leading end only flips when the other one saves more
# than REVERSE_HYST_DEG of turning, so a target near 90° doesn't make it
# flip-flop between directions.
REVERSE_DRIVE        = True
REVERSE_HYST_DEG     = 20.0

# Step-by-step control: the camera reaches the PC 100–200 ms late (Wi-Fi) and
# the motors can't run slow enough to steer continuously on old frames, so a
# cart moves in discrete steps: decide → one timed command → stop → wait until
# the video shows it standing still → measure what the step did → next step.
# What each step achieved is learned per cart (turn rate, forward speed), so
# the next command is sized to what this cart actually does.
STEP_MODE            = True   # False = continuous control (simulation / low latency)
STEP_LATENCY_SECS    = 0.25   # after a stop, ignore frames this long (video delay)
STEP_STILL_FRAMES    = 3      # then: this many consecutive frames without motion...
STEP_STILL_SIDES     = 0.04   # ...(less than this × marker side between frames...
STEP_STILL_DEG       = 1.5    # ...and less than this rotation) = stopped
STEP_SETTLE_MAX_SECS = 1.5    # give up waiting for stillness after this
STEP_MIN_SECS        = 0.06   # shortest command
STEP_TURN_AIM        = 0.8    # turn for this fraction of the error (overshooting costs
                              # a turn back; a little short is fixed by the next step)
STEP_TURN_MAX_SECS   = 0.6    # longest single turn (then look again)...
STEP_TURN_MAX_DEG    = 90.0   # ...aiming at most this far in one step
STEP_FWD_MAX_RADII   = 1.5    # longest single forward step, in cart radii...
STEP_FWD_MAX_SECS    = 0.8    # ...and in seconds
STEP_TURN_RATE_INIT  = 240.0  # deg/s guess before anything is measured — high on
STEP_FWD_RADII_PER_S_INIT = 3.0   # purpose: too-short first steps are safe
STEP_TURN_RATE_MIN   = 20.0
STEP_TURN_RATE_MAX   = 900.0
STEP_FWD_RATE_MIN    = 0.3    # cells/s
STEP_LEARN           = 0.5    # weight of the newest measurement in the learned rates
STEP_NO_MOVE_DEG     = 2.0    # less rotation than this = the turn didn't happen
STEP_NO_MOVE_SHRINK  = 0.6    # didn't move → assume slower → next command longer
STEP_STUCK_STEPS     = 4      # this many forward steps in a row without progress = stuck

# Safety tunables
PERSON_LOOKAHEAD     = 12     # grid cells of upcoming path to check for blockers
PERSON_MARGIN        = 2      # grid-cell safety margin around each detected person
OBSTACLE_MARGIN      = 2      # grid-cell safety margin around detected obstacles
PLAN_EXTRA_MARGIN    = 1      # extra clearance when planning, so mask flicker at
                              # the edge doesn't stop a cart on a fresh path
PLAN_CLEARANCE_RADII = 0.75   # plus a comfortable gap (× cart radius) from static
                              # obstacles where the space allows; dropped for a leg
                              # that only fits tightly (narrow passages)
STATIC_HIT_FRAMES    = 3      # a static obstacle / wall must be on the path this many
                              # frames in a row to stop the cart (edge flicker filter);
                              # people, movers and carts stop it at once
CART_RADIUS_CELLS    = 2      # default cart radius until a marker has been measured
CART_BODY_RADIUS_SIDES = 1.5  # cart body + wheels half-diagonal, in marker sides
# The margins above are minimums: the measured cart radius (from the marker
# size and the calibration) takes over when larger, so safety margins match
# the real cart whatever the grid's cell size is.
BLOCK_REPLAN_SECS    = 3.0    # seconds of blocking before trying to replan around it
START_CLEAR_CELLS    = 3      # ignore blockers this close to a cart when planning
RESUME_CLEAR_FRAMES  = 10     # path must stay clear this many frames before resuming

# Stuck detection: driving forward without getting anywhere
STUCK_SECS           = 2.0    # forward driving this long...
STUCK_MIN_RADII      = 0.3    # ...with less progress than this (× cart radius) = stuck
STUCK_MEMORY_SECS    = 15.0   # how long the unseen blockage is remembered

YIELD_COOLDOWN_SECS  = 10.0   # a cart gives way at most this often

# Heading calibration: short forward drive to measure marker vs travel direction
CAL_SPEED            = 255    # full power: mismatched motors curve less, so the
                              # measured direction is cleaner
CAL_TARGET_SIDES     = 1.0    # drive until the marker has moved this far (marker sides)...
CAL_MAX_SECS         = 5.0    # ...or this long — motor latency and ramp-up make a
                              # fixed drive time unreliable
CAL_MIN_MOVE_SIDES   = 0.5    # displacement needed for a valid measurement
CAL_SPIN_DEG         = 90.0   # rotation before moving a marker side = spinning in place
CAL_PAUSE_SECS       = 0.6    # rest between calibration phases
CAL_TURN_SPEED       = 255    # turn-direction check: a short left-turn command…
CAL_TURN_DEG         = 30.0   # …until it has rotated this much…
CAL_TURN_SECS        = 1.5    # …or for this long
CAL_TURN_MIN_DEG     = 8.0    # less rotation than this = inconclusive
# Start-PWM measurement (last two calibration phases, forward then turn in
# place): bisection with short RAW probes (exact PWM, no firmware breakaway
# boost), each from rest: push for START_PROBE_SECS, stop, let the delayed
# video show whether it moved, halve the range. A motor that is powered but
# not turning is stalled — maximum current, the worst electrical noise for the
# cart's Bluetooth — so it is never held there for more than one short probe
# (the old slow ramp could stall the motors for seconds). The result becomes
# Cart.min_pwm_drive / min_pwm_turn: planner speeds 1..255 are rescaled to
# min..255, so a cart with other motors, driver, battery or load still moves.
START_PWM_FROM       = 60     # lowest PWM considered (assumed not to start)
START_PWM_RESOLUTION = 8      # stop bisecting when the range is this narrow
START_PROBE_SECS     = 0.30   # one probe's push (the firmware soft start takes 60 ms)
START_SETTLE_SECS    = 0.50   # motors off after a probe: video delay + coasting, and
                              # rest before the next probe
START_MOVE_SIDES     = 0.15   # marker motion (marker sides) = it moved
START_TURN_DEG       = 5.0    # rotation = it turned
START_PWM_MARGIN     = 10     # added to the measured PWM: battery sag, floor patches
# Blind motor test (🧭 on a cart the camera can't see): the calibration's motor
# commands without measuring anything — checks power / link under load
MOTOR_TEST_STEPS     = [("forward", 2.0, (255, 255)),
                        ("pause",   0.6, (0, 0)),
                        ("left turn", 1.5, (-CAL_TURN_SPEED, CAL_TURN_SPEED))]
MOTOR_TEST_TICK_MS   = 50
MOTOR_TEST_VERIFY_S  = 8.0    # after the test: wait this long for a fresh reply
OFFSET_SNAP_DEG      = 20.0   # markers are mounted square to the cart: a measured
                              # offset this close to a multiple of 90° is snapped to it
CAL_FREE_RADII       = 1.5    # room the cart's centre needs in every direction
                              # (× cart radius; obstacle cells already include the body)

# Heading check while driving: a wrong marker offset (calibration skipped or
# failed) makes a cart orbit its target instead of reaching it
HEADING_CHECK_SIDES  = 1.5    # straight forward travel (marker sides) per check
HEADING_FIX_DEG      = 30.0   # offset error that counts as wrong mounting, not drift

FLOOR_CAPTURE_FRAMES = 20    # frames averaged into the empty-floor background
FLOOR_CAPTURE_EVERY  = 2      # use every Nth camera frame while capturing
CART_FOOTPRINT_SCALE = 2.6    # cart body ≈ marker size × this

# Occupancy matrix cell codes (occupancy_matrix()); later codes win on overlap
OCC_FREE     = 0
OCC_WALL     = 1
OCC_MARGIN   = 2    # free, but too close to a wall for the cart body
OCC_OBSTACLE = 3    # static obstacle + safety margin
OCC_MOVER    = 4    # moving object + swept area + margin
OCC_PERSON   = 5    # YOLO person + margin
OCC_CART     = 6    # cart body


class ProdEngine(QObject):
    state_updated = pyqtSignal()    # emitted each camera frame (canvas repaint)
    log_message   = pyqtSignal(str)
    calibration_invalidated = pyqtSignal()   # camera resolution changed the mapping

    def __init__(self,
                 grid:          Grid,
                 rm:            RoutineManager,
                 bt:            BTController,
                 detector:      CartDetector,
                 calibration:   FloorCalibration,
                 person_detector: Optional[PersonDetector] = None,
                 obstacle_detector: Optional[ObstacleDetector] = None,
                 parent=None):
        super().__init__(parent)
        self._grid            = grid
        self._rm              = rm
        self._bt              = bt
        self._tracker         = CartTracker(detector)
        self._calib           = calibration
        self._person_detector = person_detector
        self._obstacle_detector = obstacle_detector

        self._carts:      Dict[str, Cart]                   = {}
        self._marker_map: Dict[int, str]                    = {}   # marker_id → cart_id
        self._paths:      Dict[str, List[Tuple[int, int]]]  = {}
        self._path_idx:   Dict[str, int]                    = {}
        self._detections: List[Detection]                   = []   # tracked carts + other markers
        self._person_dets: List[PersonDetection]            = []   # YOLO results
        self._person_cells: Set[Tuple[int, int]]            = set()  # grid cells blocked by persons
        self._obstacles: List[Obstacle]                     = []   # camera obstacles (pixels)
        self._obstacle_cells: Set[Tuple[int, int]]          = set()  # grid cells blocked by obstacles
        self._obstacle_clear_band: Set[Tuple[int, int]]     = set()  # planning-only extra gap
        self._static_hits: Dict[str, int]                   = {}     # flicker filter (frames)
        self._obstacle_plan_cells: Set[Tuple[int, int]]     = set()  # same, wider margin for planning
        self._mover_cells: Set[Tuple[int, int]]             = set()  # people / moving objects
        self._block_since: Dict[str, datetime]              = {}   # cart_id → when blocking started
        self._clear_frames: Dict[str, int]                  = {}   # cart_id → clear frames while blocked
        self._lost: Set[str]                                = set()  # carts stopped for not being seen
        self._floor_capture_left = 0
        self._floor_roi_key: Optional[tuple] = None
        self._grid_bounds: Optional[Tuple[int, int, int, int]] = None
        self._floor_poly: Optional[np.ndarray] = None
        self._obstacle_tracker = ObstacleTracker()
        self._wall_cache: Optional[Tuple[int, Set[Tuple[int, int]]]] = None
        self._wall_cache_mask: Optional[np.ndarray] = None
        self._static_dist_cache: Optional[np.ndarray] = None
        self._static_dist_frame = -1
        self._frame_counter = 0
        self._detection_blind = False      # obstacle detection lost: carts held
        self._cart_radius_cells: float = CART_RADIUS_CELLS
        self._progress: Dict[str, Tuple[datetime, float, float]] = {}  # stuck watch
        self._stuck_cells: Dict[str, Tuple[datetime, Set[Tuple[int, int]]]] = {}
        self._calibrating: Dict[str, dict] = {}                        # heading calibration
        self._heading_watch: Dict[str, dict] = {}                      # running offset estimate
        self._steer: Dict[str, dict] = {}                              # continuous: turn/drive mode
        self._step: Dict[str, dict] = {}                               # step control: current step
        self._step_rates: Dict[str, dict] = {}                         # step control: learned rates
        self._reverse: Dict[str, bool] = {}                            # driving backwards (bidirectional)
        self.calibration_refusal: Dict[str, str] = {}                  # why 🧭 couldn't start
        self._yielding: Set[str] = set()                               # moving aside for another cart
        self._wait_reason: Dict[str, str] = {}                         # last logged wait reason
        self._last_yield: Dict[str, datetime] = {}
        self._radius_measured = False

        # Routine scheduling timer (~1 Hz)
        self._sched_timer = QTimer(self)
        self._sched_timer.setInterval(1000)
        self._sched_timer.timeout.connect(self._check_routines)
        self._sched_timer.start()

        # Blind motor tests (no camera needed): own timer, so commands keep
        # flowing faster than the cart's 500 ms command watchdog
        self._motor_tests: Dict[str, dict] = {}
        self._test_timer = QTimer(self)
        self._test_timer.setInterval(MOTOR_TEST_TICK_MS)
        self._test_timer.timeout.connect(self._motor_test_tick)

    # ── Cart management ───────────────────────────────────────────────────────

    def add_cart(self, cart: Cart) -> None:
        self._carts[cart.id] = cart
        if cart.marker_id is not None:
            self._marker_map[cart.marker_id] = cart.id

    def remove_cart(self, cart_id: str) -> None:
        self._motor_tests.pop(cart_id, None)
        cart = self._carts.pop(cart_id, None)
        if cart and cart.marker_id is not None:
            self._marker_map.pop(cart.marker_id, None)
        self._paths.pop(cart_id, None)
        self._path_idx.pop(cart_id, None)
        self._block_since.pop(cart_id, None)
        self._lost.discard(cart_id)
        for d in (self._progress, self._stuck_cells, self._calibrating, self._heading_watch,
                  self._static_hits, self._steer, self._step, self._step_rates, self._last_yield, self._wait_reason, self._clear_frames,
                  self._reverse):
            d.pop(cart_id, None)
        self._yielding.discard(cart_id)
        self._bt.stop(cart_id)

    def set_marker(self, cart_id: str, marker_id: Optional[int]) -> None:
        cart = self._carts.get(cart_id)
        if cart is None:
            return
        if cart.marker_id is not None:
            self._marker_map.pop(cart.marker_id, None)
        cart.marker_id = marker_id
        if marker_id is not None:
            self._marker_map[marker_id] = cart_id

    @property
    def carts(self) -> Dict[str, Cart]:
        return self._carts

    @property
    def detections(self) -> List[Detection]:
        return self._detections

    @property
    def person_detections(self) -> List[PersonDetection]:
        return self._person_dets

    @property
    def person_detector_available(self) -> bool:
        return self._person_detector is not None and self._person_detector.available

    @property
    def obstacles(self) -> List[Obstacle]:
        return self._obstacles

    @property
    def floor_captured(self) -> bool:
        return self._obstacle_detector is not None and self._obstacle_detector.has_background

    def grid_changed(self) -> None:
        """Walls were edited (e.g. floor re-derived from a new calibration):
        drop everything cached from the old layout."""
        self._wall_cache = None
        self._wall_cache_mask = None
        self._static_dist_frame = -1
        self._grid_bounds = None

    def calibration_changed(self) -> None:
        """The camera→grid mapping changed (point added/removed, cleared).
        Everything in grid space derived from the old mapping is stale:
        planned paths, the measured cart radius and the obstacle-detection ROI."""
        self.grid_changed()
        if self._paths or self._calibrating:
            self.stop_all()
            self.log_message.emit("⏹ Calibration changed — carts stopped, "
                                  "re-dispatch their routines")
        self._radius_measured   = False
        self._cart_radius_cells = CART_RADIUS_CELLS
        self._floor_roi_key     = None
        if not self._calib.is_calibrated:
            self._floor_poly = None
            if self._obstacle_detector is not None:
                self._obstacle_detector.set_roi(None)
        self._obstacle_cells, self._obstacle_plan_cells = self._build_obstacle_cells()
        self._person_cells = self._build_person_cells()

    @property
    def detection_blind(self) -> bool:
        return self._detection_blind

    @property
    def paths(self) -> Dict[str, Tuple[List[Tuple[int, int]], int]]:
        """cart_id → (planned grid path, index of the next waypoint)."""
        return {cid: (path, self._path_idx.get(cid, 0)) for cid, path in self._paths.items()}

    # ── Ambient analysis in grid space ────────────────────────────────────────

    def occupancy_matrix(self) -> np.ndarray:
        """Snapshot of the scene as the planner sees it: one OCC_* code per
        grid cell (rows × cols, uint8). Without calibration only walls are known."""
        rows, cols = self._grid.rows, self._grid.cols
        occ = np.full((rows, cols), OCC_FREE, np.uint8)
        occ[self._grid.cells() != 0] = OCC_WALL
        if not self._calib.is_calibrated:
            return occ

        def mark(cells: Set[Tuple[int, int]], code: int) -> None:
            if not cells:
                return
            rc = np.array(list(cells), np.int32)
            rc = rc[(rc[:, 0] >= 0) & (rc[:, 0] < rows) & (rc[:, 1] >= 0) & (rc[:, 1] < cols)]
            occ[rc[:, 0], rc[:, 1]] = code

        mark(self._wall_margin_cells(), OCC_MARGIN)
        mark(self._obstacle_cells,      OCC_OBSTACLE)
        mark(self._mover_cells,         OCC_MOVER)
        mark(self._person_cells,        OCC_PERSON)
        rad = max(1, math.ceil(self._cart_radius_cells))
        for r, c, _, _ in self.cart_grid_poses().values():
            mark(_disc(int(round(r)), int(round(c)), rad), OCC_CART)
        return occ

    def cart_grid_poses(self) -> Dict[str, Tuple[float, float, float, str]]:
        """cart_id → (row, col, travel heading deg, detection source) for every
        cart the tracker knows about. Empty without calibration."""
        if not self._calib.is_calibrated:
            return {}
        poses = {}
        for cid, d in self._build_pos_map().items():
            row, col = self._calib.pixel_to_grid(d.cx, d.cy)
            poses[cid] = (float(row), float(col),
                          self._grid_heading(d, self._carts[cid].heading_offset), d.source)
        return poses

    @property
    def cart_radius_cells(self) -> float:
        return self._cart_radius_cells

    # ── Empty-floor background ────────────────────────────────────────────────

    def forget_unseen(self) -> None:
        """Operator override for object permanence: drop remembered obstacles
        and carts that are no longer seen (e.g. removed while out of view)."""
        n_ob = self._obstacle_tracker.forget_remembered()
        n_c  = self._tracker.forget_memory()
        self.log_message.emit(f"🧹 Forgot {n_ob} unseen obstacle(s) and {n_c} unseen cart(s)")

    def forget_region(self, polygon_px: np.ndarray) -> None:
        """Operator override: something here was removed by hand — forget it."""
        self._obstacle_tracker.forget_region(polygon_px)

    def capture_floor(self) -> None:
        """Start capturing the empty-floor background over the next frames.
        Carts may stay in view (they are masked out); other objects may not."""
        if self._obstacle_detector is None:
            return
        self._floor_capture_left = FLOOR_CAPTURE_FRAMES * FLOOR_CAPTURE_EVERY
        self.log_message.emit("📷 Capturing empty floor — keep the area clear…")
        seen = {d.marker_id for d in self._detections}
        unseen = [c.name for c in self._carts.values()
                  if c.marker_id is not None and c.marker_id not in seen]
        if unseen:
            self.log_message.emit(f"⚠ Not visible during floor capture: {', '.join(unseen)} — "
                                  f"if in view, they will be captured as floor")

    def clear_floor(self) -> None:
        if self._obstacle_detector is not None:
            self._obstacle_detector.clear_background()
            self._obstacle_tracker.reset()
            self._obstacles = []
            self._obstacle_cells, self._obstacle_plan_cells = set(), set()
            self._obstacle_clear_band = set()
            self._mover_cells = set()

    # ── Per-frame update (called by camera source) ────────────────────────────

    def on_frame(self, img: QImage) -> None:
        self._frame_counter += 1
        frame_bgr = _qimage_to_bgr(img)
        od        = self._obstacle_detector

        # Calibration points are image pixels: keep them on this resolution
        msg = self._calib.match_image_size(frame_bgr.shape[1], frame_bgr.shape[0])
        if msg:
            self.log_message.emit(f"⚠ {msg}")
            self.calibration_invalidated.emit()   # UI re-derives walls etc.
        had_floor = self.floor_captured

        # Floor segmentation first — its mask lets the tracker follow a cart
        # whose marker is washed out by glare
        fg = None
        if od is not None and self._floor_capture_left == 0:
            if self._calib.is_calibrated:
                self._sync_floor_roi()
            # Last frame's cart footprints: lets the detector heal floor that
            # was hidden under parked carts during the capture
            fg = od.update(frame_bgr, [d.footprint(CART_FOOTPRINT_SCALE)
                                       for d in self._detections
                                       if d.marker_id in self._marker_map])
            if had_floor and not self.floor_captured:
                self.log_message.emit("⚠ Camera resolution differs from the saved empty "
                                      "floor — capture the floor again")

        # Cart tracking: ArUco + ROI re-search + blob fallback
        tracked = self._tracker.update(frame_bgr, ids=set(self._marker_map) or None,
                                       fg_mask=fg, fg_scale=od.scale if od else 1.0)
        seen    = {d.marker_id for d in tracked}
        self._update_cart_radius(tracked)
        self._detections = tracked + [d for d in self._tracker.raw
                                      if d.marker_id not in seen]

        if self._floor_capture_left > 0:
            self._capture_floor_step(frame_bgr)

        if od is not None and fg is not None:
            raw = od.obstacles([d.footprint(CART_FOOTPRINT_SCALE) for d in tracked])
            # Object permanence: obstacles lost mid-floor stay remembered;
            # carts seen moving this frame count as things that can carry them off
            movers = [d.footprint(CART_FOOTPRINT_SCALE) for d in tracked
                      if d.source not in (SRC_MEMORY, SRC_PREDICTED)]
            # People the person detector sees (last result) can carry things off too
            movers += [np.float32([[p.x1, p.y1], [p.x2, p.y1], [p.x2, p.y2], [p.x1, p.y2]])
                       for p in self._person_dets]
            self._obstacles = self._obstacle_tracker.update(
                raw, frame_bgr.shape, movers, self._floor_poly)
            self._obstacle_cells, self._obstacle_plan_cells = self._build_obstacle_cells()
            if self._detection_blind:
                self._detection_blind = False
                self.log_message.emit("✓ Obstacle detection back — carts may move again")
        elif od is not None and od.enabled and od.has_background and not od.alignment_ok:
            # Can't see obstacles right now (camera moved or its view is blocked):
            # keep everything known as remembered — it hasn't gone anywhere — and
            # hold the carts until detection is back (fail safe)
            self._obstacles = [Obstacle(o.contour, o.area_px, OB_REMEMBERED, o.track_id)
                               for o in self._obstacles if o.state != OB_MOVING]
            self._obstacle_cells, self._obstacle_plan_cells = self._build_obstacle_cells()
            if not self._detection_blind:
                self._detection_blind = True
                self.log_message.emit("⚠ Obstacle detection lost (camera moved or view "
                                      "blocked) — carts held; recapture the floor if the "
                                      "camera really moved")
        else:
            # Detection off (by the operator) or no floor captured yet
            self._detection_blind = False
            self._obstacles = []
            self._obstacle_cells, self._obstacle_plan_cells = set(), set()
            self._obstacle_clear_band = set()
            self._mover_cells = set()

        # Person detection (async — submits and returns immediately)
        if self._person_detector and self._person_detector.enabled:
            self._person_detector.detect_async(frame_bgr)
            self._person_dets  = self._person_detector.latest()
            self._person_cells = self._build_person_cells()
        else:
            self._person_dets  = []
            self._person_cells = set()

        if self._calib.is_calibrated:
            self._control_step()

        self.state_updated.emit()

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_step(self) -> None:
        pos_map = self._build_pos_map()
        now     = datetime.now()

        for cart_id, cart in self._carts.items():
            if cart_id in self._motor_tests:
                continue                    # driven by _motor_test_tick
            det  = pos_map.get(cart_id)
            if det is not None and det.source == SRC_MEMORY:
                det = None      # not seen: stays stopped (others still avoid its spot)
            path = self._paths.get(cart_id)

            if cart_id in self._calibrating:
                self._calibration_step(cart_id, cart, det, now)
                continue

            if not path:
                if cart.state == CartState.MOVING:
                    self._bt.stop(cart_id)
                    cart.state = CartState.IDLE
                self._block_since.pop(cart_id, None)
                self._progress.pop(cart_id, None)
                self._heading_watch.pop(cart_id, None)
                self._steer.pop(cart_id, None); self._step.pop(cart_id, None)
                self._reverse.pop(cart_id, None)
                continue

            if det is None:
                # Cart not visible (tracker gave up predicting) — never drive blind
                self._steer.pop(cart_id, None); self._step.pop(cart_id, None)
                if cart_id not in self._lost:
                    self._lost.add(cart_id)
                    self._bt.stop(cart_id)
                    cart.state = CartState.WAITING
                    where = "assumed still at its last position" \
                        if cart_id in pos_map else "left the view"
                    self.log_message.emit(f"⚠ {cart.name} lost from view — stopped ({where})")
                continue
            if cart_id in self._lost:
                self._lost.discard(cart_id)
                cart.state = CartState.MOVING
                self.log_message.emit(f"▶ {cart.name} visible again")

            row, col = self._calib.pixel_to_grid(det.cx, det.cy)
            idx      = self._path_idx.get(cart_id, 0)
            radius   = self._cart_radius_cells
            arrival  = max(ARRIVAL_CELLS, 0.25 * radius)

            # Advance past already-reached path nodes; a node that is the
            # routine's current waypoint counts as reaching that waypoint, so a
            # later replan continues from there instead of from the first one
            while idx < len(path):
                trow, tcol = path[idx]
                if math.hypot(tcol - col, trow - row) < arrival:
                    if cart.current_target_waypoint() == path[idx]:
                        cart.waypoint_index += 1
                    idx += 1
                else:
                    break
            self._path_idx[cart_id] = idx

            if idx >= len(path) and cart_id in self._yielding and \
                    cart.waypoint_index < len(cart.waypoints):
                # Moved aside in the middle of a routine: head for the rest of
                # the route again (retrying every BLOCK_REPLAN_SECS)
                self._bt.stop(cart_id)
                self._step.pop(cart_id, None)
                since = self._block_since.get(cart_id)
                if since is None or (now - since).total_seconds() >= BLOCK_REPLAN_SECS:
                    here = (int(round(row)), int(round(col)))
                    if self._replan_around(cart_id, cart, here, self._blocked_cells(
                            cart_id, pos_map, near=here, planning=True)):
                        self._yielding.discard(cart_id)
                continue

            if idx >= len(path):
                self._paths.pop(cart_id, None)
                self._bt.stop(cart_id)
                self._block_since.pop(cart_id, None)
                if cart_id in self._yielding and not cart.waypoints:
                    self.log_message.emit(f"↔ {cart.name} moved aside")
                else:
                    self.log_message.emit(f"✓ {cart.name} reached destination")
                self._yielding.discard(cart_id)
                cart.clear_task()
                continue

            # ── Safety check: persons, obstacles, other carts ─────────────────
            here    = (int(round(row)), int(round(col)))
            blocked = self._blocked_cells(cart_id, pos_map, near=here)
            lookahead = _lookahead_cells((row, col), path[idx:],
                                         max(PERSON_LOOKAHEAD, 2 * radius))
            # Static obstacles / walls: inside the safety margin the cart may
            # move away from them, never closer (the near-clear above lets it
            # start from a tight spot; this keeps it from then driving in)
            closer = self._closer_to_static(lookahead, row, col)
            hit = (lookahead & blocked) | closer
            # Only static things (obstacles, walls) in the way: their edges
            # flicker with lighting, so they must stay on the path a few frames
            # in a row. Anything else (people, movers, carts) stops it at once.
            static_only = hit and cart_id not in self._block_since and \
                not (hit - self._obstacle_cells - self._wall_margin_cells() - closer)
            if static_only:
                n = self._static_hits.get(cart_id, 0) + 1
                self._static_hits[cart_id] = n
                if n < STATIC_HIT_FRAMES:
                    hit = set()
            else:
                self._static_hits.pop(cart_id, None)
            if hit:
                self._clear_frames[cart_id] = 0
                if cart_id not in self._block_since:
                    self._block_since[cart_id] = now
                    self._bt.stop(cart_id)
                    cart.state = CartState.WAITING
                    what = "person" if hit & self._person_cells else \
                           "moving object" if hit & self._mover_cells else \
                           "obstacle" if hit & (self._obstacle_cells | closer) else "cart"
                    self.log_message.emit(f"⚠ {cart.name} stopped — {what} on path "
                                          f"near cell {min(hit)}")
                else:
                    blocked_secs = (now - self._block_since[cart_id]).total_seconds()
                    if blocked_secs >= BLOCK_REPLAN_SECS:
                        # Try to find a detour around the blockage; if
                        # other carts are what's in the way, one gives way
                        self._block_since.pop(cart_id)
                        if not self._replan_around(cart_id, cart, here, self._blocked_cells(
                                cart_id, pos_map, near=here, planning=True)):
                            blockers = self._blocking_carts(cart_id, hit, pos_map)
                            if blockers:
                                self._resolve_cart_block(cart_id, blockers, pos_map, now)
                self._progress.pop(cart_id, None)
                self._heading_watch.pop(cart_id, None)
                self._steer.pop(cart_id, None); self._step.pop(cart_id, None)
                continue   # don't drive this frame

            # Path cleared — resume once it has stayed clear for a moment
            # (debounces flicker at an obstacle's edge)
            if cart_id in self._block_since:
                n_clear = self._clear_frames.get(cart_id, 0) + 1
                self._clear_frames[cart_id] = n_clear
                if n_clear < RESUME_CLEAR_FRAMES:
                    continue
                self._block_since.pop(cart_id)
                self._clear_frames.pop(cart_id, None)
                cart.state = CartState.MOVING
                self.log_message.emit(f"▶ {cart.name} resuming — path clear")

            if self._detection_blind:
                # Obstacle detection is down: don't drive into what we can't see
                self._bt.stop(cart_id)
                cart.state = CartState.WAITING
                self._progress.pop(cart_id, None)
                self._steer.pop(cart_id, None); self._step.pop(cart_id, None)
                continue

            # ── Steer toward next waypoint ────────────────────────────────────
            trow, tcol    = path[idx]
            bearing       = math.degrees(math.atan2(-(trow - row), tcol - col))
            heading       = self._grid_heading(det, cart.heading_offset)
            # In step mode the leading end is only chosen between steps, so a
            # running command keeps the direction it was sized for
            st_now  = self._step.get(cart_id)
            if STEP_MODE and st_now is not None:
                reverse = st_now.get("reverse", False)
            else:
                reverse = self._choose_direction(cart_id, _angle_diff(bearing, heading))
            # Heading of the end that leads: steering below works on it, the
            # same way for both directions
            travel        = heading + 180.0 if reverse else heading
            heading_err   = _angle_diff(bearing, travel)

            if STEP_MODE:
                if self._step_control(cart_id, cart, det, row, col, heading, heading_err,
                                      (trow, tcol), now, reverse):
                    # stuck: no progress over several driving steps
                    self._progress.pop(cart_id, None)
                    self._remember_stuck(cart_id, row, col, travel, now)
                    self.log_message.emit(f"⚠ {cart.name} stuck — driving steps aren't moving "
                                          f"it; replanning around the spot ahead")
                    self._replan_around(cart_id, cart, here, self._blocked_cells(
                        cart_id, pos_map, near=here, planning=True))
                    continue
                cart.state = CartState.MOVING
                continue

            # ── Continuous control (ideal / low-latency setups) ───────────────
            steer    = self._steer.setdefault(cart_id, {"mode": "turn"})
            driving  = steer["mode"] == "drive"
            if abs(heading_err) > (HEADING_DRIVE_MAX_DEG if driving else HEADING_TOL_DEG):
                steer["mode"] = "turn"
                self._progress.pop(cart_id, None)       # turning in place is progress
                self._track_heading(cart, cart_id, det, forward=False)
                if heading_err > 0:
                    self._drive(cart_id, -TURN_SPEED, TURN_SPEED)
                else:
                    self._drive(cart_id, TURN_SPEED, -TURN_SPEED)
            else:
                steer["mode"] = "drive"
                if self._is_stuck(cart_id, row, col, now):
                    self._bt.stop(cart_id)
                    self._progress.pop(cart_id, None)
                    self._remember_stuck(cart_id, row, col, travel, now)
                    self.log_message.emit(f"⚠ {cart.name} stuck — not moving while driving; "
                                          f"replanning around the spot ahead")
                    self._replan_around(cart_id, cart, here, self._blocked_cells(
                        cart_id, pos_map, near=here, planning=True))
                    continue
                self._track_heading(cart, cart_id, det, forward=True, reverse=reverse)
                # heading_err > 0: target is counter-clockwise (to the left) →
                # right side faster, same convention as the turn-in-place above
                # Slow down while the heading is still off, so a lagging
                # correction doesn't swing it far past the line
                base = DRIVE_SPEED * (1.0 - 0.5 * min(1.0, abs(heading_err) / HEADING_DRIVE_MAX_DEG))
                correction = heading_err * STEER_GAIN
                self._drive(cart_id, *_drive_cmd(base, correction, reverse))

            cart.state = CartState.MOVING

    # ── Step-by-step control ──────────────────────────────────────────────────

    def _step_control(self, cart_id: str, cart: Cart, det: Detection, row: float,
                      col: float, heading: float, heading_err: float,
                      target: Tuple[int, int], now: datetime,
                      reverse: bool = False) -> bool:
        """One frame of the decide → act → settle → measure loop. Each command
        runs for a computed time, then the cart stops and the video must show
        it standing still (after the stream's latency) before the result is
        measured and the next command is chosen. The measured result teaches
        the cart's turn rate / forward speed, so the next step is sized right.
        Returns True when the cart is stuck (forward steps don't move it)."""
        rates = self._step_rates.setdefault(cart_id, {
            "turn": STEP_TURN_RATE_INIT,
            "fwd":  STEP_FWD_RADII_PER_S_INIT * self._cart_radius_cells,
            "no_move": 0})
        st = self._step.get(cart_id)
        measured = det.source in ("marker", "marker_roi")
        if st is not None and measured:
            # Rotation summed frame by frame: a step that turns more than
            # 180° can't wrap around and read as a turn the other way
            st["rot"] += _angle_diff(heading, st["last_h"])
            st["last_h"] = heading

        # ── Act: keep the command alive until its time is up ─────────────────
        if st is not None and st["phase"] == "act":
            if now < st["until"]:
                self._drive(cart_id, *st["cmd"])        # resend (firmware watchdog)
                return False
            self._bt.stop(cart_id)
            st.update(phase="settle", stopped=now,
                      on_secs=max(0.02, (now - st["started"]).total_seconds()),
                      still=0, last=None)
            return False

        # ── Settle: wait for video evidence that the cart has stopped ────────
        if st is not None and st["phase"] == "settle":
            self._bt.stop(cart_id)
            waited = (now - st["stopped"]).total_seconds()
            if waited >= STEP_LATENCY_SECS and measured:
                cur = (det.cx, det.cy, heading)
                if st["last"] is not None:
                    moved = math.hypot(cur[0] - st["last"][0], cur[1] - st["last"][1])
                    turned = abs(_angle_diff(cur[2], st["last"][2]))
                    still = moved < STEP_STILL_SIDES * det.side_px and turned < STEP_STILL_DEG
                    st["still"] = st["still"] + 1 if still else 0
                st["last"] = cur
            if st["still"] < STEP_STILL_FRAMES and waited < STEP_SETTLE_MAX_SECS:
                return False
            if not measured:
                return False                            # never judge on a predicted pose
            stuck = self._step_measure(cart_id, cart, det, row, col, heading, st, rates)
            self._step.pop(cart_id, None)
            if stuck:
                rates["no_move"] = 0
                return True
            # fall through: decide the next step on this fresh, settled frame

        # ── Decide ────────────────────────────────────────────────────────────
        if not measured:
            self._bt.stop(cart_id)                      # wait for a real marker reading
            return False
        if abs(heading_err) > HEADING_TOL_DEG:
            sign = 1 if heading_err > 0 else -1         # + = left (CCW): right side forward
            secs = STEP_TURN_AIM * min(abs(heading_err), STEP_TURN_MAX_DEG) / rates["turn"]
            secs = max(STEP_MIN_SECS, min(STEP_TURN_MAX_SECS, secs))
            cmd  = (-sign * TURN_SPEED, sign * TURN_SPEED)
            kind = "turn"
        else:
            dist = math.hypot(target[1] - col, target[0] - row)
            step = min(dist, STEP_FWD_MAX_RADII * self._cart_radius_cells)
            secs = max(STEP_MIN_SECS, min(STEP_FWD_MAX_SECS, step / rates["fwd"]))
            # small trim for the remaining error, fixed for the whole step
            trim = heading_err * STEER_GAIN
            cmd  = _drive_cmd(DRIVE_SPEED, trim, reverse)
            kind = "fwd"        # a driving step, either direction
        st = {"phase": "act", "kind": kind, "cmd": cmd, "reverse": reverse,
              "started": now, "until": now + timedelta(seconds=secs),
              "row": row, "col": col, "rot": 0.0, "last_h": heading,
              "cx": det.cx, "cy": det.cy, "want": heading_err}
        self._step[cart_id] = st
        self._drive(cart_id, *cmd)
        # Stop on time, not on the next camera frame: at 10–15 fps waiting for
        # a frame would stretch a short command by up to a whole frame
        QTimer.singleShot(int(secs * 1000), lambda: self._step_end(cart_id, st))
        return False

    def _step_end(self, cart_id: str, st: dict) -> None:
        """Timer: end a step's command exactly on time (if it's still the
        current step — a safety stop or reset may have replaced it)."""
        if self._step.get(cart_id) is not st or st["phase"] != "act":
            return
        self._bt.stop(cart_id)
        st.update(phase="settle", stopped=st["until"],
                  on_secs=max(0.02, (st["until"] - st["started"]).total_seconds()),
                  still=0, last=None)

    def _step_measure(self, cart_id: str, cart: Cart, det: Detection, row: float,
                      col: float, heading: float, st: dict, rates: dict) -> bool:
        """Compare the settled pose with the pose before the step; learn the
        turn rate / forward speed from it. True = stuck."""
        if st["kind"] == "turn":
            rot = st["rot"]
            if rot * st["want"] <= 0 or abs(rot) < STEP_NO_MOVE_DEG:
                # no evidence it turned: the pulse was too short to break away
                rates["turn"] = max(STEP_TURN_RATE_MIN, rates["turn"] * STEP_NO_MOVE_SHRINK)
            else:
                obs = abs(rot) / st["on_secs"]
                rates["turn"] = max(STEP_TURN_RATE_MIN, min(STEP_TURN_RATE_MAX,
                                    (1 - STEP_LEARN) * rates["turn"] + STEP_LEARN * obs))
            return False

        moved = math.hypot(row - st["row"], col - st["col"])
        # Feed the heading-offset check with this step's actual motion (a
        # reverse step moves opposite to where the front faces)
        dx, dy = det.cx - st["cx"], det.cy - st["cy"]
        if st.get("reverse"):
            dx, dy = -dx, -dy
        self._heading_evidence(cart, cart_id, dx, dy, det)
        if moved < STUCK_MIN_RADII * self._cart_radius_cells:
            rates["fwd"] = max(STEP_FWD_RATE_MIN, rates["fwd"] * STEP_NO_MOVE_SHRINK)
            rates["no_move"] += 1
            return rates["no_move"] >= STEP_STUCK_STEPS
        rates["no_move"] = 0
        obs = moved / st["on_secs"]
        rates["fwd"] = max(STEP_FWD_RATE_MIN,
                           (1 - STEP_LEARN) * rates["fwd"] + STEP_LEARN * obs)
        return False

    def _heading_evidence(self, cart: Cart, cart_id: str, dx: float, dy: float,
                          det: Detection) -> None:
        """One forward movement (pixels) with the marker angle at its end, for
        the running heading-offset check (see _track_heading)."""
        if det.source not in ("marker", "marker_roi"):
            return
        step = math.hypot(dx, dy)
        if step <= 0.5:
            return
        w = self._heading_watch.setdefault(cart_id, {"c": 0.0, "s": 0.0, "dist": 0.0,
                                                     "forward": False})
        off = math.radians(_angle_diff(det.angle_deg, math.degrees(math.atan2(-dy, dx))))
        w["c"] += step * math.cos(off)
        w["s"] += step * math.sin(off)
        w["dist"] += step
        self._check_heading_offset(cart, w, det)

    def _choose_direction(self, cart_id: str, err_front: float) -> bool:
        """Which end of the (bidirectional) cart leads toward a target that
        is `err_front` degrees off its front: True = drive backwards. The end
        needing less turning wins, with hysteresis against flip-flopping."""
        if not REVERSE_DRIVE:
            return False
        reverse   = self._reverse.get(cart_id, False)
        err_back  = _angle_diff(err_front + 180.0, 0.0)
        cur, other = (abs(err_back), abs(err_front)) if reverse else \
                     (abs(err_front), abs(err_back))
        if other < cur - REVERSE_HYST_DEG:
            reverse = not reverse
        self._reverse[cart_id] = reverse
        return reverse

    def _track_heading(self, cart: Cart, cart_id: str, det: Detection, forward: bool,
                       reverse: bool = False) -> None:
        """Compare where the cart actually goes with where its marker says it
        faces. A large mismatch means the mounting offset is wrong (e.g. heading
        calibration never succeeded): steering on it would make the cart circle
        its target, so correct it. Each frame's movement counts when the cart
        was told to drive forward on the previous frame, weighted by distance,
        so the turn-and-drive pattern of a circling cart still adds up quickly.
        `forward` is the command being issued this frame."""
        if det.source not in ("marker", "marker_roi"):
            self._heading_watch.pop(cart_id, None)      # angle not measured this frame
            return
        w = self._heading_watch.setdefault(cart_id, {"c": 0.0, "s": 0.0, "dist": 0.0,
                                                     "forward": False})
        if w["forward"]:
            dx, dy = det.cx - w["x"], det.cy - w["y"]
            if w.get("reverse"):                        # backwards: front faces away
                dx, dy = -dx, -dy
            step = math.hypot(dx, dy)
            if step > 0.5:                              # below that it's marker jitter
                off = math.radians(_angle_diff(det.angle_deg, math.degrees(math.atan2(-dy, dx))))
                w["c"] += step * math.cos(off)
                w["s"] += step * math.sin(off)
                w["dist"] += step
        w.update(x=det.cx, y=det.cy, forward=forward, reverse=reverse)
        self._check_heading_offset(cart, w, det)

    def _check_heading_offset(self, cart: Cart, w: dict, det: Detection) -> None:
        """Enough accumulated forward motion in `w`: correct the cart's
        heading offset if the motion disagrees with it."""
        if w["dist"] < HEADING_CHECK_SIDES * det.side_px:
            return
        offset = math.degrees(math.atan2(w["s"], w["c"]))
        w.update(c=0.0, s=0.0, dist=0.0)
        err = _angle_diff(offset, cart.heading_offset)
        if abs(err) > HEADING_FIX_DEG:
            new, _ = _snap_offset(offset)
            self.log_message.emit(f"🧭 {cart.name}: drives {err:+.0f}° off its marker — "
                                  f"heading offset corrected {cart.heading_offset:+.0f}° "
                                  f"→ {new:+.0f}° (measured {offset:+.0f}°)")
            cart.heading_offset = new

    def _replan_around(self, cart_id: str, cart: Cart, start: Tuple[int, int],
                       blocked: Set[Tuple[int, int]]) -> bool:
        """Try to plan a new path from current position to remaining waypoints,
        treating blocked cells as impassable. Falls back to waiting if no path found."""
        remaining_waypoints = cart.waypoints[cart.waypoint_index:]
        if not remaining_waypoints:
            return False

        path = self._plan_path(start, remaining_waypoints, extra_blocked=blocked)
        if path:
            self._paths[cart_id] = path
            self._path_idx[cart_id] = 0
            cart.state = CartState.MOVING
            self._wait_reason.pop(cart_id, None)
            self.log_message.emit(f"↺ {cart.name} replanned around blockage")
            return True
        # No detour found — keep waiting, reset block timer so we retry.
        # If the destination itself is taken, say so (once) and, if a cart is
        # parked on it, ask that cart to move.
        now = datetime.now()
        self._block_since[cart_id] = now
        target  = remaining_waypoints[0]
        pos_map = self._build_pos_map()
        on_goal = self._blocking_carts(cart_id, {target}, pos_map)
        if on_goal:
            reason = "destination occupied by another cart"
            self._resolve_cart_block(cart_id, on_goal, pos_map, now)
        elif target in (self._person_cells | self._obstacle_plan_cells |
                        self._wall_margin_cells()):
            reason = "destination blocked by an obstacle"
        else:
            reason = "no detour found"
        if self._wait_reason.get(cart_id) != reason:
            self._wait_reason[cart_id] = reason
            self.log_message.emit(f"⏳ {cart.name} waiting — {reason}")
        return False

    # ── Cart–cart deadlocks ───────────────────────────────────────────────────

    def _blocking_carts(self, cart_id: str, hit: Set[Tuple[int, int]],
                        pos_map: Dict[str, Detection]) -> List[str]:
        """Other carts whose footprint discs contain the blocked path cells."""
        rad = max(CART_RADIUS_CELLS, math.ceil(2 * self._cart_radius_cells))
        out = []
        for other_id, d in pos_map.items():
            if other_id == cart_id:
                continue
            r0, c0 = (int(round(v)) for v in self._calib.pixel_to_grid(d.cx, d.cy))
            if hit & _disc(r0, c0, rad):
                out.append(other_id)
        return out

    def _resolve_cart_block(self, cart_id: str, blockers: List[str],
                            pos_map: Dict[str, Detection], now: datetime) -> None:
        """Carts blocking each other with no detour: one of them moves aside.
        An idle blocker always yields; between two busy carts the one later in
        name order yields (fixed priority, so they don't both back off).
        A cart yields at most once per YIELD_COOLDOWN_SECS."""
        cart = self._carts[cart_id]
        for other_id in blockers:
            other = self._carts.get(other_id)
            if other is None:
                continue
            other_idle = other_id not in self._paths
            yielder_id = other_id if other_idle or other.name > cart.name else cart_id
            last = self._last_yield.get(yielder_id)
            if last is not None and (now - last).total_seconds() < YIELD_COOLDOWN_SECS:
                continue
            away_from = other_id if yielder_id == cart_id else cart_id
            if self._move_aside(yielder_id, away_from, pos_map):
                self._last_yield[yielder_id] = now
                return

    def _move_aside(self, cart_id: str, away_from: str,
                    pos_map: Dict[str, Detection]) -> bool:
        """Send cart_id to a free spot 2.5–4.5 cart radii away, as far as
        possible from `away_from` and its route; then continue its routine."""
        me, them = pos_map.get(cart_id), pos_map.get(away_from)
        if me is None or them is None:
            return False
        cart     = self._carts[cart_id]
        r0, c0   = self._calib.pixel_to_grid(me.cx, me.cy)
        tr, tc   = self._calib.pixel_to_grid(them.cx, them.cy)
        here     = (int(round(r0)), int(round(c0)))
        their_path = self._paths.get(away_from, [])[self._path_idx.get(away_from, 0):][:4]
        blocked  = self._blocked_cells(cart_id, pos_map, near=here, planning=True)
        R        = self._cart_radius_cells
        best = None
        for dist in (2.5 * R, 3.5 * R, 4.5 * R):
            for k in range(16):
                a  = 2 * math.pi * k / 16
                cell = (int(round(r0 - dist * math.sin(a))), int(round(c0 + dist * math.cos(a))))
                if not self._grid.is_passable(*cell) or cell in blocked:
                    continue
                score = math.hypot(cell[0] - tr, cell[1] - tc) + min(
                    (math.hypot(cell[0] - pr, cell[1] - pc) for pr, pc in their_path),
                    default=0.0)
                if best is None or score > best[0]:
                    best = (score, cell)
        if best is None:
            return False
        spot = best[1]
        path = self._plan_path(here, [spot] + cart.waypoints[cart.waypoint_index:],
                               extra_blocked=blocked)
        if not path:
            path = self._plan_path(here, [spot], extra_blocked=blocked)
        if not path:
            return False
        self._paths[cart_id] = path
        self._path_idx[cart_id] = 0
        self._block_since.pop(cart_id, None)
        self._yielding.add(cart_id)
        cart.state = CartState.MOVING
        self.log_message.emit(f"↔ {cart.name} giving way to {self._carts[away_from].name}")
        return True

    # ── Person cell computation ───────────────────────────────────────────────

    def _build_person_cells(self) -> Set[Tuple[int, int]]:
        """Convert YOLO bounding boxes to a set of blocked grid cells."""
        if not self._calib.is_calibrated or not self._person_dets:
            return set()

        # Scale factors from camera frame to… we don't know frame size here.
        # PersonDetection coords are in camera pixel space (same as homography input),
        # so pass directly to pixel_to_grid.
        cells: Set[Tuple[int, int]] = set()
        margin = max(PERSON_MARGIN, math.ceil(self._cart_radius_cells))
        for det in self._person_dets:
            # Convert bbox corners to grid, then fill the enclosed region
            r0, c0 = self._calib.pixel_to_grid(det.x1, det.y1)
            r1, c1 = self._calib.pixel_to_grid(det.x2, det.y2)
            rmin = int(math.floor(min(r0, r1))) - margin
            rmax = int(math.ceil( max(r0, r1))) + margin
            cmin = int(math.floor(min(c0, c1))) - margin
            cmax = int(math.ceil( max(c0, c1))) + margin
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    if 0 <= r < self._grid.rows and 0 <= c < self._grid.cols:
                        cells.add((r, c))
        return cells

    def _build_obstacle_cells(self) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """Rasterise obstacle polygons into grid cells. Returns (static cells
        padded by OBSTACLE_MARGIN, the same + PLAN_EXTRA_MARGIN). Moving
        obstacles (people…) go to self._mover_cells, swept area included."""
        self._mover_cells = set()
        self._obstacle_clear_band = set()
        if not self._calib.is_calibrated or not self._obstacles:
            return set(), set()
        rows, cols = self._grid.rows, self._grid.cols
        static = np.zeros((rows, cols), np.uint8)
        moving = np.zeros((rows, cols), np.uint8)
        for ob in self._obstacles:
            xy = self._calib.pixels_to_grid_xy(ob.blocking_polygon)   # (col, row)
            cv2.fillPoly(moving if ob.state == OB_MOVING else static,
                         [np.round(xy).astype(np.int32)], 1)

        def padded(mask, margin: int) -> Set[Tuple[int, int]]:
            if not mask.any():
                return set()
            k = 2 * margin + 1
            m = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            rs, cs = np.nonzero(m)
            return set(zip(rs.tolist(), cs.tolist()))

        margin = max(OBSTACLE_MARGIN, math.ceil(self._cart_radius_cells))
        self._mover_cells = padded(moving, margin)
        plan = padded(static, margin + PLAN_EXTRA_MARGIN)
        wide = margin + PLAN_EXTRA_MARGIN + \
            max(1, math.ceil(PLAN_CLEARANCE_RADII * self._cart_radius_cells))
        self._obstacle_clear_band = padded(static, wide) - plan
        return padded(static, margin), plan

    def _blocked_cells(self, cart_id: str, pos_map: Dict[str, Detection],
                       near: Optional[Tuple[int, int]] = None,
                       planning: bool = False) -> Set[Tuple[int, int]]:
        """Cells this cart must avoid: persons, obstacles and the other carts.
        Static cells within START_CLEAR_CELLS of `near` are left free so a cart
        parked next to an obstacle can still plan its way out — but never
        people or moving obstacles: someone right next to the cart must stop it."""
        # Planning keeps a comfortable gap (clear band) on top of the margin;
        # _theta drops the band again where only a tight path exists
        obstacle_cells = self._obstacle_plan_cells | self._obstacle_clear_band \
            if planning else self._obstacle_cells
        blocked = obstacle_cells | self._wall_margin_cells()
        radius  = self._cart_radius_cells
        now     = datetime.now()
        for cid, (since, cells) in list(self._stuck_cells.items()):
            if (now - since).total_seconds() > STUCK_MEMORY_SECS:
                del self._stuck_cells[cid]
            else:
                blocked = blocked | cells
        for other_id, d in pos_map.items():
            if other_id == cart_id:
                continue
            r0, c0 = (int(round(v)) for v in self._calib.pixel_to_grid(d.cx, d.cy))
            # Two cart bodies must not overlap: block a disc of both radii
            blocked = blocked | _disc(r0, c0, max(CART_RADIUS_CELLS, math.ceil(2 * radius)))
        if near is not None and blocked:
            r0, c0 = near
            blocked = blocked - _disc(r0, c0, max(START_CLEAR_CELLS, math.ceil(radius)))
        return blocked | self._person_cells | self._mover_cells

    def _closer_to_static(self, cells: Set[Tuple[int, int]],
                          row: float, col: float) -> Set[Tuple[int, int]]:
        """Cells where the body would touch a static obstacle or wall AND that
        are closer to it than the cart is now."""
        dist = self._static_dist()
        if dist is None or not cells:
            return set()
        rows, cols = dist.shape
        r0 = min(max(int(round(row)), 0), rows - 1)
        c0 = min(max(int(round(col)), 0), cols - 1)
        d_here = dist[r0, c0]
        R = self._cart_radius_cells
        return {(r, c) for r, c in cells
                if 0 <= r < rows and 0 <= c < cols
                and dist[r, c] < R and dist[r, c] < d_here - 0.5}

    def _static_dist(self) -> Optional[np.ndarray]:
        """Per-cell distance (cells) to the nearest static obstacle, wall or
        grid edge, recomputed once per frame."""
        if self._static_dist_frame == self._frame_counter:
            return self._static_dist_cache
        rows, cols = self._grid.rows, self._grid.cols
        free = np.ones((rows + 2, cols + 2), np.uint8)
        free[0, :] = free[-1, :] = 0
        free[:, 0] = free[:, -1] = 0
        if self._wall_cache_mask is None:
            self._wall_cache_mask = self._grid.cells() != EMPTY
        free[1:-1, 1:-1][self._wall_cache_mask] = 0
        if self._calib.is_calibrated:
            for ob in self._obstacles:
                if ob.state == OB_MOVING:
                    continue
                xy = self._calib.pixels_to_grid_xy(ob.contour) + 1    # (col, row), padded grid
                cv2.fillPoly(free, [np.round(xy).astype(np.int32)], 0)
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 3)[1:-1, 1:-1]
        self._static_dist_cache, self._static_dist_frame = dist, self._frame_counter
        return dist

    def _wall_margin_cells(self) -> Set[Tuple[int, int]]:
        """Passable cells within one cart radius of a wall or the grid edge:
        the planner treats walls as thin lines, but the cart's body is wide."""
        margin = max(1, math.ceil(self._cart_radius_cells))
        if self._wall_cache is not None and self._wall_cache[0] == margin:
            return self._wall_cache[1]
        rows, cols = self._grid.rows, self._grid.cols
        walls = np.ones((rows + 2, cols + 2), np.uint8)            # border = wall
        walls[1:-1, 1:-1] = self._grid.cells() != EMPTY
        k = 2 * margin + 1
        near = cv2.dilate(walls, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        near = (near[1:-1, 1:-1] > 0) & (walls[1:-1, 1:-1] == 0)
        rs, cs = np.nonzero(near)
        cells = set(zip(rs.tolist(), cs.tolist()))
        self._wall_cache = (margin, cells)
        return cells

    def _px_to_cells(self, x: float, y: float, dist_px: float) -> float:
        """Length of a dist_px image distance at (x, y), in grid cells (worst axis)."""
        pts = self._calib.pixels_to_grid_xy(
            np.float32([[x, y], [x + dist_px, y], [x, y + dist_px]]))
        return float(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[0])))

    def _update_cart_radius(self, tracked: List[Detection]) -> None:
        """Track the cart body radius in grid cells from the measured marker size."""
        if not self._calib.is_calibrated:
            return
        radii = [self._px_to_cells(d.cx, d.cy, CART_BODY_RADIUS_SIDES * d.side_px)
                 for d in tracked if d.marker_id in self._marker_map
                 and d.source not in (SRC_PREDICTED, SRC_MEMORY)]
        if radii:
            r = max(radii)
            if self._radius_measured:
                r = 0.9 * self._cart_radius_cells + 0.1 * r
            self._radius_measured = True
            self._cart_radius_cells = max(CART_RADIUS_CELLS, r)

    def _grid_heading(self, det: Detection, offset: float = 0.0) -> float:
        """Travel direction in grid coordinates (deg, 0 = +col, 90 = −row): the
        marker heading minus the cart's mounting offset. The image angle only
        equals the grid angle when the calibration keeps the axes aligned; a
        rotated camera (portrait camera, landscape grid) needs the direction
        mapped through the homography."""
        a  = math.radians(det.angle_deg - offset)
        L  = max(det.side_px, 10.0)
        g  = self._calib.pixels_to_grid_xy(np.float32(
            [[det.cx, det.cy], [det.cx + L * math.cos(a), det.cy - L * math.sin(a)]]))
        dc, dr = g[1] - g[0]
        return math.degrees(math.atan2(-dr, dc))

    def _sync_floor_roi(self) -> None:
        """Limit obstacle detection to the calibrated floor area — the part of
        the grid that has passable cells (cells outside the camera's view are
        usually walled off)."""
        if self._grid_bounds is None:
            free = self._grid.cells() == EMPTY
            rows = np.flatnonzero(free.any(axis=1)).tolist()
            cols = np.flatnonzero(free.any(axis=0)).tolist()
            self._grid_bounds = (rows[0], cols[0], rows[-1] + 1, cols[-1] + 1) if rows \
                else (0, 0, self._grid.rows, self._grid.cols)
        r0, c0, r1, c1 = self._grid_bounds
        poly = self._calib.floor_polygon_px(r1, c1, r0, c0)
        key  = None if poly is None else tuple(np.round(poly, 1).ravel())
        if key != self._floor_roi_key:
            self._floor_roi_key = key
            self._floor_poly = poly
            self._obstacle_detector.set_roi(poly)

    def _capture_floor_step(self, frame_bgr: np.ndarray) -> None:
        od = self._obstacle_detector
        self._floor_capture_left -= 1
        if self._floor_capture_left % FLOOR_CAPTURE_EVERY == 0:
            # Mask out cart bodies so carts parked in view don't become "floor"
            feet = [d.footprint(CART_FOOTPRINT_SCALE) for d in self._detections]
            od.capture_frame(frame_bgr, exclude=feet)
        if self._floor_capture_left == 0:
            if od.finish_capture():
                od.save()
                self._obstacle_tracker.reset()
                self.log_message.emit("✓ Floor captured — obstacle detection active")
            else:
                self.log_message.emit("✕ Floor capture failed")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_pos_map(self) -> Dict[str, Detection]:
        pos = {}
        for d in self._detections:
            cid = self._marker_map.get(d.marker_id)
            if cid and cid in self._carts:
                pos[cid] = d
        return pos

    def _plan_path(self,
                   start: Tuple[int, int],
                   waypoints: List[Tuple[int, int]],
                   extra_blocked: Optional[Set[Tuple[int, int]]] = None
                   ) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = []
        prev = start
        for wp in waypoints:
            seg = self._theta(prev, wp, extra_blocked)
            if seg:
                path.extend(seg[1:])
                prev = wp
            else:
                return []
        return path

    def _theta(self, start: Tuple[int, int], goal: Tuple[int, int],
               blocked: Optional[Set[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        """Theta* leg with comfortable clearance from static obstacles; where
        that doesn't fit (narrow passage), the same leg with only the margin."""
        seg = theta_star(self._grid.is_passable, self._grid.rows, self._grid.cols,
                         start, goal, blocked=blocked)
        if not seg and blocked and self._obstacle_clear_band:
            seg = theta_star(self._grid.is_passable, self._grid.rows, self._grid.cols,
                             start, goal, blocked=blocked - self._obstacle_clear_band)
        return seg

    def _detected_grid_pos(self, cart_id: str) -> Optional[Tuple[int, int]]:
        """Grid cell of a cart the camera currently sees (not a remembered one)."""
        for d in self._detections:
            if self._marker_map.get(d.marker_id) == cart_id and d.source != SRC_MEMORY:
                row, col = self._calib.pixel_to_grid(d.cx, d.cy)
                return (int(round(row)), int(round(col)))
        return None

    # ── Routine trigger ───────────────────────────────────────────────────────

    def preview_route(self, cart_id: str, waypoints: List[Tuple[int, int]]
                      ) -> Tuple[List[Tuple[int, int]], Optional[int], Optional[Tuple[int, int]]]:
        """Plan `waypoints` the way trigger_routine would if it ran now.
        Returns (path, index of the first unreachable waypoint or None, start
        cell). When the cart isn't visible the check starts at the first
        waypoint (start None), so only the legs between waypoints are checked."""
        start = self._detected_grid_pos(cart_id)
        if not waypoints:
            return [], None, start
        anchor  = start if start is not None else waypoints[0]
        blocked = self._blocked_cells(cart_id, self._build_pos_map(), near=start,
                                      planning=True) if self._calib.is_calibrated else set()
        path, prev = [], anchor
        for i, wp in enumerate(waypoints):
            if wp == prev:
                continue
            seg = self._theta(prev, wp, blocked)
            if not seg:
                return path, i, start
            path.extend(seg[1:])
            prev = wp
        return path, None, start

    def occupancy_notice(self) -> str:
        """Why the occupancy matrix may be incomplete right now ('' if it isn't)."""
        if not self._calib.is_calibrated:
            return ("Camera not calibrated — use ⊕ Calib to click 4+ floor points; "
                    "nothing can be mapped onto the grid until then")
        if self._detection_blind:
            return "⚠ Obstacle detection lost — carts held"
        if not self.floor_captured:
            return "Floor not captured — obstacles are not detected"
        return ""

    def trigger_routine(self, routine: Routine) -> None:
        cart = self._carts.get(routine.cart_id)
        if cart is None:
            return
        start = self._detected_grid_pos(routine.cart_id)
        if start is None:
            # Planning from a stale/assumed position could send it anywhere
            self.log_message.emit(f"✕ {cart.name}: not visible to the camera — "
                                  f"'{routine.name}' not started")
            return
        if routine.cart_id in self._calibrating:
            self.log_message.emit(f"✕ {cart.name}: heading calibration running — "
                                  f"'{routine.name}' not started")
            return
        blocked = self._blocked_cells(routine.cart_id, self._build_pos_map(), near=start,
                                      planning=True) \
            if self._calib.is_calibrated else set()
        path  = self._plan_path(start, routine.waypoints, extra_blocked=blocked)
        if path:
            self._paths[routine.cart_id] = path
            self._path_idx[routine.cart_id] = 0
            cart.assign_routine(routine.id, routine.waypoints)
            self._wait_reason.pop(routine.cart_id, None)
            self._yielding.discard(routine.cart_id)
            self.log_message.emit(f"▶ {cart.name}: routine '{routine.name}' started")
        else:
            self.log_message.emit(f"✕ {cart.name}: no path found for '{routine.name}'")

    # ── Stuck detection ───────────────────────────────────────────────────────

    def _is_stuck(self, cart_id: str, row: float, col: float, now: datetime) -> bool:
        """True after STUCK_SECS of forward driving with no real progress."""
        rec = self._progress.get(cart_id)
        if rec is None or math.hypot(row - rec[1], col - rec[2]) > \
                STUCK_MIN_RADII * self._cart_radius_cells:
            self._progress[cart_id] = (now, row, col)
            return False
        return (now - rec[0]).total_seconds() > STUCK_SECS

    def _remember_stuck(self, cart_id: str, row: float, col: float,
                        heading: float, now: datetime) -> None:
        """Something the camera can't see is in the way: block a disc just
        ahead of the cart for a while so the planner goes around it."""
        r  = self._cart_radius_cells
        h  = math.radians(heading)
        cr = int(round(row - (r + 1) * math.sin(h)))
        cc = int(round(col + (r + 1) * math.cos(h)))
        self._stuck_cells[cart_id] = (now, _disc(cr, cc, max(1, math.ceil(0.7 * r))))

    # ── Heading calibration ───────────────────────────────────────────────────

    def calibrate_heading(self, cart_id: str, quiet: bool = False) -> bool:
        """Measure where the cart's marker points relative to its direction of
        travel: drive straight forward CAL_TARGET_SIDES and compare the motion with
        the marker angle. Result goes to Cart.heading_offset. The cart must be
        idle, visible and have free space around it (the direction is unknown)."""
        cart = self._carts.get(cart_id)
        det  = self._build_pos_map().get(cart_id)
        if det is not None and det.source in (SRC_MEMORY, SRC_PREDICTED):
            det = None
        if cart is None or det is None or cart_id in self._paths:
            if cart is not None:
                self.calibration_refusal[cart_id] = "needs an idle cart visible to the camera"
                if not quiet:
                    self.log_message.emit(f"✕ {cart.name}: heading calibration "
                                          f"{self.calibration_refusal[cart_id]}")
            return False
        if not self._bt.is_connected(cart_id):
            # Without a link every DRIVE is dropped silently and the cart
            # "doesn't move" — say what's actually wrong
            self.calibration_refusal[cart_id] = "not connected over Bluetooth"
            if not quiet:
                self.log_message.emit(f"✕ {cart.name}: heading calibration needs the cart "
                                      f"connected — 🔵 Bluetooth → assign its device → Connect")
            return False
        row, col = self._calib.pixel_to_grid(det.cx, det.cy)
        # The direction is unknown and the speed in px/s isn't measured yet:
        # the centre must be able to move ~a cart length in any direction
        # without entering padded obstacle / wall cells
        reach    = CAL_FREE_RADII * self._cart_radius_cells
        blocked  = self._person_cells | self._obstacle_cells | self._wall_margin_cells()
        area     = _disc(int(round(row)), int(round(col)), math.ceil(reach))
        if blocked & area or not all(self._grid.is_passable(r, c) for r, c in area):
            self.calibration_refusal[cart_id] = "needs free space around the cart"
            if not quiet:
                self.log_message.emit(f"✕ {cart.name}: not enough free space around it "
                                      f"for heading calibration")
            return False
        self.calibration_refusal.pop(cart_id, None)
        self._calibrating[cart_id] = self._new_cal_phase("straight", datetime.now())
        self._calibrating[cart_id]["rest_until"] = datetime.now()    # start at once
        self.log_message.emit(f"🧭 {cart.name}: calibrating heading — short forward drive, "
                              f"then a short turn")
        return True

    # ── Blind motor test ──────────────────────────────────────────────────────

    def is_visible(self, cart_id: str) -> bool:
        """The camera currently sees this cart (marker or tracked body)."""
        det = self._build_pos_map().get(cart_id)
        return det is not None and det.source not in (SRC_MEMORY, SRC_PREDICTED)

    def motor_test(self, cart_id: str) -> bool:
        """Run the calibration's motor commands (MOTOR_TEST_STEPS) without the
        camera: nothing is measured — it shows whether the motors run and
        whether the link / ESP32 survive full power. The cart drives blind, so
        it must be lifted or have room."""
        cart = self._carts.get(cart_id)
        if cart is None:
            return False
        if not self._bt.is_connected(cart_id):
            self.log_message.emit(f"✕ {cart.name}: motor test needs the cart connected "
                                  f"over Bluetooth")
            return False
        if cart_id in self._paths or cart_id in self._calibrating:
            self.log_message.emit(f"✕ {cart.name}: motor test needs an idle cart")
            return False
        now = datetime.now()
        status = getattr(self._bt, "status", None)
        up0 = status(cart_id).uptime_s if status else None
        self._motor_tests[cart_id] = {"step": 0, "step_t0": now, "t0": now,
                                      "uptime0": up0, "done_at": None}
        cart.state = CartState.MOVING
        self._test_timer.start()
        plan = ", ".join(f"{name} {secs:g} s" for name, secs, _ in MOTOR_TEST_STEPS)
        self.log_message.emit(f"🔧 {cart.name}: blind motor test ({plan}) — watch the "
                              f"wheels and the DEVICES panel")
        return True

    def _motor_test_tick(self) -> None:
        now = datetime.now()
        for cart_id, st in list(self._motor_tests.items()):
            cart = self._carts.get(cart_id)
            if cart is None:
                del self._motor_tests[cart_id]
                continue
            elapsed = (now - st["t0"]).total_seconds()
            status = getattr(self._bt, "status", None)
            if status is not None:
                s = status(cart_id)
                if s.last_reply_s is not None:
                    # When the cart last answered (the PC notices a dead link
                    # only seconds later): pinpoints when it failed
                    st["last_ok"] = max(0.0, elapsed - s.last_reply_s)
            if not self._bt.is_connected(cart_id):
                del self._motor_tests[cart_id]
                cart.state = CartState.IDLE
                when = ""
                if "last_ok" in st:
                    t_ok = st["last_ok"]
                    when = ("; it stopped answering right when the motors started"
                            if t_ok < 0.3 else
                            f"; it last answered {t_ok:.1f} s into the test "
                            f"({_motor_test_phase(t_ok)})")
                self.log_message.emit(
                    f"✕ {cart.name}: Bluetooth link lost during the motor test{when} — "
                    f"watch for the cart's reset reason when it reconnects "
                    f"(BROWNOUT = power)")
                continue
            if st["done_at"] is not None:
                self._bt.stop(cart_id)
                self._motor_test_verify(cart_id, cart, st, now)
                continue
            name, secs, (l, r) = MOTOR_TEST_STEPS[st["step"]]
            if (now - st["step_t0"]).total_seconds() >= secs:
                st["step"] += 1
                st["step_t0"] = now
                if st["step"] >= len(MOTOR_TEST_STEPS):
                    # Motors off; a reset under load can take the PC up to
                    # SILENT_S to notice, so confirm with a fresh reply first
                    self._bt.stop(cart_id)
                    st["done_at"] = now
                    continue
                name, secs, (l, r) = MOTOR_TEST_STEPS[st["step"]]
            if l == 0 and r == 0:
                self._bt.stop(cart_id)
            else:
                self._drive(cart_id, l, r)
        if not self._motor_tests:
            self._test_timer.stop()

    def _motor_test_verify(self, cart_id: str, cart: Cart, st: dict, now: datetime) -> None:
        """After the motors stop: wait for a reply sent after the test, and use
        the cart's uptime to tell whether it rebooted while driving."""
        status = getattr(self._bt, "status", None)
        waited = (now - st["done_at"]).total_seconds()
        total  = (now - st["t0"]).total_seconds()
        if status is None:                                   # virtual link
            s, fresh = None, True
        else:
            s = status(cart_id)
            fresh = s.last_reply_s is not None and s.last_reply_s < waited
        if not fresh:
            if waited > MOTOR_TEST_VERIFY_S:
                del self._motor_tests[cart_id]
                cart.state = CartState.IDLE
                self.log_message.emit(f"✕ {cart.name}: no reply from the cart for "
                                      f"{waited:.0f} s after the motor test — it probably "
                                      f"reset; see its reset reason when it reconnects")
            return
        del self._motor_tests[cart_id]
        cart.state = CartState.IDLE
        up0 = st["uptime0"]
        if s is not None and up0 is not None and s.uptime_s is not None \
                and s.uptime_s < up0 + total - 3:
            self.log_message.emit(
                f"⚠ {cart.name}: the cart REBOOTED during the motor test (uptime "
                f"{up0} s → {s.uptime_s} s, reset: {s.reset_reason}) — "
                f"{'power: the motors pull the supply down' if s.reset_reason == 'BROWNOUT' else 'see the reset reason'}")
            return
        extra = f" (cart up {s.uptime_s} s, no reboot)" if s and s.uptime_s is not None else ""
        self.log_message.emit(f"✓ {cart.name}: motor test done — the link and the ESP32 "
                              f"stayed up under full power{extra}")

    def _drive(self, cart_id: str, left: int, right: int, raw: bool = False) -> None:
        """Wheel command in the planner's terms (left/right of the cart, +
        forward), rescaled to the cart's measured start PWM and translated
        through its measured motor wiring. raw: exact PWM, no rescaling and no
        firmware breakaway boost (start-PWM measurement)."""
        cart = self._carts.get(cart_id)
        if cart is not None:
            if not raw:
                left, right = cart.lift(left, right)
            left, right = cart.wheel_command(left, right)
        if raw:
            self._bt.drive_raw(cart_id, left, right)
        else:
            self._bt.drive(cart_id, left, right)

    def _cal_command(self, cart_id: str, st: dict) -> None:
        """The motor command of the current calibration phase."""
        phase = st["phase"]
        if phase == "turn":
            self._drive(cart_id, -CAL_TURN_SPEED, CAL_TURN_SPEED)     # planner's "turn left"
        elif phase == "straight":
            self._drive(cart_id, CAL_SPEED, CAL_SPEED)
        elif st.get("probe") != "push":
            self._bt.stop(cart_id)                    # start PWM: between probes
        elif phase == "start_drive":
            self._drive(cart_id, st["speed"], st["speed"], raw=True)
        else:                                                         # start_turn
            self._drive(cart_id, -st["speed"], st["speed"], raw=True)

    @staticmethod
    def _new_cal_phase(phase: str, now: datetime, **extra) -> dict:
        """Fresh measurement state; the cart rests CAL_PAUSE_SECS first so the
        previous motion has died out."""
        st = {"phase": phase, "start": None, "t0": None, "angles": [],
              "speed": START_PWM_FROM, "bumped": None,
              "rot": 0.0, "last_g": None, "rest_until": now + _seconds(CAL_PAUSE_SECS)}
        st.update(extra)
        return st

    def _calibration_step(self, cart_id: str, cart: Cart, det: Optional[Detection],
                          now: datetime) -> None:
        """Heading calibration, two phases:
          straight — DRIVE both sides forward: measures the marker offset, and
                     detects a cart that spins instead (one motor group wired
                     backwards → Cart.invert_right toggled, phase retried once)
          turn     — command a left turn: a cart that turns right has its sides
                     swapped (→ Cart.swap_sides toggled)
          start_drive, start_turn — lowest PWM that starts the cart from rest,
                     straight and turning in place (→ Cart.min_pwm_drive/_turn)"""
        st = self._calibrating[cart_id]
        if not self._bt.is_connected(cart_id):
            del self._calibrating[cart_id]
            cart.state = CartState.IDLE
            self.log_message.emit(f"✕ {cart.name}: heading calibration aborted — "
                                  f"Bluetooth link lost")
            return
        if now < st["rest_until"]:
            self._bt.stop(cart_id)
            return
        elapsed = (now - st["t0"]).total_seconds() if st["t0"] is not None else 0.0
        turning = st["phase"] == "turn"
        if det is None or det.source not in ("marker", "marker_roi"):
            if st["t0"] is not None and elapsed > CAL_MAX_SECS + 1:
                self._bt.stop(cart_id)
                del self._calibrating[cart_id]
                cart.state = CartState.IDLE
                self.log_message.emit(f"✕ {cart.name}: heading calibration lost the marker")
            elif det is None:
                self._bt.stop(cart_id)              # not seen at all: never drive blind
            elif st["t0"] is not None and elapsed < CAL_MAX_SECS:
                # Tracked from its body for a moment (motion blur): keep the motion
                # steady instead of letting the cart's command watchdog stop it;
                # only marker readings go into the measurement
                self._cal_command(cart_id, st)
            return

        # Rotation so far, in grid degrees (CCW +): the planner's frame, so a
        # mirrored camera calibration can't flip the turn-direction check
        g = self._grid_heading(det, 0.0)
        if st["last_g"] is not None:
            st["rot"] += _angle_diff(g, st["last_g"])
        st["last_g"] = g
        if st["start"] is None:
            st["start"], st["t0"], st["bumped"] = (det.cx, det.cy), now, now
        elapsed = (now - st["t0"]).total_seconds()
        if turning:
            self._turn_check_step(cart_id, cart, st, elapsed)
            return
        if st["phase"] in ("start_drive", "start_turn"):
            self._start_pwm_step(cart_id, cart, st, det, now)
            return

        st["angles"].append(det.angle_deg)
        dx, dy = det.cx - st["start"][0], det.cy - st["start"][1]
        moved   = math.hypot(dx, dy)

        # Spinning on a straight command: the two motor groups oppose each other
        if abs(st["rot"]) >= CAL_SPIN_DEG and moved < CAL_TARGET_SIDES * det.side_px:
            self._bt.stop(cart_id)
            if not st.get("flipped"):
                cart.invert_right = not cart.invert_right
                self.log_message.emit(
                    f"🔧 {cart.name} spins in place when both sides are driven forward — "
                    f"its two motor groups run opposite ways; compensating (right side "
                    f"inverted: {cart.invert_right}) and retrying")
                self._calibrating[cart_id] = self._new_cal_phase("straight", now, flipped=True)
                return
            cart.invert_right = not cart.invert_right           # undo: it didn't help
            del self._calibrating[cart_id]
            cart.state = CartState.IDLE
            self.log_message.emit(
                f"✕ {cart.name} spins in place with either side inverted — probably a "
                f"single motor turns backwards (not a whole side): check each motor's "
                f"wiring at the driver / its two pins in the firmware")
            return

        if moved < CAL_TARGET_SIDES * det.side_px and elapsed < CAL_MAX_SECS:
            self._cal_command(cart_id, st)
            cart.state = CartState.MOVING
            return
        # Done: compare displacement direction (image) with the mean marker angle
        self._bt.stop(cart_id)
        cart.state = CartState.IDLE
        del self._calibrating[cart_id]
        if moved < CAL_MIN_MOVE_SIDES * det.side_px:
            if not getattr(self._bt, "is_responding", lambda _: True)(cart_id):
                why = ("the cart never answered over Bluetooth — check that the new "
                       "firmware is flashed and nothing else is connected to it")
            else:
                why = (f"the link is fine and it was driven at full power ({CAL_SPEED}), "
                       f"so check the motors: battery charge, motor-driver power and wiring")
            self.log_message.emit(f"✕ {cart.name}: did not move during heading "
                                  f"calibration — {why}")
            return
        travel = math.degrees(math.atan2(-dy, dx))
        a      = np.radians(st["angles"])
        marker = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))
        raw = _angle_diff(marker, travel)
        cart.heading_offset, snapped = _snap_offset(raw)
        if snapped:
            self.log_message.emit(f"✓ {cart.name}: heading offset {cart.heading_offset:+.0f}° "
                                  f"(measured {raw:+.0f}°, snapped to the marker's mounting)")
        else:
            self.log_message.emit(f"⚠ {cart.name}: heading offset {raw:+.0f}° — not close to "
                                  f"0/90/180/−90°, so the cart probably curved or slipped "
                                  f"during the test drive; check the arrow and re-run 🧭")
        # Second phase: does a left-turn command turn it left?
        self._calibrating[cart_id] = self._new_cal_phase("turn", datetime.now())
        cart.state = CartState.MOVING

    def _turn_check_step(self, cart_id: str, cart: Cart, st: dict, elapsed: float) -> None:
        rot = st["rot"]
        if abs(rot) < CAL_TURN_DEG and elapsed < CAL_TURN_SECS:
            self._drive(cart_id, -CAL_TURN_SPEED, CAL_TURN_SPEED)   # planner's "turn left"
            cart.state = CartState.MOVING
            return
        self._bt.stop(cart_id)
        # Wiring is known now: measure the start PWM (forward, then turning)
        self._calibrating[cart_id] = self._new_cal_phase("start_drive", datetime.now())
        if abs(rot) < CAL_TURN_MIN_DEG:
            self.log_message.emit(
                f"⚠ {cart.name}: barely rotated on a turn command ({rot:+.0f}°) — turn "
                f"direction not checked; turning in place may need more power")
        elif rot > 0:
            self.log_message.emit(f"✓ {cart.name}: turn direction OK ({rot:+.0f}° on a "
                                  f"left-turn command)")
        else:
            cart.swap_sides = not cart.swap_sides
            self.log_message.emit(
                f"🔧 {cart.name} turned right on a left-turn command ({rot:+.0f}°) — its "
                f"left/right motor groups are swapped; compensated (sides swapped: "
                f"{cart.swap_sides})")

    def _start_pwm_step(self, cart_id: str, cart: Cart, st: dict, det: Detection,
                        now: datetime) -> None:
        """Find the lowest RAW PWM that starts the cart from rest (start_drive)
        or turns it in place (start_turn) by bisection over short probes
        (push → stop → look); store it (+ margin) as the cart's start PWM."""
        turning = st["phase"] == "start_turn"
        what    = "turning in place" if turning else "driving forward"
        cart.state = CartState.MOVING

        def probe(pwm: int) -> None:
            # From rest: this frame's pose is the reference for the probe
            st.update(probe="push", speed=pwm, until=now + _seconds(START_PROBE_SECS),
                      start=(det.cx, det.cy), rot=0.0, t0=now)
            self._cal_command(cart_id, st)

        if st.get("probe") is None:                       # first frame of the phase
            st.update(lo=START_PWM_FROM, hi=255, hi_ok=False)
            probe((st["lo"] + st["hi"]) // 2)
            return
        if st["probe"] == "push":
            if now < st["until"]:
                self._cal_command(cart_id, st)
                return
            self._bt.stop(cart_id)
            st.update(probe="settle", until=now + _seconds(START_SETTLE_SECS))
            return
        # Settle: motors off until the video has caught up with the probe
        self._bt.stop(cart_id)
        if now < st["until"]:
            return
        if turning:
            started = abs(st["rot"]) >= START_TURN_DEG
        else:
            started = math.hypot(det.cx - st["start"][0], det.cy - st["start"][1]) \
                >= START_MOVE_SIDES * det.side_px
        if started:
            st["hi"], st["hi_ok"] = st["speed"], True
        elif st["speed"] >= 255:
            # Full power without the boost didn't move it: keep the previous
            # value (the firmware's boost still helps)
            self.log_message.emit(
                f"⚠ {cart.name}: did not start {what} even at PWM 255 without the "
                f"breakaway boost — start PWM not measured (is the firmware with the "
                f"RAW command flashed? battery charged?)")
            self._next_start_phase(cart_id, cart, turning)
            return
        else:
            st["lo"] = st["speed"]
        if st["hi"] - st["lo"] > START_PWM_RESOLUTION:
            probe((st["lo"] + st["hi"]) // 2)
            return
        if not st["hi_ok"]:
            probe(255)                                    # every probe failed: try full power
            return
        pwm = min(255, st["hi"] + START_PWM_MARGIN)
        st["speed"] = st["hi"]
        if turning:
            cart.min_pwm_turn = pwm
        else:
            cart.min_pwm_drive = pwm
        self.log_message.emit(f"✓ {cart.name}: starts {what} at PWM {st['speed']} → "
                              f"start PWM {pwm} (speeds 1–255 now map to {pwm}–255)")
        self._next_start_phase(cart_id, cart, turning)

    def _next_start_phase(self, cart_id: str, cart: Cart, was_turning: bool) -> None:
        if not was_turning:
            self._calibrating[cart_id] = self._new_cal_phase("start_turn", datetime.now())
            return
        del self._calibrating[cart_id]
        cart.state = CartState.IDLE
        self.log_message.emit(f"🧭 {cart.name}: calibration done")

    # ── Routine scheduling ────────────────────────────────────────────────────

    def _check_routines(self) -> None:
        now  = datetime.now()
        busy = {cid for cid, c in self._carts.items() if c.state != CartState.IDLE}
        for routine in self._rm.check_due(now, busy):
            self._rm.mark_triggered(routine, now)
            self.trigger_routine(routine)

    # ── Emergency stop ────────────────────────────────────────────────────────

    def stop_all(self) -> None:
        self._bt.stop_all()
        for cart in self._carts.values():
            if cart.state != CartState.IDLE:
                cart.state = CartState.IDLE
        self._paths.clear()
        self._block_since.clear()
        self._lost.clear()
        self._progress.clear()
        self._steer.clear()
        self._step.clear()
        self._reverse.clear()
        self._calibrating.clear()
        self._motor_tests.clear()
        self._test_timer.stop()
        self._yielding.clear()
        self._wait_reason.clear()


# ── Module-level helpers ──────────────────────────────────────────────────────

def _qimage_to_bgr(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr  = img.bits().asstring(h * w * 3)
    rgb  = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3)).copy()
    return rgb[:, :, ::-1]  # RGB → BGR


def _disc(r0: int, c0: int, rad: int) -> Set[Tuple[int, int]]:
    rad2 = rad * rad
    return {(r0 + dr, c0 + dc)
            for dr in range(-rad, rad + 1)
            for dc in range(-rad, rad + 1) if dr * dr + dc * dc <= rad2}


def _lookahead_cells(start: Tuple[float, float], path: List[Tuple[int, int]],
                     max_cells: int) -> Set[Tuple[int, int]]:
    """Grid cells along the path from `start` through the next nodes, up to
    max_cells of travel. Theta* nodes can be far apart, so the cells on the
    straight segments between them are what the cart actually drives over."""
    cells: Set[Tuple[int, int]] = set()
    travelled = 0.0
    r0, c0 = start
    for r1, c1 in path:
        seg   = math.hypot(r1 - r0, c1 - c0)
        steps = max(1, int(seg * 2))            # half-cell sampling
        for i in range(1, steps + 1):
            t = i / steps
            cells.add((int(round(r0 + (r1 - r0) * t)), int(round(c0 + (c1 - c0) * t))))
            if travelled + seg * t >= max_cells:
                return cells
        travelled += seg
        r0, c0 = r1, c1
    return cells


def _motor_test_phase(t: float) -> str:
    """Which MOTOR_TEST_STEPS step was running t seconds into the test."""
    for name, secs, _ in MOTOR_TEST_STEPS:
        if t < secs:
            return f"during '{name}'"
        t -= secs
    return "after the motors stopped"


def _seconds(s: float) -> timedelta:
    return timedelta(seconds=s)


def _drive_cmd(base: float, trim: float, reverse: bool) -> Tuple[int, int]:
    """(left, right) for driving with `trim` steering toward the CCW side
    (trim > 0 = turn left, as seen from the leading end's heading).
    Backwards, both wheels run negative and swap magnitudes: the rotation
    (right − left) keeps the sign of `trim`, so the same heading error
    steers the leading end — the back — the right way."""
    a = max(1, min(255, int(base - trim)))
    b = max(1, min(255, int(base + trim)))
    return (-b, -a) if reverse else (a, b)


def _angle_diff(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def _snap_offset(raw: float) -> Tuple[float, bool]:
    """Marker mounting offset: snapped to the nearest multiple of 90° when
    within OFFSET_SNAP_DEG of it (markers sit square on the cart), otherwise
    the raw value. Returns (offset in (-180, 180], snapped?)."""
    nearest = _angle_diff(round(raw / 90.0) * 90.0, 0.0)
    if nearest == -180.0:
        nearest = 180.0
    if abs(_angle_diff(raw, nearest)) <= OFFSET_SNAP_DEG:
        return nearest, True
    return round(raw, 1), False
