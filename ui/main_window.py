"""
Main application window — dark blue theme.

Simulation mode  : MapCanvas + SimEngine (virtual grid, timer-driven).
Production mode  : ProdCanvas + ProdEngine (camera feed, BT cart control).
Virtual mode     : production mode on the digital twin — VirtualCamera and
                   VirtualBTController (simulation/digital_twin.py) instead of
                   the webcam and Bluetooth; files kept under data/virtual/.
"""

import itertools
import math
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QComboBox, QToolBar, QStatusBar,
    QMessageBox, QTextEdit, QSplitter, QFrame, QButtonGroup,
    QToolButton, QMenu, QInputDialog,
)

from config import CART_COLORS, ROUTINES_FILE, GRID_ROWS, GRID_COLS, CART_SIZE

VIRTUAL_DATA = "data/virtual"
from core.grid import Grid
from core.cart import Cart
from core.routine_manager import RoutineManager
from simulation.sim_engine import SimEngine
from simulation.maze import generate_default_map
from ui.map_canvas import MapCanvas
from ui.cart_panel import CartPanel
from ui.routine_dialog import RoutineDialog


class MainWindow(QMainWindow):
    def __init__(self, mode: str, camera_index: int = 0, twin_config=None):
        super().__init__()
        # Virtual mode is production mode with simulated hardware
        self._virtual = mode == "virtual"
        self.mode = "production" if self._virtual else mode
        self._camera_index = camera_index
        self.setWindowTitle(f"Cart Automation System  [{mode.upper()}]")
        self.resize(1440, 860)

        self.grid = Grid(GRID_ROWS, GRID_COLS)
        if self._virtual:
            # Virtual carts get new ids every session, so routes are kept with
            # their carts in fleet templates instead of a routines file
            self.rm = RoutineManager(None)
        else:
            # Production derives its floor from the camera calibration
            # (_sync_floor_walls); the factory maze is for simulation only
            if self.mode == "simulation":
                generate_default_map(self.grid)
            self.rm = RoutineManager(ROUTINES_FILE)
        self._color_cycle = itertools.cycle(CART_COLORS)

        # Mode-specific resources
        self._camera          = None
        self._bt_controller   = None
        self._prod_engine     = None    # ProdEngine (production only)
        self._calibration     = None    # FloorCalibration (production only)
        self._detector        = None    # CartDetector (production only)
        self._person_detector = None    # PersonDetector (production only)
        self._obstacle_detector = None  # ObstacleDetector (production only)
        self.engine           = None    # SimEngine (simulation only)
        self._world           = None    # VirtualWorld (virtual only)
        self._twin_populated  = False
        self._needs_heading: set = set()   # virtual carts still to be 🧭-calibrated
        self._heading_tried: dict = {}     # cart id → last refusal reason logged
        self._twin_headings_done = False

        if self.mode == "simulation":
            self.engine = SimEngine(self.grid, self.rm)
        else:
            self._init_production(camera_index, twin_config)

        self._build_toolbar()
        self._build_body()
        self._build_status()
        if self.mode == "production":
            self._update_calib_status()          # calibration may already be loaded
            self._was_calibrated = self._calibration.is_calibrated
            if self._calibration.problem:
                self._append_log(f"✕ Saved calibration not usable: "
                                 f"{self._calibration.problem}")

        if self.mode == "simulation":
            self.engine.state_updated.connect(self._on_sim_state_updated)
            self.engine.log_message.connect(self._append_log)
            self.engine.spawn_obstacles()
        else:
            self._prod_engine.state_updated.connect(self._on_prod_state_updated)
            self._prod_engine.log_message.connect(self._append_log)
            self._prod_engine.calibration_invalidated.connect(self._after_calibration_change)
            if self._camera and self._camera.is_open():
                if not self._virtual:
                    w, h = self._camera.resolution
                    self._append_log(f"📷 Camera running at {w}×{h}")
                self._camera.frame_ready.connect(self._canvas.on_frame)
                self._camera.frame_ready.connect(self._prod_engine.on_frame)
                self._camera.start()
            else:
                self._append_log("⚠ Camera not available — check connection")
            if self._virtual:
                self._prod_engine.log_message.connect(self._on_engine_log)
                self._heading_timer = QTimer(self)
                self._heading_timer.timeout.connect(self._retry_headings)
                self._heading_timer.start(3000)
                self._start_virtual()

        active_carts = self.engine.carts if self.mode == "simulation" else self._prod_engine.carts
        self._cart_panel.refresh_carts(active_carts)

        self._panel_timer = QTimer(self)
        self._panel_timer.timeout.connect(self._refresh_panel)
        self._panel_timer.start(500)

    def _init_production(self, camera_index: int, twin_config=None) -> None:
        from production.cart_detector import CartDetector
        from production.calibration import FloorCalibration
        from production.obstacle_detector import ObstacleDetector
        from production.prod_engine import ProdEngine

        self._detector       = CartDetector()
        if self._virtual:
            from simulation.digital_twin import (
                TwinConfig, VirtualBTController, VirtualCamera, VirtualWorld,
                calibrate_to_world,
            )
            self._world          = VirtualWorld(twin_config or TwinConfig())
            self._camera         = VirtualCamera(self._world)
            self._bt_controller  = VirtualBTController(self._world)
            self._calibration    = FloorCalibration(f"{VIRTUAL_DATA}/calibration.json")
            calibrate_to_world(self._world, self._calibration, self.grid)
            self._calibration.save()
            # Fresh floor capture each run; no person detection (no people in the twin)
            self._obstacle_detector = ObstacleDetector(f"{VIRTUAL_DATA}/floor_background.npz")
        else:
            from production.camera_source import CameraSource
            from production.bt_controller import BTController
            from production.person_detector import PersonDetector
            self._camera         = CameraSource(camera_index)
            self._bt_controller  = BTController()
            self._calibration    = FloorCalibration()
            self._person_detector = PersonDetector()   # loads yolo11n.pt in background
            self._obstacle_detector = ObstacleDetector()
            self._calibration.load()
            self._obstacle_detector.load()             # last captured empty floor, if any

        self._prod_engine = ProdEngine(
            grid             = self.grid,
            rm               = self.rm,
            bt               = self._bt_controller,
            detector         = self._detector,
            calibration      = self._calibration,
            person_detector  = self._person_detector,
            obstacle_detector = self._obstacle_detector,
        )
        self._sync_floor_walls()

        # Log BT events
        self._bt_controller.connected.connect(
            lambda cid: self._append_log(f"🔵 BT connected: {self._cart_label(cid)}"))
        self._bt_controller.disconnected.connect(
            lambda cid: self._append_log(f"⚫ BT disconnected: {self._cart_label(cid)}"))
        self._bt_controller.error.connect(
            lambda cid, msg: self._append_log(f"⚠ BT error ({self._cart_label(cid)}): {msg}"))
        if hasattr(self._bt_controller, "responding"):
            self._bt_controller.responding.connect(
                lambda cid: self._append_log(f"✓ {self._cart_label(cid)} answering over "
                                             f"Bluetooth — firmware OK"))
            self._bt_controller.restarted.connect(self._on_cart_restarted)
            self._bt_controller.stalled.connect(
                lambda cid, n, longest: self._append_log(
                    f"⚠ {self._cart_label(cid)}: Bluetooth link stalled {n}× (longest "
                    f"{longest / 1000:.1f} s) without restarting — radio interference "
                    f"(motor noise / Wi-Fi), not power"))
            self._bt_controller.silent.connect(
                lambda cid, replied: self._append_log(
                    f"⚠ {self._cart_label(cid)}: connected but not answering — "
                    + ("it was answering before: either it restarted (a 'restarted' "
                       "line follows the reconnect) or its radio link stalled "
                       "(a 'link stalled' line follows instead)"
                       if replied else
                       "old firmware (no PONG/heartbeat) or the link stalled; "
                       "power-cycle the cart")))

    # ------------------------------------------------------------------
    # Virtual mode (digital twin)
    # ------------------------------------------------------------------

    def _start_virtual(self) -> None:
        """One app cart per virtual cart (marker + virtual BT link), then capture
        the empty floor; obstacles are dropped once the capture is done."""
        for mid in sorted(self._world.carts):
            self._link_virtual_cart(mid, f"V{mid}")
        self._append_log(f"🧪 Digital twin: {len(self._world.carts)} virtual carts, "
                         f"marker offset {self._world.cfg.marker_offset:g}°")
        QTimer.singleShot(300, self._prod_engine.capture_floor)

    def _twin_missions(self) -> None:
        """Send every idle cart to a random free spot."""
        from core.routine import SCHEDULE_TIME, Routine
        from simulation.digital_twin import mission_goal
        from core.cart import CartState
        busy_goals = []
        for cart in self._prod_engine.carts.values():
            if cart.state != CartState.IDLE:
                continue
            seen = [(d.cx, d.cy) for d in self._prod_engine.detections
                    if d.marker_id != cart.marker_id]
            if not any(d.marker_id == cart.marker_id and d.source != "memory"
                       for d in self._prod_engine.detections):
                continue                              # can't dispatch an unseen cart
            goal = mission_goal(self._world, self._calibration, seen + busy_goals)
            if goal is None:
                continue
            busy_goals.append(goal[0])
            rt = Routine.make(cart.id, "random mission", SCHEDULE_TIME)
            rt.waypoints = [goal[1]]
            self._prod_engine.trigger_routine(rt)

    def _twin_obstacle(self, add: bool) -> None:
        # Objects only come and go with someone carrying them
        if self._world.spawn_walker("drop" if add else "pickup"):
            self._append_log("🚶 Someone comes in to " + ("set an object down" if add
                                                          else "pick an object up"))
        else:
            self._append_log("✕ No free route for a person to bring an object" if add
                             else "✕ No object a person could reach to pick up")

    def _twin_person(self) -> None:
        if self._world.spawn_walker("pass"):
            self._append_log("🚶 Someone walks across the floor")
        else:
            self._append_log("✕ No free route for a person to walk through")

    # ------------------------------------------------------------------
    # Routes drawn on the camera image (production and virtual)
    # ------------------------------------------------------------------

    def _on_route_drawn(self, pts) -> None:
        cart_id = self._cart_panel.selected_cart_id
        if cart_id is None:
            self._append_log("✕ Select a cart first, then draw its route")
            return
        if not self._calibration or not self._calibration.is_calibrated:
            self._append_log("✕ Calibrate the camera before drawing routes")
            return
        cells = []
        for px, py in pts:
            r, c = self._calibration.pixel_to_grid(px, py)
            r = min(max(int(round(r)), 0), self.grid.rows - 1)
            c = min(max(int(round(c)), 0), self.grid.cols - 1)
            if not self.grid.is_passable(r, c):
                self._append_log(f"⚠ Waypoint ({r},{c}) is off the floor — skipped")
                continue
            if not cells or cells[-1] != (r, c):
                cells.append((r, c))
        if not cells:
            return
        dlg = RoutineDialog(self.grid, cart_id, parent=self, waypoints=cells,
                            engine=self._prod_engine)
        if dlg.exec() == RoutineDialog.DialogCode.Accepted:
            routine = dlg.result_routine()
            self.rm.add(routine)
            self._cart_panel.refresh_routines(cart_id)
            self._append_log(f"+ Route '{routine.name}' ({len(routine.waypoints)} waypoints)")
        self._refresh_route_overlay()

    def _refresh_route_overlay(self) -> None:
        """Show the selected cart's routes on the camera image."""
        cart_id = self._cart_panel.selected_cart_id
        overlay = []
        if cart_id and self._calibration and self._calibration.is_calibrated:
            cart = self._prod_engine.carts.get(cart_id)
            colour = cart.color if cart else "#facc15"
            for r in self.rm.for_cart(cart_id):
                overlay.append((colour, [self._calibration.grid_to_pixel(row, col)
                                         for row, col in r.waypoints]))
        self._canvas.route_overlay = overlay
        self._canvas.update()

    # ------------------------------------------------------------------
    # Virtual mode: placing things, cart set-up, templates
    # ------------------------------------------------------------------

    def _build_twin_toolbar(self) -> None:
        tb2 = QToolBar("Twin")
        tb2.setMovable(False)
        tb2.setStyleSheet("spacing: 4px; padding: 2px;")
        self.addToolBarBreak()
        self.addToolBar(tb2)
        tb2.addWidget(QLabel("  Twin: "))
        for label, tip, slot in [
            ("🎲  Missions", "Send every idle cart to a random free spot", self._twin_missions),
            ("＋ Obstacle", "A person walks in and sets an object down",
             lambda: self._twin_obstacle(True)),
            ("－ Obstacle", "A person walks in and picks an object up",
             lambda: self._twin_obstacle(False)),
            ("🚶 Person", "A person walks across the floor", lambda: self._twin_person()),
            ("⚙  Cart Setup", "Physics and marker mounting of the selected cart",
             self._twin_cart_setup),
        ]:
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            tb2.addWidget(b)
        tb2.addSeparator()
        for kind, label in (("fleet", "🚚  Fleet"), ("scenario", "🗺  Scenario")):
            btn = QToolButton()
            btn.setText(label + "  ▾")
            btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            menu = QMenu(btn)
            menu.addAction(f"Save {kind} template…", lambda k=kind: self._save_template(k))
            menu.addAction(f"Load {kind} template…", lambda k=kind: self._load_template(k))
            btn.setMenu(menu)
            btn.setToolTip("Fleet: carts, their set-up and routes" if kind == "fleet"
                           else "Scenario: objects on the floor and foot traffic")
            tb2.addWidget(btn)

    def _next_marker_id(self) -> int:
        used = set(self._world.carts) | {c.marker_id for c in self._prod_engine.carts.values()}
        return next(i for i in range(1, 50) if i not in used)

    def _link_virtual_cart(self, mid: int, name: str, color: Optional[str] = None,
                           heading_offset: Optional[float] = None,
                           calibrate: bool = False) -> Cart:
        """App cart for virtual cart `mid`: marker, virtual BT link. The grid
        position is a placeholder — the engine localises carts from the image."""
        r, c = self._find_spawn()
        cart = Cart.make(name, color or next(self._color_cycle), r, c)
        cart.marker_id = mid
        if heading_offset is not None:
            cart.heading_offset = heading_offset
        self._prod_engine.add_cart(cart)
        bt = self._bt_controller
        bt.assign(cart.id, bt.device_for_marker(mid))
        bt.connect_cart(cart.id)
        self._cart_panel.refresh_carts(self._prod_engine.carts)
        if calibrate:
            # Like an operator would: 🧭 once the camera has seen it — retried
            # every few seconds until there's room around the cart
            self._needs_heading.add(cart.id)
            QTimer.singleShot(800, self._retry_headings)
        return cart

    def _retry_headings(self) -> None:
        from core.cart import CartState
        for cid in list(self._needs_heading):
            cart = self._prod_engine.carts.get(cid)
            if cart is None:
                self._needs_heading.discard(cid)
            elif cart.state == CartState.IDLE and cid not in self._prod_engine._calibrating:
                # Retries are quiet; the log explains a refusal when its reason changes
                eng = self._prod_engine
                if not eng.calibrate_heading(cid, quiet=True):
                    reason = eng.calibration_refusal.get(cid)
                    if reason and self._heading_tried.get(cid) != reason:
                        self._heading_tried[cid] = reason
                        self._append_log(f"🧭 {cart.name}: waiting to calibrate — {reason} "
                                         f"(retrying)")

    def _on_engine_log(self, msg: str) -> None:
        # A successful 🧭 measurement clears the cart's pending calibration
        if msg.startswith("✓ ") and "heading offset" in msg:
            for cid, cart in self._prod_engine.carts.items():
                if msg.startswith(f"✓ {cart.name}:"):
                    self._needs_heading.discard(cid)

    def _twin_place_cart(self, px: float, py: float, heading: float) -> None:
        s = self._world.cfg.scale
        x, y = px / s, py / s
        if not self._world.can_place_cart(x, y):
            self._append_log("✕ No room for a cart there (obstacle, another cart or edge)")
            return
        mid = self._next_marker_id()
        self._world.add_cart(mid, at=(x, y), heading=None if math.isnan(heading) else heading)
        self._link_virtual_cart(mid, f"V{mid}", calibrate=True)
        self._append_log(f"🚗 Cart V{mid} (marker {mid}) put down — calibrating its heading")

    def _twin_place_obstacle(self, px: float, py: float) -> None:
        s = self._world.cfg.scale
        k = self._world.n_obstacle_kinds
        kind = self._next_obstacle_kind = (getattr(self, "_next_obstacle_kind", -1) + 1) % k
        if not self._world.place_obstacle(kind, px / s, py / s,
                                          float(self._world.rng.uniform(-180, 180))):
            self._append_log("✕ No room for an object there")

    def _twin_remove_obstacle(self, px: float, py: float) -> None:
        s = self._world.cfg.scale
        idx = self._world.obstacle_at(px / s, py / s)
        if idx is None:
            return
        poly = self._world.obstacles[idx].polygon * s
        self._world.remove_obstacle(idx)
        self._prod_engine.forget_region(poly)          # taken away by the operator

    def _twin_cart_setup(self) -> None:
        cart_id = self._cart_panel.selected_cart_id
        cart = self._prod_engine.carts.get(cart_id) if cart_id else None
        if cart is None or cart.marker_id not in self._world.carts:
            self._append_log("✕ Select a cart first")
            return
        from ui.twin_dialogs import CartSetupDialog
        before = self._world.cart_params(cart.marker_id)
        dlg = CartSetupDialog(cart.name, before, parent=self)
        if dlg.exec() != CartSetupDialog.DialogCode.Accepted:
            return
        p = dlg.params()
        vc = self._world.carts[cart.marker_id]
        vc.px_per_unit, vc.motor_tau, vc.deadband = p["px_per_unit"], p["motor_tau"], p["deadband"]
        vc.gain_l, vc.gain_r = p["gain_l"], p["gain_r"]
        if p["marker_offset"] != before["marker_offset"]:
            self._world.set_mounting(cart.marker_id, p["marker_offset"])
            self._append_log(f"🧭 {cart.name}: marker re-mounted — recalibrating its heading")
            self._prod_engine.calibrate_heading(cart.id)
        self._append_log(f"⚙ {cart.name} set up")

    def _save_template(self, kind: str) -> None:
        from simulation import templates
        name, ok = QInputDialog.getText(self, f"Save {kind} template", "Template name:")
        if not ok or not name.strip():
            return
        data = templates.fleet_snapshot(self._world, self._prod_engine, self.rm,
                                        self._calibration) if kind == "fleet" \
            else templates.scenario_snapshot(self._world)
        path = templates.save(kind, name.strip(), data)
        self._append_log(f"💾 {kind.capitalize()} template saved: {path}")

    def _load_template(self, kind: str) -> None:
        from simulation import templates
        names = templates.list_templates(kind)
        if not names:
            self._append_log(f"✕ No {kind} templates saved yet")
            return
        name, ok = QInputDialog.getItem(self, f"Load {kind} template", "Template:", names, 0, False)
        if not ok:
            return
        data = templates.load(kind, name)
        if kind == "scenario":
            n = templates.apply_scenario(data, self._world)
            self._prod_engine.forget_unseen()     # the floor was rearranged by hand
            self._append_log(f"🗺 Scenario '{name}': {n} objects, "
                             f"{self._world.cfg.walkers_per_min:g} people/min")
            return
        self._prod_engine.stop_all()
        for cart_id in list(self._prod_engine.carts):
            self._remove_cart(cart_id)
        from core.routine import Routine
        for spec in templates.fleet_specs(data, self._world, self._calibration):
            mid = spec["marker_id"]
            self._world.add_cart(mid, at=(spec["x"], spec["y"]), heading=spec["heading"],
                                 params=spec["params"])
            cart = self._link_virtual_cart(mid, spec["name"], spec.get("color"),
                                           heading_offset=spec.get("heading_offset", 0.0),
                                           calibrate=not spec.get("heading_offset"))
            for d in spec["routines"]:
                self.rm.add(Routine.from_dict({**d, "cart_id": cart.id}))
        self._cart_panel.refresh_carts(self._prod_engine.carts)
        self._cart_panel.refresh_routines()
        self._refresh_route_overlay()
        self._append_log(f"🚚 Fleet '{name}': {len(data.get('carts', []))} carts loaded")

    def closeEvent(self, event) -> None:
        if self._camera:
            self._camera.release()
        if self._bt_controller:
            self._bt_controller.stop_all()
            self._bt_controller.disconnect_all()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        tb = QToolBar("Controls")
        tb.setMovable(False)
        tb.setStyleSheet("spacing: 4px; padding: 2px;")
        self.addToolBar(tb)

        mode_lbl = QLabel(f"  {'VIRTUAL' if self._virtual else self.mode.upper()}  ")
        mode_lbl.setStyleSheet(
            "background:#1d4ed8; color:#e0f2fe; font-weight:bold;"
            "border-radius:4px; padding:2px 8px; font-size:11px;"
            if self.mode == "simulation" else
            "background:#7c3aed; color:#ede9fe; font-weight:bold;"
            "border-radius:4px; padding:2px 8px; font-size:11px;"
            if self._virtual else
            "background:#dc2626; color:#fee2e2; font-weight:bold;"
            "border-radius:4px; padding:2px 8px; font-size:11px;"
        )
        tb.addWidget(mode_lbl)
        tb.addSeparator()

        if self.mode == "simulation":
            self._start_btn = QPushButton("▶  Start")
            self._start_btn.setCheckable(True)
            self._start_btn.setFixedWidth(90)
            self._start_btn.toggled.connect(self._on_start_stop)
            tb.addWidget(self._start_btn)
            tb.addSeparator()

            tb.addWidget(QLabel("  Speed "))
            self._speed_combo = QComboBox()
            self._speed_combo.setFixedWidth(64)
            for label, val in [("1×", 1), ("2×", 2), ("5×", 5), ("10×", 10)]:
                self._speed_combo.addItem(label, val)
            self._speed_combo.currentIndexChanged.connect(self._on_speed_change)
            tb.addWidget(self._speed_combo)
            tb.addSeparator()

            tb.addWidget(QLabel("  Obstacles "))
            self._obs_combo = QComboBox()
            self._obs_combo.setFixedWidth(52)
            for n in [0, 2, 5, 8, 12]:
                self._obs_combo.addItem(str(n), n)
            self._obs_combo.setCurrentIndex(2)
            self._obs_combo.currentIndexChanged.connect(self._on_obstacles_change)
            tb.addWidget(self._obs_combo)
            tb.addSeparator()

            reset_btn = QPushButton("↺  Reset Floor")
            reset_btn.clicked.connect(self._reset_maze)
            tb.addWidget(reset_btn)

            tb.addWidget(QLabel("  "))
            tb.addWidget(QLabel("Right-click map → toggle wall"))
        else:
            # Production toolbar
            if self._virtual:
                self._build_twin_toolbar()
            bt_btn = QPushButton("🔵  BT & Markers")
            bt_btn.clicked.connect(self._open_bt_selector)
            tb.addWidget(bt_btn)
            tb.addSeparator()

            self._person_det_btn = QPushButton("👁  Person Det  ON")
            self._person_det_btn.setCheckable(True)
            self._person_det_btn.setChecked(True)
            if self._person_detector is None:
                self._person_det_btn.setText("👁  Person Det  —")
                self._person_det_btn.setEnabled(False)
            self._person_det_btn.setFixedWidth(140)
            self._person_det_btn.toggled.connect(self._on_person_det_toggled)
            self._person_det_btn.setToolTip(
                "Toggle YOLO person detection (yolo11n). "
                "Model loads in background on first run."
            )
            tb.addWidget(self._person_det_btn)
            tb.addSeparator()

            heading_btn = QPushButton("🧭  Calibrate Heading")
            heading_btn.setToolTip(
                "Drive each idle, visible cart briefly forward to measure how its "
                "marker is mounted relative to its direction of travel. Needs free "
                "space around the cart."
            )
            heading_btn.clicked.connect(self._calibrate_headings)
            tb.addWidget(heading_btn)

            forget_btn = QPushButton("🧹  Forget Unseen")
            forget_btn.setToolTip(
                "Objects and carts lost from view mid-floor are kept where they "
                "were last seen (things don't vanish). Use this if you removed "
                "them while the camera couldn't see."
            )
            forget_btn.clicked.connect(lambda: self._prod_engine.forget_unseen())
            tb.addWidget(forget_btn)

            floor_btn = QPushButton("🧱  Capture Floor")
            floor_btn.clicked.connect(self._capture_floor)
            floor_btn.setToolTip(
                "Record the empty floor (~1 s) as the reference for obstacle "
                "detection. Clear all obstacles first; carts may stay in view."
            )
            tb.addWidget(floor_btn)

            self._obstacle_det_btn = QPushButton("🧱  Obstacle Det  ON")
            self._obstacle_det_btn.setCheckable(True)
            self._obstacle_det_btn.setChecked(True)
            self._obstacle_det_btn.setFixedWidth(160)
            self._obstacle_det_btn.toggled.connect(self._on_obstacle_det_toggled)
            tb.addWidget(self._obstacle_det_btn)
            tb.addSeparator()

            stop_btn = QPushButton("⏹  Stop All")
            stop_btn.clicked.connect(self._emergency_stop)
            stop_btn.setStyleSheet("background:#7f1d1d; color:#fca5a5;")
            tb.addWidget(stop_btn)

    def _build_body(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: cart panel + log
        left_w = QWidget()
        left_w.setFixedWidth(268)
        ll = QVBoxLayout(left_w)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        self._cart_panel = CartPanel(self.rm)
        self._cart_panel.cart_selected.connect(self._on_cart_selected)
        self._cart_panel.cart_add_request.connect(self._add_cart)
        self._cart_panel.cart_del_request.connect(self._remove_cart)
        self._cart_panel.routine_add_request.connect(self._add_routine)
        self._cart_panel.routine_edit_request.connect(self._edit_routine)
        self._cart_panel.routine_del_request.connect(self._del_routine)
        self._cart_panel.routine_run_request.connect(self._run_routine)
        ll.addWidget(self._cart_panel, stretch=1)

        self._device_panel = None
        if self.mode == "production" and not self._virtual:
            from ui.production.device_status import DeviceStatusPanel
            self._device_panel = DeviceStatusPanel(self._bt_controller)
            ll.addWidget(self._device_panel)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #1e3a5f;")
        ll.addWidget(sep)

        log_lbl = QLabel("  Event Log")
        log_lbl.setStyleSheet(
            "font-weight:bold; color:#60a5fa; font-size:10px; padding:4px;"
        )
        ll.addWidget(log_lbl)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(130)
        self._log.setStyleSheet(
            "font-size:10px; font-family:monospace; color:#94a3b8;"
            "background:#060912; border:none;"
        )
        ll.addWidget(self._log)

        splitter.addWidget(left_w)

        # Right: canvas
        right_w = QWidget()
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        if self.mode == "production":
            from ui.production.prod_canvas import ProdCanvas
            self._canvas = ProdCanvas(calibration=self._calibration)
            self._canvas.set_grid_size(self.grid.rows, self.grid.cols)
            self._canvas.calib_point_added.connect(self._on_calib_point)
            self._canvas.calib_point_removed.connect(self._on_calib_point_removed)
            self._canvas.route_finished.connect(self._on_route_drawn)
            if self._virtual:
                self._canvas.cart_place_requested.connect(self._twin_place_cart)
                self._canvas.obstacle_place_requested.connect(self._twin_place_obstacle)
                self._canvas.obstacle_remove_requested.connect(self._twin_remove_obstacle)
            from ui.production.occupancy_canvas import OccupancyCanvas
            self._occ_canvas = OccupancyCanvas()
            self._draw_strip = self._build_drawing_strip()
            right_lay.addWidget(self._draw_strip)
            right_lay.addWidget(self._build_view_bar())
            self._view_split = QSplitter(Qt.Orientation.Horizontal)
            self._view_split.addWidget(self._canvas)
            self._view_split.addWidget(self._occ_canvas)
            right_lay.addWidget(self._view_split)
            self._set_view("combined")
        else:
            self._canvas = MapCanvas(self.grid)
            self._canvas.wall_toggled.connect(self._on_wall_toggled)
            right_lay.addWidget(self._canvas)
        splitter.addWidget(right_w)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter)

    def _build_drawing_strip(self) -> QWidget:
        from ui.production.prod_canvas import DrawMode

        strip = QWidget()
        strip.setFixedHeight(40)
        strip.setStyleSheet(
            "background-color: #07101f;"
            "border-bottom: 1px solid #1e3a5f;"
        )
        lay = QHBoxLayout(strip)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(6)

        lay.addWidget(QLabel("Zones:"))

        self._draw_btn_group = QButtonGroup(strip)
        self._draw_btn_group.setExclusive(False)
        self._draw_mode_btns: dict = {}

        for mode, label, tip in [
            (DrawMode.RECT,      "☐ Rect",    "Draw rectangle (press & drag)"),
            (DrawMode.ELLIPSE,   "○ Ellipse", "Draw ellipse (press & drag)"),
            (DrawMode.POLYGON,   "⬡ Poly",    "Click vertices, double-click to close"),
            (DrawMode.ERASE,     "✕ Erase",   "Click a zone to remove it"),
            (DrawMode.CALIBRATE, "⊕ Calib",   "Click floor points to calibrate camera→grid"),
            (DrawMode.ROUTE,     "🛣 Route",   "Draw a route for the selected cart: click "
                                              "waypoints on the image, right-click undoes the "
                                              "last one, double-click or Enter finishes"),
        ] + ([
            (DrawMode.PLACE_CART, "🚗 Cart",  "Put a cart down: click, or press and drag "
                                              "toward where it should face"),
            (DrawMode.PLACE_OBST, "📦 Object", "Put an object down (click); right-click "
                                              "an object to take it away"),
        ] if self._virtual else []):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedWidth(80)
            btn.setToolTip(tip)
            btn.toggled.connect(lambda checked, m=mode: self._on_draw_mode(m, checked))
            self._draw_btn_group.addButton(btn)
            self._draw_mode_btns[mode] = btn
            lay.addWidget(btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #1e3a5f;")
        lay.addWidget(sep)

        clear_btn = QPushButton("🗑 Clear Zones")
        clear_btn.setFixedWidth(96)
        clear_btn.clicked.connect(self._canvas.clear_zones)
        lay.addWidget(clear_btn)

        clear_calib_btn = QPushButton("↺ Calib")
        clear_calib_btn.setFixedWidth(72)
        clear_calib_btn.setToolTip("Clear calibration points and start over")
        clear_calib_btn.clicked.connect(self._clear_calibration)
        lay.addWidget(clear_calib_btn)

        lay.addStretch()

        self._calib_status = QLabel("Not calibrated")
        self._calib_status.setStyleSheet("color: #64748b; font-size: 10px;")
        lay.addWidget(self._calib_status)

        return strip

    # Production views: (key, label, tip, camera video, camera analysis, occupancy)
    _VIEWS = [
        ("combined",  "🎥+📊 Combined",  "Camera with the analysis drawn on top",  True,  True,  False),
        ("video",     "🎥 Video",        "Raw camera feed, no overlays",            True,  False, False),
        ("analysis",  "📊 Analysis",     "Detections, obstacles and paths only, "
                                         "without the camera image",               False, True,  False),
        ("occupancy", "▦ Occupancy",     "The scene as the planner's occupancy "
                                         "matrix (grid cells)",                    False, False, True),
        ("split",     "◫ Split",         "Video and occupancy matrix side by side", True,  False, True),
    ]

    def _build_view_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(34)
        bar.setStyleSheet("background-color: #07101f; border-bottom: 1px solid #1e3a5f;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 3, 10, 3)
        lay.setSpacing(6)
        lay.addWidget(QLabel("View:"))

        group = QButtonGroup(bar)
        group.setExclusive(True)
        self._view_btns: dict = {}
        for i, (key, label, tip, *_layers) in enumerate(self._VIEWS):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(f"{tip}  (Ctrl+{i + 1})")
            btn.clicked.connect(lambda _, k=key: self._set_view(k))
            QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self, activated=lambda k=key: self._set_view(k))
            group.addButton(btn)
            self._view_btns[key] = btn
            lay.addWidget(btn)
        lay.addStretch()
        return bar

    def _set_view(self, key: str) -> None:
        _, _, _, video, analysis, occupancy = next(v for v in self._VIEWS if v[0] == key)
        show_camera = video or analysis
        self._canvas.set_layers(video, analysis)
        self._canvas.setVisible(show_camera)
        self._occ_canvas.setVisible(occupancy)
        # Drawing tools work on the camera image only
        self._draw_strip.setEnabled(show_camera)
        self._view_btns[key].setChecked(True)
        if occupancy:
            self._refresh_occupancy()
        if show_camera and occupancy:
            half = max(1, self._view_split.width() // 2)
            self._view_split.setSizes([half, half])

    def _refresh_occupancy(self) -> None:
        eng = self._prod_engine
        if eng is None or not self._occ_canvas.isVisible():
            return
        self._occ_canvas.update_state(eng.occupancy_matrix(), eng.carts,
                                      eng.cart_grid_poses(), eng.paths,
                                      eng.cart_radius_cells, eng.occupancy_notice())

    def _update_calib_status(self) -> None:
        if self._calibration is None:
            return
        n = self._calibration.point_count
        if self._calibration.is_calibrated and self._calibration.warning:
            self._calib_status.setText(f"⚠ Calibrated ({n} pts) — "
                                       f"{self._calibration.warning}")
            self._calib_status.setStyleSheet("color: #fb923c; font-size: 10px;")
        elif self._calibration.is_calibrated:
            self._calib_status.setText(
                f"✓ Calibrated ({n} pts, fit ±{self._calibration.max_error_cells:.1f} cells)")
            self._calib_status.setStyleSheet("color: #4ade80; font-size: 10px;")
        elif self._calibration.problem:
            self._calib_status.setText(f"✕ {n} pts, not usable: {self._calibration.problem}")
            self._calib_status.setStyleSheet("color: #f87171; font-size: 10px;")
        else:
            needed = max(0, 4 - n)
            self._calib_status.setText(
                f"Calib: {n} pts — need {needed} more" if n > 0
                else "Not calibrated"
            )
            self._calib_status.setStyleSheet("color: #64748b; font-size: 10px;")

    def _on_draw_mode(self, mode, checked: bool) -> None:
        from ui.production.prod_canvas import DrawMode
        if checked:
            for m, btn in self._draw_mode_btns.items():
                if m != mode and btn.isChecked():
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
            self._canvas.set_draw_mode(mode)
        else:
            if not any(b.isChecked() for b in self._draw_mode_btns.values()):
                self._canvas.set_draw_mode(DrawMode.NONE)

    def _sync_floor_walls(self) -> None:
        """Real production: the floor is the area the calibration points enclose;
        every grid cell outside it is a wall. Uncalibrated, the whole grid is free.
        (Virtual mode sets its walls in calibrate_to_world.)"""
        if self._virtual or self._calibration is None:
            return
        from config import EMPTY, WALL
        cells = self.grid.cells()
        floor = self._calibration.floor_mask(self.grid.rows, self.grid.cols)
        if floor is None:
            cells[:] = EMPTY
        else:
            cells[:] = WALL
            cells[floor] = EMPTY
        self._prod_engine.grid_changed()

    def _on_calib_point(self, px: float, py: float, row: int, col: int) -> None:
        n = self._calibration.point_count if self._calibration else 0
        self._append_log(f"⊕ Calib point #{n}: pixel ({px:.0f},{py:.0f}) → grid ({row},{col})")
        self._after_calibration_change()

    def _on_calib_point_removed(self) -> None:
        n = self._calibration.point_count if self._calibration else 0
        self._append_log(f"↶ Last calib point removed ({n} left)")
        self._after_calibration_change()

    def _clear_calibration(self) -> None:
        if self._calibration:
            self._calibration.clear()
            self._calibration.save()
        self._append_log("↺ Calibration cleared")
        self._after_calibration_change()

    def _after_calibration_change(self) -> None:
        """Re-derive everything that depends on the camera→grid mapping and
        redraw the matrix now, without waiting for the next camera frame."""
        cal = self._calibration
        was = getattr(self, "_was_calibrated", False)
        self._update_calib_status()
        self._sync_floor_walls()
        if self._prod_engine:
            self._prod_engine.calibration_changed()
        if cal is not None:
            if cal.is_calibrated and not was:
                self._append_log(f"✓ Camera calibrated — {cal.point_count} points, "
                                 f"fit ±{cal.max_error_cells:.1f} cells")
            if cal.problem:
                self._append_log(f"✕ Calibration not usable: {cal.problem}")
            elif cal.warning:
                self._append_log(f"⚠ Calibration: {cal.warning}")
            self._was_calibrated = cal.is_calibrated
        self._refresh_occupancy()
        self._canvas.update()

    _RESET_HINTS = {
        "BROWNOUT": "its supply voltage collapsed — the motors' start-up current is "
                    "pulling the battery down. Charge/replace the battery, power the "
                    "ESP32 from its own supply or regulator, add a 470–1000 µF capacitor "
                    "across the motor-driver supply, and check the GND/power wires",
        "PANIC":    "the firmware crashed — connect the ESP32 over USB and read the "
                    "Serial Monitor (115200 baud) for the crash message",
        "WATCHDOG": "the firmware hung — connect over USB and check the Serial Monitor",
        "POWERON":  "it was switched on (or its power was cut completely: a loose "
                    "battery/power wire also shows up like this)",
        "EXTERNAL": "its reset button / EN pin was triggered",
    }

    def _on_cart_restarted(self, cart_id: str, reason: str, uptime: int) -> None:
        hint = self._RESET_HINTS.get(reason, "unknown cause")
        icon = "ℹ" if reason == "POWERON" else "⚠"
        self._append_log(f"{icon} {self._cart_label(cart_id)} restarted {uptime} s ago "
                         f"(reset: {reason or '?'}) — {hint}")

    def _cart_label(self, cart_id: str) -> str:
        cart = self._prod_engine.carts.get(cart_id) if self._prod_engine else None
        return cart.name if cart else (cart_id or "BT")

    def _open_bt_selector(self) -> None:
        from ui.production.bt_selector import BTSelectorDialog
        carts = self._prod_engine.carts if self._prod_engine else {}
        dlg = BTSelectorDialog(carts, self._bt_controller, parent=self)
        if dlg.exec():
            # Sync updated marker_ids back into the prod engine's mapping
            for cart in carts.values():
                self._prod_engine.set_marker(cart.id, cart.marker_id)

    def _on_person_det_toggled(self, enabled: bool) -> None:
        if self._person_detector:
            self._person_detector.enabled = enabled
        label = "ON" if enabled else "OFF"
        self._person_det_btn.setText(f"👁  Person Det  {label}")
        self._append_log(f"{'👁' if enabled else '⊘'} Person detection {'enabled' if enabled else 'disabled'}")

    def _calibrate_headings(self) -> None:
        from core.cart import CartState
        eng = self._prod_engine
        busy, blind = [], []
        for cart in eng.carts.values():
            if cart.state != CartState.IDLE:
                busy.append(cart.name)
            elif not self._virtual and self._bt_controller.is_connected(cart.id) and \
                    not (self._calibration.is_calibrated and eng.is_visible(cart.id)):
                blind.append(cart)                 # can't measure: offer a motor test
            else:
                eng.calibrate_heading(cart.id)
        if busy:
            self._append_log(f"🧭 Skipped (busy — stop them first): {', '.join(busy)}")
        if blind:
            why = ("the camera isn't calibrated" if not self._calibration.is_calibrated
                   else "the camera can't see them")
            names = ", ".join(c.name for c in blind)
            ans = QMessageBox.question(
                self, "Blind motor test",
                f"Heading calibration can't measure {names}: {why}.\n\n"
                f"Run a blind motor test instead? The motors run at full power with "
                f"nothing watching — 2 s forward, then a 1.5 s left turn — to check "
                f"whether the cart's power and Bluetooth link survive.\n\n"
                f"Lift the cart off the floor or make sure it has room.")
            if ans == QMessageBox.StandardButton.Yes:
                for cart in blind:
                    eng.motor_test(cart.id)

    def _capture_floor(self) -> None:
        if self._prod_engine:
            self._prod_engine.capture_floor()

    def _on_obstacle_det_toggled(self, enabled: bool) -> None:
        if self._obstacle_detector:
            self._obstacle_detector.enabled = enabled
        label = "ON" if enabled else "OFF"
        self._obstacle_det_btn.setText(f"🧱  Obstacle Det  {label}")

    def _emergency_stop(self) -> None:
        if self._prod_engine:
            self._prod_engine.stop_all()
        self._append_log("⏹ Emergency stop — all carts halted")

    def _build_status(self) -> None:
        bar = QStatusBar()
        bar.setStyleSheet("font-size:11px; color:#64748b;")
        self.setStatusBar(bar)
        self._status_tick  = QLabel("  Frames: 0")
        self._status_carts = QLabel("  Carts: 0")
        self._status_mode  = QLabel(f"  {self.mode.upper()}")
        bar.addWidget(self._status_mode)
        bar.addWidget(QLabel(" | "))
        bar.addWidget(self._status_tick)
        bar.addWidget(QLabel(" | "))
        bar.addWidget(self._status_carts)
        self._tick_count = 0

    # ------------------------------------------------------------------
    # Engine → UI
    # ------------------------------------------------------------------

    def _on_sim_state_updated(self) -> None:
        self._tick_count += 1
        self._status_tick.setText(f"  Ticks: {self._tick_count}")
        self._status_carts.setText(f"  Carts: {len(self.engine.carts)}")
        self._canvas.update_state(self.engine.carts, self.engine.obstacles)

    def _on_prod_state_updated(self) -> None:
        self._tick_count += 1
        self._status_tick.setText(f"  Frames: {self._tick_count}")
        n_persons = len(self._prod_engine.person_detections)
        status = f"  Carts: {len(self._prod_engine.carts)}"
        if n_persons:
            status += f"  ⚠ {n_persons} person{'s' if n_persons > 1 else ''}"
        if self._virtual and self._prod_engine.floor_captured and not self._twin_populated:
            # The operator's set-up steps, done for you: measure each cart's
            # marker mounting on the empty floor (🧭), then let the world fill up
            if not self._twin_headings_done:
                self._twin_headings_done = True
                self._append_log("🧭 Measuring each cart's marker mounting (heading calibration)")
                self._calibrate_headings()
            elif not self._prod_engine._calibrating:
                self._twin_populated = True
                for _ in range(self._world.cfg.n_obstacles):
                    self._world.add_obstacle()
                self._world.walkers_enabled = True
        if self._virtual:
            hits = sum(c.collisions for c in self._world.carts.values())
            status += f"  · twin t={self._world.t:.0f}s, collisions {hits}"
        if not self._prod_engine.floor_captured:
            status += "  · floor not captured"
        elif self._prod_engine.obstacles:
            status += f"  · {len(self._prod_engine.obstacles)} obstacles"
        self._status_carts.setText(status)
        self._canvas.update_state(
            self._prod_engine.carts,
            self._prod_engine.detections,
            paths=dict(self._prod_engine._paths),
            person_detections=self._prod_engine.person_detections,
            obstacles=self._prod_engine.obstacles,
        )
        self._refresh_occupancy()

    def _refresh_panel(self) -> None:
        carts = self.engine.carts if self.mode == "simulation" else self._prod_engine.carts
        self._cart_panel.refresh_carts(carts)
        if self._device_panel is not None:
            self._device_panel.refresh(carts)

    def _append_log(self, msg: str) -> None:
        self._log.append(msg)

    # ------------------------------------------------------------------
    # Toolbar handlers (simulation)
    # ------------------------------------------------------------------

    def _on_start_stop(self, checked: bool) -> None:
        if checked:
            self.engine.start()
            self._start_btn.setText("⏸  Pause")
        else:
            self.engine.stop()
            self._start_btn.setText("▶  Start")

    def _on_speed_change(self) -> None:
        self.engine.set_speed(self._speed_combo.currentData())

    def _on_obstacles_change(self) -> None:
        self.engine.spawn_obstacles(self._obs_combo.currentData())

    def _on_wall_toggled(self, row: int, col: int) -> None:
        self.grid.toggle_wall(row, col)
        self._canvas.update()

    def _on_cart_selected(self, cart_id: str) -> None:
        self._canvas.highlight_cart(cart_id)
        if self.mode == "production":
            self._occ_canvas.highlight_cart(cart_id)
            self._refresh_route_overlay()

    def _reset_maze(self) -> None:
        was_running = self.engine.running
        self.engine.stop()
        generate_default_map(self.grid)
        self.engine.spawn_obstacles(self._obs_combo.currentData())
        self._canvas.update()
        if was_running:
            self.engine.start()
        self._append_log("↺ Maze reset to default")

    # ------------------------------------------------------------------
    # Cart management
    # ------------------------------------------------------------------

    def _add_cart(self, name: str) -> None:
        if self._virtual:
            # A real (virtual) cart on a free spot, not just an entry in the list
            mid = self._next_marker_id()
            self._world.add_cart(mid)
            self._link_virtual_cart(mid, name, calibrate=True)
            return
        color = next(self._color_cycle)
        r, c = self._find_spawn()
        cart = Cart.make(name, color, r, c)
        if self.mode == "simulation":
            self.engine.add_cart(cart)
            self._cart_panel.refresh_carts(self.engine.carts)
        else:
            self._prod_engine.add_cart(cart)
            self._cart_panel.refresh_carts(self._prod_engine.carts)
        self._append_log(f"+ Cart '{name}' added")

    def _remove_cart(self, cart_id: str) -> None:
        for r in self.rm.for_cart(cart_id):
            self.rm.remove(r.id)
        if self._virtual:
            cart = self._prod_engine.carts.get(cart_id)
            if cart is not None and cart.marker_id is not None:
                self._world.remove_cart(cart.marker_id)     # taken off the floor
            self._bt_controller.unassign(cart_id)
        if self.mode == "simulation":
            self.engine.remove_cart(cart_id)
            self._cart_panel.refresh_carts(self.engine.carts)
        else:
            self._prod_engine.remove_cart(cart_id)
            self._cart_panel.refresh_carts(self._prod_engine.carts)
        self._cart_panel.refresh_routines()

    def _find_spawn(self):
        ch, cw = CART_SIZE
        occupied = set()
        carts = self.engine.carts if self.mode == "simulation" else self._prod_engine.carts
        for cart in carts.values():
            occupied |= cart.all_cells()
        for r in range(1, self.grid.rows - ch):
            for c in range(1, self.grid.cols - cw):
                if all(
                    self.grid.is_passable(r + dr, c + dc)
                    and (r + dr, c + dc) not in occupied
                    for dr in range(ch) for dc in range(cw)
                ):
                    return r, c
        return 1, 1

    # ------------------------------------------------------------------
    # Routine management
    # ------------------------------------------------------------------

    def _add_routine(self, cart_id: str) -> None:
        dlg = RoutineDialog(self.grid, cart_id, parent=self, engine=self._prod_engine)
        if dlg.exec() == RoutineDialog.DialogCode.Accepted:
            routine = dlg.result_routine()
            self.rm.add(routine)
            self._cart_panel.refresh_routines(cart_id)
            self._append_log(f"+ Routine '{routine.name}' added")
            if self.mode == "production":
                self._refresh_route_overlay()

    def _edit_routine(self, routine_id: str) -> None:
        routine = self.rm.get(routine_id)
        if not routine:
            return
        dlg = RoutineDialog(self.grid, routine.cart_id, routine=routine, parent=self,
                            engine=self._prod_engine)
        if dlg.exec() == RoutineDialog.DialogCode.Accepted:
            updated = dlg.result_routine()
            self.rm.update(updated)
            self._cart_panel.refresh_routines(routine.cart_id)
            self._append_log(f"✎ Routine '{updated.name}' updated")
            if self.mode == "production":
                self._refresh_route_overlay()

    def _del_routine(self, routine_id: str) -> None:
        routine = self.rm.get(routine_id)
        if not routine:
            return
        ans = QMessageBox.question(
            self, "Delete Routine", f"Delete '{routine.name}'?"
        )
        if ans == QMessageBox.StandardButton.Yes:
            self.rm.remove(routine_id)
            self._cart_panel.refresh_routines(routine.cart_id)
            self._append_log(f"✗ Routine '{routine.name}' deleted")
            if self.mode == "production":
                self._refresh_route_overlay()

    def _run_routine(self, routine_id: str) -> None:
        routine = self.rm.get(routine_id)
        if not routine:
            return
        if self.mode == "simulation":
            self.engine.trigger_routine(routine)
            if not self.engine.running:
                self.engine.start()
                self._start_btn.setChecked(True)
        else:
            if self._virtual and routine.cart_id in self._needs_heading:
                cart = self._prod_engine.carts.get(routine.cart_id)
                self._append_log(f"⚠ {cart.name if cart else 'cart'}: heading not calibrated "
                                 f"yet (🧭 needs free space around it) — it may drive the "
                                 f"wrong way")
            self._prod_engine.trigger_routine(routine)
