"""
Dialog for creating / editing a Routine.

The dialog embeds a map where the user clicks cells to add waypoints:
  Simulation — static mini-map of the walls.
  Production — live mirror of the occupancy matrix (obstacles, people, carts
               as they are now). Waypoints on walls / obstacles are refused and
               the route is planned from the cart's current position as you
               build it, so routes that can't be driven show up before saving.
"""

from typing import List, Optional, Tuple

import numpy as np

from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QMouseEvent, QImage
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QListWidget, QListWidgetItem, QPushButton, QWidget,
    QTimeEdit, QGroupBox, QSizePolicy, QMessageBox,
)
from PyQt6.QtCore import QTime

from config import MINI_CELL_SIZE, GRID_ROWS, GRID_COLS
from core.grid import Grid
from core.routine import Routine, SCHEDULE_TIME, SCHEDULE_INTERVAL


_MINI_EMPTY = (13, 18, 32)    # #0d1220
_MINI_WALL  = (30, 58, 95)    # #1e3a5f
_MINI_W = GRID_COLS * MINI_CELL_SIZE   # 576 px
_MINI_H = GRID_ROWS * MINI_CELL_SIZE   # 324 px
_LIVE_W, _LIVE_H = GRID_COLS * 4, GRID_ROWS * 4   # 768×432 live occupancy map

_C_WAYPOINT = QColor("#f72585")
_C_BAD      = QColor("#ef4444")
_C_ROUTE    = QColor(247, 37, 133, 200)
_C_UNCHECKED = QColor(148, 163, 184, 150)

LIVE_REFRESH_MS = 250      # occupancy mirror refresh
ROUTE_CHECK_MS  = 1000     # re-plan the preview as the scene changes


class _MiniMap(QWidget):
    """Clickable mini-map for waypoint picking. Fixed 576×324 display."""
    waypoints_changed = pyqtSignal()
    refused           = pyqtSignal(str)   # why a click didn't add a waypoint

    def __init__(self, grid: Grid, parent=None):
        super().__init__(parent)
        self.grid = grid
        self.waypoints: List[Tuple[int, int]] = []
        self.setFixedSize(_MINI_W, _MINI_H)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        col = event.pos().x() // MINI_CELL_SIZE
        row = event.pos().y() // MINI_CELL_SIZE
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            cell = (row, col)
            if cell in self.waypoints:
                self.waypoints.remove(cell)
            elif self.grid.is_wall(row, col):
                self.refused.emit(f"({row}, {col}) is a wall")
                return
            else:
                self.waypoints.append(cell)
            self.update()
            self.waypoints_changed.emit()

    def paintEvent(self, _) -> None:
        p = QPainter(self)

        # Fast background via numpy → QImage scaled to display size
        cells = self.grid.cells()
        rgb = np.empty((GRID_ROWS, GRID_COLS, 3), dtype=np.uint8)
        rgb[:] = _MINI_EMPTY
        rgb[cells == 1] = _MINI_WALL
        img = QImage(rgb.tobytes(), GRID_COLS, GRID_ROWS,
                     GRID_COLS * 3, QImage.Format.Format_RGB888)
        p.drawImage(
            QRectF(0, 0, _MINI_W, _MINI_H),
            img,
            QRectF(0, 0, GRID_COLS, GRID_ROWS),
        )

        # Waypoints — larger dots so they're visible at 3 px/cell
        rad = MINI_CELL_SIZE + 1   # 4 px radius = 8 px diameter
        p.setFont(QFont("Arial", 5, QFont.Weight.Bold))
        for i, (wr, wc) in enumerate(self.waypoints):
            cx = wc * MINI_CELL_SIZE + MINI_CELL_SIZE // 2
            cy = wr * MINI_CELL_SIZE + MINI_CELL_SIZE // 2
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor("#f72585")))
            p.drawEllipse(QRectF(cx - rad, cy - rad, 2 * rad, 2 * rad))
            p.setPen(QPen(QColor("#ffffff")))
            p.drawText(QRectF(cx - rad, cy - rad, 2 * rad, 2 * rad),
                       Qt.AlignmentFlag.AlignCenter, str(i + 1))

        p.end()


def _live_map(engine, cart_id: str):
    """Production waypoint picker: the occupancy canvas, live, plus the route
    being built. Imported lazily — simulation mode has no production modules."""
    from production.prod_engine import (
        OCC_CART, OCC_MARGIN, OCC_MOVER, OCC_OBSTACLE, OCC_PERSON, OCC_WALL,
    )
    from ui.production.occupancy_canvas import OccupancyCanvas

    refuse = {OCC_WALL:     "is off the floor (wall)",
              OCC_MARGIN:   "is too close to a wall for the cart's body",
              OCC_OBSTACLE: "is on an obstacle"}
    transient = {OCC_PERSON: "a person", OCC_MOVER: "a moving object",
                 OCC_CART:   "a cart"}

    class _LiveMap(OccupancyCanvas):
        waypoints_changed = pyqtSignal()
        refused           = pyqtSignal(str)

        def __init__(self):
            super().__init__()
            self.waypoints: List[Tuple[int, int]] = []
            self.route: List[Tuple[int, int]] = []     # planned cells, from the start
            self.start: Optional[Tuple[int, int]] = None
            self.bad_index: Optional[int] = None
            self.setFixedSize(_LIVE_W, _LIVE_H)
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.highlight_cart(cart_id)
            self.refresh()

        def refresh(self) -> None:
            self.update_state(engine.occupancy_matrix(), engine.carts,
                              engine.cart_grid_poses(), engine.paths,
                              engine.cart_radius_cells, engine.occupancy_notice())

        def cell_state(self, cell: Tuple[int, int]) -> Optional[str]:
            """What occupies a cell right now, if it's something that moves."""
            what = transient.get(int(self._occ[cell]))
            pose = engine.cart_grid_poses().get(cart_id)
            if what == "a cart" and pose and \
                    abs(pose[0] - cell[0]) + abs(pose[1] - cell[1]) <= 2 * engine.cart_radius_cells:
                return None                                  # its own cart
            return what

        def mousePressEvent(self, event: QMouseEvent) -> None:
            cell = self._cell_at(event.pos())
            if cell is None:
                return
            if cell in self.waypoints:
                self.waypoints.remove(cell)
            else:
                why = refuse.get(int(self._occ[cell]))
                if why:
                    self.refused.emit(f"Row {cell[0]}, col {cell[1]} {why} — pick a free cell")
                    return
                self.waypoints.append(cell)
            self.update()
            self.waypoints_changed.emit()

        def paintEvent(self, event) -> None:
            super().paintEvent(event)
            if self._occ is None:
                return
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            scale = self._viewport()[0]
            centre = lambda rc: self._cell_center(*rc)
            # Planned route (reachable part)
            pts = ([self.start] if self.start else []) + self.route
            p.setPen(QPen(_C_ROUTE, max(2.0, scale * 0.6)))
            for a, b in zip(pts, pts[1:]):
                p.drawLine(centre(a), centre(b))
            # The leg that can't be planned, then the unchecked rest
            if self.bad_index is not None:
                legs = ([self.start] if self.start else [None]) + self.waypoints
                for i in range(self.bad_index, len(self.waypoints)):
                    a, b = legs[i], legs[i + 1]
                    if a is None:
                        continue
                    colour = _C_BAD if i == self.bad_index else _C_UNCHECKED
                    p.setPen(QPen(colour, 2, Qt.PenStyle.DashLine))
                    p.drawLine(centre(a), centre(b))
            # Waypoints
            rad = max(7.0, scale * 1.8)
            p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            for i, wp in enumerate(self.waypoints):
                c = centre(wp)
                p.setPen(QPen(QColor("#ffffff"), 1))
                p.setBrush(QBrush(_C_BAD if i == self.bad_index else _C_WAYPOINT))
                p.drawEllipse(c, rad, rad)
                p.drawText(QRectF(c.x() - rad, c.y() - rad, 2 * rad, 2 * rad),
                           Qt.AlignmentFlag.AlignCenter, str(i + 1))
            p.end()

    return _LiveMap()


class RoutineDialog(QDialog):
    def __init__(self, grid: Grid, cart_id: str, routine: Optional[Routine] = None,
                 parent=None, waypoints: Optional[List[Tuple[int, int]]] = None,
                 engine=None):
        """engine: the ProdEngine in production mode — the map then mirrors
        its occupancy matrix live and the route is checked as it is built."""
        super().__init__(parent)
        self.grid = grid
        self.cart_id = cart_id
        self._editing = routine
        self._engine = engine
        self._route_ok = True

        self.setWindowTitle("Edit Routine" if routine else "New Routine")
        self.setMinimumWidth(1000 if engine else 780)
        self._build_ui()

        if routine:
            self._populate(routine)
        elif waypoints:
            # Route drawn on the camera image: start from those waypoints
            self._mini_map.waypoints = [tuple(w) for w in waypoints]
            self._mini_map.update()
            self._sync_wp_list()

        if engine is not None:
            self._live_timer = QTimer(self)
            self._live_timer.timeout.connect(self._mini_map.refresh)
            self._live_timer.start(LIVE_REFRESH_MS)
            self._check_timer = QTimer(self)
            self._check_timer.timeout.connect(self._check_route)
            self._check_timer.start(ROUTE_CHECK_MS)
            self._check_route()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        form.addRow("Name:", self._name_edit)

        self._sched_combo = QComboBox()
        self._sched_combo.addItem("Time of day", SCHEDULE_TIME)
        self._sched_combo.addItem("Repeat interval", SCHEDULE_INTERVAL)
        self._sched_combo.currentIndexChanged.connect(self._on_sched_change)
        form.addRow("Schedule:", self._sched_combo)

        # Time of day row
        self._time_widget = QWidget()
        tl = QHBoxLayout(self._time_widget)
        tl.setContentsMargins(0, 0, 0, 0)
        self._time_edit = QTimeEdit()
        self._time_edit.setDisplayFormat("HH:mm")
        tl.addWidget(self._time_edit)
        tl.addStretch()
        form.addRow("Time:", self._time_widget)

        # Interval row
        self._interval_widget = QWidget()
        il = QHBoxLayout(self._interval_widget)
        il.setContentsMargins(0, 0, 0, 0)
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(5, 86400)
        self._interval_spin.setValue(60)
        self._interval_spin.setSuffix(" s")
        il.addWidget(self._interval_spin)
        il.addStretch()
        form.addRow("Interval:", self._interval_widget)

        self._repeat_check = QCheckBox("Repeat after completing all waypoints")
        form.addRow("", self._repeat_check)
        self._enabled_check = QCheckBox("Enabled")
        self._enabled_check.setChecked(True)
        form.addRow("", self._enabled_check)

        root.addLayout(form)

        # Waypoints + mini map
        wp_group = QGroupBox("Waypoints  (click on map to add/remove)")
        wp_layout = QHBoxLayout(wp_group)

        self._wp_list = QListWidget()
        self._wp_list.setFixedWidth(160)
        wp_btn_col = QVBoxLayout()
        self._rm_wp_btn = QPushButton("Remove selected")
        self._clear_wp_btn = QPushButton("Clear all")
        self._rm_wp_btn.clicked.connect(self._remove_waypoint)
        self._clear_wp_btn.clicked.connect(self._clear_waypoints)
        wp_btn_col.addWidget(self._rm_wp_btn)
        wp_btn_col.addWidget(self._clear_wp_btn)
        wp_btn_col.addStretch()

        wp_layout.addWidget(self._wp_list)
        wp_layout.addLayout(wp_btn_col)

        self._mini_map = _live_map(self._engine, self.cart_id) if self._engine \
            else _MiniMap(self.grid)
        self._mini_map.waypoints_changed.connect(self._sync_wp_list)
        self._mini_map.refused.connect(self._show_refusal)
        map_col = QVBoxLayout()
        map_col.addWidget(self._mini_map)
        self._status = QLabel(
            "Live floor: obstacles, people and carts as they are now. "
            "Click to add a waypoint, click it again to remove it."
            if self._engine else "Click to add a waypoint, click it again to remove it.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #94a3b8; font-size: 11px;")
        map_col.addWidget(self._status)
        wp_layout.addLayout(map_col)

        root.addWidget(wp_group)

        # OK / Cancel
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Save")
        ok_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self._on_ok)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        root.addLayout(btn_row)

        self._on_sched_change()

    def _on_sched_change(self) -> None:
        is_time = self._sched_combo.currentData() == SCHEDULE_TIME
        self._time_widget.setVisible(is_time)
        self._interval_widget.setVisible(not is_time)

    def _sync_wp_list(self) -> None:
        self._wp_list.clear()
        for i, (r, c) in enumerate(self._mini_map.waypoints):
            self._wp_list.addItem(f"  {i+1}.  row {r}, col {c}")
        self._check_route()

    def _remove_waypoint(self) -> None:
        row = self._wp_list.currentRow()
        if 0 <= row < len(self._mini_map.waypoints):
            self._mini_map.waypoints.pop(row)
            self._mini_map.update()
            self._sync_wp_list()

    def _clear_waypoints(self) -> None:
        self._mini_map.waypoints.clear()
        self._mini_map.update()
        self._sync_wp_list()

    def _show_refusal(self, msg: str) -> None:
        self._status.setText(f"✕ {msg}")
        self._status.setStyleSheet("color: #f87171; font-size: 11px;")

    def _set_status(self, text: str, level: str) -> None:
        colour = {"ok": "#4ade80", "warn": "#fb923c", "bad": "#f87171",
                  "info": "#94a3b8"}[level]
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {colour}; font-size: 11px;")

    def _check_route(self) -> None:
        """Production: plan the route as the engine would right now and show
        whether it can be driven."""
        if self._engine is None:
            return
        m   = self._mini_map
        wps = list(m.waypoints)
        cart = self._engine.carts.get(self.cart_id)
        name = cart.name if cart else "the cart"
        path, bad, start = self._engine.preview_route(self.cart_id, wps)
        m.route, m.bad_index, m.start = path, bad, start
        m.update()
        self._route_ok = bad is None

        notice = self._engine.occupancy_notice()
        prefix = f"{notice}. " if notice else ""
        if not wps:
            self._set_status(prefix + "Click the live floor to add waypoints.",
                             "warn" if notice else "info")
            return
        if bad is not None:
            frm = f"{name}'s current position" if bad == 0 else f"waypoint {bad}"
            self._set_status(prefix + f"✕ Waypoint {bad + 1} can't be reached from {frm} "
                             f"with what is on the floor now — move it or clear the way.",
                             "bad")
            return
        busy = [(i + 1, what) for i, wp in enumerate(wps)
                if (what := m.cell_state(wp)) is not None]
        if busy:
            listed = ", ".join(f"{i} ({what})" for i, what in busy)
            self._set_status(prefix + f"⚠ Reachable, but these waypoints are occupied right now: "
                             f"{listed}. The cart will wait for them to clear.", "warn")
        elif start is None:
            self._set_status(prefix + f"✓ Waypoints connect. {name} isn't visible, so the way "
                             f"to waypoint 1 wasn't checked.", "warn")
        else:
            pts = [start] + path
            length = sum(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                         for a, b in zip(pts, pts[1:]))
            self._set_status(prefix + f"✓ Route reachable from {name}'s current position "
                             f"(≈ {length:.0f} cells long).", "warn" if notice else "ok")

    # ------------------------------------------------------------------
    # Populate / collect
    # ------------------------------------------------------------------

    def _populate(self, routine: Routine) -> None:
        self._name_edit.setText(routine.name)
        idx = self._sched_combo.findData(routine.schedule_type)
        if idx >= 0:
            self._sched_combo.setCurrentIndex(idx)
        if routine.time_of_day:
            t = QTime.fromString(routine.time_of_day, "HH:mm")
            self._time_edit.setTime(t)
        if routine.interval_seconds:
            self._interval_spin.setValue(routine.interval_seconds)
        self._repeat_check.setChecked(routine.repeat)
        self._enabled_check.setChecked(routine.enabled)
        self._mini_map.waypoints = list(routine.waypoints)
        self._mini_map.update()
        self._sync_wp_list()

    def _on_ok(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            self._name_edit.setPlaceholderText("Name is required!")
            return
        if not self._mini_map.waypoints:
            return  # need at least one waypoint
        if self._engine is not None:
            self._check_route()
            if not self._route_ok and QMessageBox.question(
                    self, "Route can't be driven now",
                    "With what is on the floor right now, part of this route can't be "
                    "reached (shown in red).\n\nSave it anyway? It will only run once the "
                    "way is clear.",
                    QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Save:
                return
        self.accept()

    def result_routine(self) -> Routine:
        """Call after exec() returned Accepted."""
        sched = self._sched_combo.currentData()
        if self._editing:
            r = self._editing
            r.name = self._name_edit.text().strip()
            r.schedule_type = sched
        else:
            from core.routine import Routine as R
            r = R.make(self.cart_id, self._name_edit.text().strip(), sched)

        r.waypoints = list(self._mini_map.waypoints)
        r.repeat = self._repeat_check.isChecked()
        r.enabled = self._enabled_check.isChecked()
        r.last_triggered = None  # reset so it fires fresh

        if sched == SCHEDULE_TIME:
            r.time_of_day = self._time_edit.time().toString("HH:mm")
            r.interval_seconds = None
        else:
            r.time_of_day = None
            r.interval_seconds = self._interval_spin.value()

        return r
