"""
Top-view grid canvas — dark blue theme, dynamically scaled to widget size.

Drawing pipeline:
  1. Numpy array → QImage for fast cell background (no per-cell Python loop).
  2. Grid lines drawn only when scale ≥ 4 px/cell to avoid visual clutter.
  3. All entities drawn in pixel space: positions = cell_coord × scale + offset.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRectF
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPen, QBrush, QMouseEvent, QImage,
)
from PyQt6.QtWidgets import QWidget

from config import GRID_ROWS, GRID_COLS
from core.cart import Cart, CartState
from core.grid import Grid
from simulation.sim_obstacles import SimObstacle


# ── Dark palette ──────────────────────────────────────────────────────────────
_C_CANVAS   = QColor("#060912")
_C_EMPTY    = (13, 18, 32)      # #0d1220 — numpy RGB tuple
_C_WALL     = (30, 58, 95)      # #1e3a5f — numpy RGB tuple
_C_GRID     = QColor("#131c30")
_C_OBSTACLE = QColor("#ef4444")
_C_OBS_RIM  = QColor("#fca5a5")
_C_WP       = QColor("#f472b6")
_C_WP_TEXT  = QColor("#ffffff")
_C_BLOCKED  = QColor("#ff3333")
_C_WAITING  = QColor("#fb923c")
_C_SEL_RING = QColor("#ffffff")


class MapCanvas(QWidget):
    waypoint_picked = pyqtSignal(int, int)   # (row, col)
    wall_toggled    = pyqtSignal(int, int)   # (row, col)

    def __init__(self, grid: Grid, parent=None):
        super().__init__(parent)
        self.grid = grid
        self.carts: Dict[str, Cart] = {}
        self.obstacles: List[SimObstacle] = []
        self.highlighted_cart_id: Optional[str] = None
        self.preview_waypoints: List[Tuple[int, int]] = []
        self._picking = False

        self.setMinimumSize(400, 225)
        self.setMouseTracking(True)
        self.setStyleSheet(f"background-color: {_C_CANVAS.name()};")

    # ── Public API ────────────────────────────────────────────────────────────

    def update_state(self, carts: Dict[str, Cart],
                     obstacles: List[SimObstacle]) -> None:
        self.carts = carts
        self.obstacles = obstacles
        self.update()

    def set_picking(self, on: bool) -> None:
        self._picking = on
        self.setCursor(
            Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor
        )

    def set_preview_waypoints(self, wps: List[Tuple[int, int]]) -> None:
        self.preview_waypoints = list(wps)
        self.update()

    def highlight_cart(self, cart_id: Optional[str]) -> None:
        self.highlighted_cart_id = cart_id
        self.update()

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        row, col = self._cell_at(event.pos())
        if row is None:
            return
        if event.button() == Qt.MouseButton.LeftButton and self._picking:
            self.waypoint_picked.emit(row, col)
        elif event.button() == Qt.MouseButton.RightButton:
            self.wall_toggled.emit(row, col)

    def _cell_at(self, pos: QPoint) -> Tuple[Optional[int], Optional[int]]:
        scale, ox, oy = self._get_viewport()
        col = int((pos.x() - ox) / scale)
        row = int((pos.y() - oy) / scale)
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            return row, col
        return None, None

    # ── Viewport ──────────────────────────────────────────────────────────────

    def _get_viewport(self) -> Tuple[float, float, float]:
        """(scale, off_x, off_y): maps 1 grid cell → scale pixels."""
        scale = min(self.width() / GRID_COLS, self.height() / GRID_ROWS)
        ox = (self.width()  - scale * GRID_COLS) / 2
        oy = (self.height() - scale * GRID_ROWS) / 2
        return scale, ox, oy

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale, ox, oy = self._get_viewport()
        self._draw_background(p, scale, ox, oy)
        self._draw_grid_lines(p, scale, ox, oy)
        self._draw_paths(p, scale, ox, oy)
        self._draw_obstacles(p, scale, ox, oy)
        self._draw_waypoints(p, scale, ox, oy)
        self._draw_carts(p, scale, ox, oy)
        p.end()

    # ── Background (numpy → QImage) ───────────────────────────────────────────

    def _draw_background(self, p: QPainter,
                         scale: float, ox: float, oy: float) -> None:
        cells = self.grid.cells()   # (GRID_ROWS, GRID_COLS) numpy int8
        rgb = np.empty((GRID_ROWS, GRID_COLS, 3), dtype=np.uint8)
        rgb[:] = _C_EMPTY
        rgb[cells == 1] = _C_WALL
        img = QImage(rgb.tobytes(), GRID_COLS, GRID_ROWS,
                     GRID_COLS * 3, QImage.Format.Format_RGB888)
        p.drawImage(
            QRectF(ox, oy, GRID_COLS * scale, GRID_ROWS * scale),
            img,
            QRectF(0, 0, GRID_COLS, GRID_ROWS),
        )

    def _draw_grid_lines(self, p: QPainter,
                         scale: float, ox: float, oy: float) -> None:
        if scale < 4:
            return
        p.setPen(QPen(_C_GRID, 0.5))
        W = GRID_COLS * scale
        H = GRID_ROWS * scale
        for c in range(GRID_COLS + 1):
            x = int(ox + c * scale)
            p.drawLine(x, int(oy), x, int(oy + H))
        for r in range(GRID_ROWS + 1):
            y = int(oy + r * scale)
            p.drawLine(int(ox), y, int(ox + W), y)

    # ── Paths ─────────────────────────────────────────────────────────────────

    def _draw_paths(self, p: QPainter,
                    scale: float, ox: float, oy: float) -> None:
        for cart in self.carts.values():
            if not cart.path or cart.path_index >= len(cart.path):
                continue
            color = QColor(cart.color)
            color.setAlpha(140)
            pen = QPen(color, max(1.0, scale * 0.3), Qt.PenStyle.DashLine)
            p.setPen(pen)
            ch, cw = cart.size
            sx, sy = self._cell_center(cart.row, cart.col, ch, cw, scale, ox, oy)
            for i in range(cart.path_index, len(cart.path)):
                nr, nc = cart.path[i]
                nx, ny = self._cell_center(nr, nc, ch, cw, scale, ox, oy)
                p.drawLine(int(sx), int(sy), int(nx), int(ny))
                sx, sy = nx, ny

    # ── Obstacles ─────────────────────────────────────────────────────────────

    def _draw_obstacles(self, p: QPainter,
                        scale: float, ox: float, oy: float) -> None:
        margin = max(1.0, scale * 0.12)
        for obs in self.obstacles:
            rx = ox + obs.col * scale + margin
            ry = oy + obs.row * scale + margin
            rw = obs.width  * scale - 2 * margin
            rh = obs.height * scale - 2 * margin
            p.setPen(QPen(_C_OBS_RIM, max(0.5, scale * 0.12)))
            p.setBrush(QBrush(_C_OBSTACLE))
            p.drawRoundedRect(QRectF(rx, ry, rw, rh), 2, 2)

    # ── Waypoints ─────────────────────────────────────────────────────────────

    def _draw_waypoints(self, p: QPainter,
                        scale: float, ox: float, oy: float) -> None:
        wps = self.preview_waypoints
        if not wps and self.highlighted_cart_id in self.carts:
            wps = self.carts[self.highlighted_cart_id].waypoints

        fs = max(6, int(scale * 0.7))
        p.setFont(QFont("Segoe UI", fs, QFont.Weight.Bold))
        rad = max(4.0, scale * 0.45)

        for i, (wr, wc) in enumerate(wps):
            cx = ox + (wc + 0.5) * scale
            cy = oy + (wr + 0.5) * scale
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_C_WP))
            p.drawEllipse(QRectF(cx - rad, cy - rad, 2 * rad, 2 * rad))
            p.setPen(QPen(_C_WP_TEXT))
            p.drawText(QRectF(cx - rad, cy - rad, 2 * rad, 2 * rad),
                       Qt.AlignmentFlag.AlignCenter, str(i + 1))

    # ── Carts ─────────────────────────────────────────────────────────────────

    def _draw_carts(self, p: QPainter,
                    scale: float, ox: float, oy: float) -> None:
        fs = max(6, int(scale * 0.9))
        p.setFont(QFont("Segoe UI", fs, QFont.Weight.Bold))
        margin = max(1.0, scale * 0.12)

        for cart in self.carts.values():
            ch, cw = cart.size
            rx = ox + cart.col * scale + margin
            ry = oy + cart.row * scale + margin
            rw = cw * scale - 2 * margin
            rh = ch * scale - 2 * margin

            color = QColor(cart.color)
            is_selected = cart.id == self.highlighted_cart_id
            border = _C_SEL_RING if is_selected else color.darker(160)
            pen_w = max(1.5, scale * 0.25) if is_selected else max(0.8, scale * 0.12)

            alpha = {
                CartState.MOVING:  220,
                CartState.WAITING: 140,
                CartState.BLOCKED: 100,
                CartState.IDLE:    160,
            }.get(cart.state, 200)
            color.setAlpha(alpha)

            p.setPen(QPen(border, pen_w))
            p.setBrush(QBrush(color))
            p.drawRoundedRect(QRectF(rx, ry, rw, rh), 3, 3)

            # State badge (top-right)
            badge_color, badge_char = self._state_badge(cart.state)
            if badge_char:
                br = max(3.0, scale * 0.35)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(badge_color))
                p.drawEllipse(QRectF(rx + rw - br - 1, ry + 1, br, br))

            # Name centred in rect
            p.setPen(QPen(_C_WP_TEXT))
            p.drawText(QRectF(rx, ry, rw, rh),
                       Qt.AlignmentFlag.AlignCenter, cart.name[:3])

    @staticmethod
    def _state_badge(state: CartState) -> Tuple[QColor, str]:
        return {
            CartState.WAITING: (_C_WAITING, "W"),
            CartState.BLOCKED: (_C_BLOCKED, "!"),
        }.get(state, (QColor(0, 0, 0, 0), ""))

    # ── Helper ────────────────────────────────────────────────────────────────

    def _cell_center(self, row: int, col: int,
                     h: int, w: int,
                     scale: float, ox: float, oy: float) -> Tuple[float, float]:
        x = ox + (col + w / 2) * scale
        y = oy + (row + h / 2) * scale
        return x, y
