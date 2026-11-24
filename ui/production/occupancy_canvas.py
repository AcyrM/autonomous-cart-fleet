"""
Occupancy canvas: the production scene as a grid occupancy matrix.

Shows exactly what the planner works with — one colour per cell from
ProdEngine.occupancy_matrix() — with the carts, their headings and their
planned paths on top. Hovering a cell shows its (row, col) and state.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF
from PyQt6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QBrush
from PyQt6.QtWidgets import QWidget

from core.cart import Cart
from production.cart_detector import SRC_MEMORY, SRC_PREDICTED
from production.prod_engine import (
    OCC_CART, OCC_FREE, OCC_MARGIN, OCC_MOVER, OCC_OBSTACLE, OCC_PERSON, OCC_WALL,
)


# ── Colours ───────────────────────────────────────────────────────────────────

# code → (RGB, legend label)
_CELL_STYLE = {
    OCC_FREE:     ((13,  18,  32), "Free"),
    OCC_WALL:     ((30,  58,  95), "Wall"),
    OCC_MARGIN:   ((21,  31,  52), "Wall clearance"),
    OCC_OBSTACLE: ((124, 98, 214), "Obstacle"),
    OCC_MOVER:    ((234, 128, 44), "Moving object"),
    OCC_PERSON:   ((225, 54,  92), "Person"),
    OCC_CART:     ((14, 116, 144), "Cart"),
}
_LUT = np.zeros((256, 3), np.uint8)
for _code, (_rgb, _) in _CELL_STYLE.items():
    _LUT[_code] = _rgb

_C_CANVAS   = QColor("#060912")
_C_GRID     = QColor("#131c30")
_C_TEXT     = QColor("#94a3b8")
_C_WARN     = QColor("#fb923c")
_C_SEL_RING = QColor("#ffffff")


class OccupancyCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._occ: Optional[np.ndarray] = None
        self._carts: Dict[str, Cart] = {}
        self._poses: Dict[str, Tuple[float, float, float, str]] = {}
        self._paths: Dict[str, Tuple[List[Tuple[int, int]], int]] = {}
        self._radius = 2.0
        self._notice = ""
        self.highlighted_cart_id: Optional[str] = None
        self._hover: Optional[Tuple[int, int]] = None

        self.setMinimumSize(300, 200)
        self.setMouseTracking(True)
        self.setStyleSheet(f"background-color: {_C_CANVAS.name()};")

    # ── Public API ────────────────────────────────────────────────────────────

    def update_state(self, occ: np.ndarray,
                     carts: Dict[str, Cart],
                     poses: Dict[str, Tuple[float, float, float, str]],
                     paths: Dict[str, Tuple[List[Tuple[int, int]], int]],
                     radius_cells: float,
                     notice: str = "") -> None:
        self._occ    = occ
        self._carts  = carts
        self._poses  = poses
        self._paths  = paths
        self._radius = radius_cells
        self._notice = notice
        self.update()

    def highlight_cart(self, cart_id: Optional[str]) -> None:
        self.highlighted_cart_id = cart_id
        self.update()

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._hover = self._cell_at(event.pos())
        self.update()

    def leaveEvent(self, _) -> None:
        self._hover = None
        self.update()

    def _cell_at(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        if self._occ is None:
            return None
        scale, ox, oy = self._viewport()
        row = math.floor((pos.y() - oy) / scale)
        col = math.floor((pos.x() - ox) / scale)
        rows, cols = self._occ.shape
        return (row, col) if 0 <= row < rows and 0 <= col < cols else None

    # ── Viewport ──────────────────────────────────────────────────────────────

    def _viewport(self) -> Tuple[float, float, float]:
        """(scale, off_x, off_y): maps 1 grid cell → scale pixels."""
        rows, cols = self._occ.shape if self._occ is not None else (1, 1)
        scale = min(self.width() / cols, self.height() / rows)
        return scale, (self.width() - scale * cols) / 2, (self.height() - scale * rows) / 2

    def _cell_center(self, row: float, col: float) -> QPointF:
        scale, ox, oy = self._viewport()
        return QPointF(ox + (col + 0.5) * scale, oy + (row + 0.5) * scale)

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), _C_CANVAS)
        if self._occ is None:
            p.setPen(QPen(QColor("#1e3a5f")))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for camera frames…")
            p.end()
            return

        scale, ox, oy = self._viewport()
        self._draw_cells(p, scale, ox, oy)       # before antialiasing: crisp cells
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_grid_lines(p, scale, ox, oy)
        self._draw_paths(p, scale)
        self._draw_carts(p, scale)
        self._draw_legend(p)
        self._draw_hover(p, scale, ox, oy)
        if self._notice:
            p.setPen(QPen(_C_WARN))
            p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            p.drawText(QRectF(0, 6, self.width(), 20), Qt.AlignmentFlag.AlignHCenter,
                       self._notice)
        p.end()

    def _draw_cells(self, p: QPainter, scale: float, ox: float, oy: float) -> None:
        rows, cols = self._occ.shape
        rgb = _LUT[self._occ]
        img = QImage(rgb.tobytes(), cols, rows, cols * 3, QImage.Format.Format_RGB888)
        p.drawImage(QRectF(ox, oy, cols * scale, rows * scale), img,
                    QRectF(0, 0, cols, rows))

    def _draw_grid_lines(self, p: QPainter, scale: float, ox: float, oy: float) -> None:
        if scale < 6:
            return
        rows, cols = self._occ.shape
        p.setPen(QPen(_C_GRID, 0.5))
        for c in range(cols + 1):
            x = ox + c * scale
            p.drawLine(QPointF(x, oy), QPointF(x, oy + rows * scale))
        for r in range(rows + 1):
            y = oy + r * scale
            p.drawLine(QPointF(ox, y), QPointF(ox + cols * scale, y))

    def _draw_paths(self, p: QPainter, scale: float) -> None:
        for cid, (path, idx) in self._paths.items():
            cart = self._carts.get(cid)
            if cart is None or idx >= len(path):
                continue
            colour = QColor(cart.color)
            colour.setAlpha(200)
            pen = QPen(colour, max(1.5, scale * 0.35), Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            pose = self._poses.get(cid)
            pts = ([self._cell_center(pose[0], pose[1])] if pose else []) + \
                  [self._cell_center(r, c) for r, c in path[idx:]]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
            # Destination marker
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(colour))
            p.drawEllipse(pts[-1], max(3.0, scale * 0.8), max(3.0, scale * 0.8))

    def _draw_carts(self, p: QPainter, scale: float) -> None:
        body = max(4.0, self._radius * scale)
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        for cid, (row, col, heading, source) in self._poses.items():
            cart = self._carts.get(cid)
            if cart is None:
                continue
            centre = self._cell_center(row, col)
            colour = QColor(cart.color)
            weak   = source in (SRC_PREDICTED, SRC_MEMORY)
            p.setPen(QPen(colour, 2, Qt.PenStyle.DashLine if weak else Qt.PenStyle.SolidLine))
            fill = QColor(colour)
            fill.setAlpha(70)
            p.setBrush(QBrush(fill))
            p.drawEllipse(centre, body, body)
            if cid == self.highlighted_cart_id:
                p.setPen(QPen(_C_SEL_RING, 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(centre, body + 4, body + 4)

            # Heading (grid degrees: 0 = +col, 90 = −row)
            a   = math.radians(heading)
            tip = QPointF(centre.x() + 1.6 * body * math.cos(a),
                          centre.y() - 1.6 * body * math.sin(a))
            p.setPen(QPen(colour, 2.5))
            p.drawLine(centre, tip)
            for sign in (1, -1):
                ha = a + sign * math.radians(150)
                p.drawLine(tip, QPointF(tip.x() + 7 * math.cos(ha), tip.y() - 7 * math.sin(ha)))

            p.setPen(QPen(colour))
            p.drawText(QPointF(centre.x() + body + 4, centre.y() - body), cart.name[:12])

    def _draw_legend(self, p: QPainter) -> None:
        counts = np.bincount(self._occ.ravel(), minlength=len(_CELL_STYLE))
        floor  = max(1, int(self._occ.size - counts[OCC_WALL]))
        items  = [(code, rgb, label) for code, (rgb, label) in _CELL_STYLE.items()]
        row_h, w = 17, 190
        h  = len(items) * row_h + 26
        x0 = self.width() - w - 10
        y0 = self.height() - h - 10
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(6, 9, 18, 200)))
        p.drawRoundedRect(QRectF(x0, y0, w, h), 6, 6)
        p.setFont(QFont("Segoe UI", 8))
        for i, (code, rgb, label) in enumerate(items):
            y = y0 + 6 + i * row_h
            p.setPen(QPen(QColor(90, 110, 140)))
            p.setBrush(QBrush(QColor(*rgb)))
            p.drawRect(QRectF(x0 + 8, y + 3, 11, 11))
            p.setPen(QPen(_C_TEXT))
            p.drawText(QRectF(x0 + 26, y, w - 34, row_h),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
            if code != OCC_WALL:
                p.drawText(QRectF(x0 + 26, y, w - 34, row_h),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                           f"{100 * counts[code] / floor:.1f}%")
        blocked = int(counts[[OCC_OBSTACLE, OCC_MOVER, OCC_PERSON, OCC_CART]].sum())
        p.setPen(QPen(QColor("#e2e8f0")))
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p.drawText(QRectF(x0 + 8, y0 + h - 20, w - 16, 16),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   f"Floor occupied: {100 * blocked / floor:.1f}%")

    def _draw_hover(self, p: QPainter, scale: float, ox: float, oy: float) -> None:
        if self._hover is None:
            return
        row, col = self._hover
        p.setPen(QPen(QColor("#e2e8f0"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(ox + col * scale, oy + row * scale, scale, scale))
        label = _CELL_STYLE.get(int(self._occ[row, col]), ((0, 0, 0), "?"))[1]
        text  = f"  row {row}, col {col}  ·  {label}  "
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        tw = p.fontMetrics().horizontalAdvance(text)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(6, 9, 18, 210)))
        p.drawRoundedRect(QRectF(10, 10, tw, 22), 5, 5)
        p.setPen(QPen(QColor("#e2e8f0")))
        p.drawText(QRectF(10, 10, tw, 22), Qt.AlignmentFlag.AlignCenter, text)
