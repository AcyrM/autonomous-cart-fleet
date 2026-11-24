"""
Production canvas: live camera feed + exclusion zone drawing + ArUco overlays.

Modes:
  Zone drawing  — Rect / Ellipse / Polygon / Erase (unchanged)
  Calibration   — click floor points, assign grid coords via popup dialog

Overlays drawn on top of the camera feed:
  • Exclusion zones (red)
  • Calibration points (cyan crosses + labels)
  • Cart detections: marker outline, orientation arrow, ID label — solid when
    the marker was read, dashed when tracked from the cart body / predicted
  • Camera-detected obstacles (violet outlines)
  • Cart paths: planned route projected from grid → pixel via inverse homography
  • Cart status HUD (bottom-left)

Layers (set_layers): the camera image and the analysis overlays can each be
hidden, giving a clean video view or the analysis alone on a dark backdrop.
Shapes being drawn (zones, routes, cart placement) always stay visible.
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QPointF, QRectF
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPen, QBrush,
    QMouseEvent, QKeyEvent, QImage, QPolygonF,
)
from PyQt6.QtWidgets import QWidget, QInputDialog, QMessageBox

from core.cart import Cart, CartState
from production.calibration import FloorCalibration
from production.cart_detector import Detection, SRC_BLOB, SRC_MEMORY, SRC_PREDICTED
from production.obstacle_detector import OB_MOVING, OB_REMEMBERED, Obstacle
from production.person_detector import PersonDetection


# ── Draw mode enum ────────────────────────────────────────────────────────────

class DrawMode(Enum):
    NONE        = auto()
    RECT        = auto()
    ELLIPSE     = auto()
    POLYGON     = auto()
    ERASE       = auto()
    CALIBRATE   = auto()   # click floor points to register grid coordinates
    ROUTE       = auto()   # click waypoints for the selected cart's route
    PLACE_CART  = auto()   # virtual mode: click (and drag to aim) to put a cart down
    PLACE_OBST  = auto()   # virtual mode: click to put an obstacle down, right-click removes


# ── Zone dataclass ────────────────────────────────────────────────────────────

@dataclass
class ExclusionZone:
    shape: str
    pts: List[Tuple[float, float]]


# ── Colours ───────────────────────────────────────────────────────────────────

_C_ZONE_FILL    = QColor(239,  68,  68,  70)
_C_ZONE_BORDER  = QColor(239,  68,  68, 210)
_C_PREV_FILL    = QColor(251, 191,  36,  80)
_C_PREV_BORDER  = QColor(251, 191,  36, 200)
_C_VERTEX       = QColor(251, 191,  36, 255)
_C_CALIB_PT     = QColor( 34, 211, 238, 255)   # cyan
_C_CALIB_DONE   = QColor( 74, 222, 128, 255)   # green when calibrated
_C_DETECT_BOX   = QColor( 34, 211, 238, 200)
_C_DETECT_ARROW = QColor( 34, 211, 238, 255)
_C_PATH         = QColor(251, 191,  36, 180)
_C_PERSON_FILL  = QColor(251, 146,  60,  55)   # orange, semi-transparent
_C_PERSON_EDGE  = QColor(251, 146,  60, 220)   # orange border
_C_PERSON_LABEL = QColor(254, 215, 170, 255)   # light orange text
_C_OBST_FILL    = QColor(167, 139, 250,  60)   # violet, semi-transparent
_C_OBST_EDGE    = QColor(167, 139, 250, 230)
_C_MOVER_FILL   = QColor(251, 146,  60,  60)   # orange — moving object (person…)
_C_MOVER_EDGE   = QColor(251, 146,  60, 230)
_C_MOVER_SWEEP  = QColor(251, 146,  60,  25)
_C_DETECT_WEAK  = QColor(148, 163, 184, 220)   # slate — predicted pose
_C_NO_FEED      = QColor("#060912")
_C_NO_FEED_TEXT = QColor("#1e3a5f")
_C_BACKDROP     = QColor("#0b1220")   # camera area when the video layer is hidden
_C_BACKDROP_RIM = QColor("#1e3a5f")


class ProdCanvas(QWidget):
    waypoint_picked    = pyqtSignal(int, int)
    wall_toggled       = pyqtSignal(int, int)
    calib_point_added  = pyqtSignal(float, float, int, int)  # px, py, row, col
    calib_point_removed = pyqtSignal()                       # last point undone
    route_finished     = pyqtSignal(list)                    # [(px, py), …] image px
    cart_place_requested     = pyqtSignal(float, float, float)  # px, py, heading (nan = any)
    obstacle_place_requested = pyqtSignal(float, float)
    obstacle_remove_requested = pyqtSignal(float, float)

    def __init__(self, calibration: Optional[FloorCalibration] = None, parent=None):
        super().__init__(parent)
        self._frame: Optional[QImage] = None
        self.carts: Dict[str, Cart] = {}
        self.detections: List[Detection] = []
        self.person_detections: List[PersonDetection] = []
        self.obstacles: List[Obstacle] = []
        self.highlighted_cart_id: Optional[str] = None
        self._calib: Optional[FloorCalibration] = calibration
        self._grid_size: Tuple[int, int] = (9999, 9999)   # (rows, cols) for calib input

        # Planned paths: cart_id → list of (row, col) grid waypoints
        self._paths: Dict[str, List[Tuple[int, int]]] = {}

        self._draw_mode: DrawMode = DrawMode.NONE
        self._zones: List[ExclusionZone] = []
        self._wip_pts: List[Tuple[float, float]] = []
        self._anchor: Optional[Tuple[float, float]] = None
        self._hover: Optional[Tuple[float, float]] = None
        self._route_pts: List[Tuple[float, float]] = []       # image px, being drawn
        self._place_from: Optional[Tuple[float, float]] = None  # cart placement drag
        # Routes to show (e.g. the selected cart's): [(colour, [(px, py), …]), …]
        self.route_overlay: List[Tuple[str, List[Tuple[float, float]]]] = []
        self.show_video    = True
        self.show_analysis = True

        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setStyleSheet(f"background-color: {_C_NO_FEED.name()};")

    # ── Public API ────────────────────────────────────────────────────────────

    def on_frame(self, img: QImage) -> None:
        self._frame = img
        self.update()

    def update_state(self, carts: Dict[str, Cart],
                     detections: List[Detection],
                     paths: Optional[Dict[str, List[Tuple[int, int]]]] = None,
                     person_detections: Optional[List[PersonDetection]] = None,
                     obstacles: Optional[List[Obstacle]] = None) -> None:
        self.carts            = carts
        self.detections       = detections
        self.person_detections = person_detections or []
        self.obstacles        = obstacles or []
        if paths is not None:
            self._paths = paths
        self.update()

    def highlight_cart(self, cart_id: Optional[str]) -> None:
        self.highlighted_cart_id = cart_id
        self.update()

    def set_layers(self, video: bool, analysis: bool) -> None:
        self.show_video    = video
        self.show_analysis = analysis
        self.update()

    def set_calibration(self, calib: FloorCalibration) -> None:
        self._calib = calib

    def set_grid_size(self, rows: int, cols: int) -> None:
        """Limits for the grid coordinates typed in for calibration points."""
        self._grid_size = (rows, cols)

    def set_draw_mode(self, mode: DrawMode) -> None:
        self._cancel_wip()
        self._draw_mode = mode
        cursor = {
            DrawMode.ROUTE:      Qt.CursorShape.CrossCursor,
            DrawMode.PLACE_CART: Qt.CursorShape.CrossCursor,
            DrawMode.PLACE_OBST: Qt.CursorShape.CrossCursor,
            DrawMode.RECT:      Qt.CursorShape.CrossCursor,
            DrawMode.ELLIPSE:   Qt.CursorShape.CrossCursor,
            DrawMode.POLYGON:   Qt.CursorShape.CrossCursor,
            DrawMode.ERASE:     Qt.CursorShape.PointingHandCursor,
            DrawMode.CALIBRATE: Qt.CursorShape.CrossCursor,
        }.get(mode, Qt.CursorShape.ArrowCursor)
        self.setCursor(cursor)
        self.update()

    def current_draw_mode(self) -> DrawMode:
        return self._draw_mode

    def clear_zones(self) -> None:
        self._zones.clear()
        self._cancel_wip()
        self.update()

    def zones(self) -> List[ExclusionZone]:
        return list(self._zones)

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        fx, fy = self._to_frac(event.pos())
        ix, iy = self._to_image(event.pos())
        right  = event.button() == Qt.MouseButton.RightButton

        if self._draw_mode == DrawMode.ROUTE:
            if right and self._route_pts:
                self._route_pts.pop()                 # undo last waypoint
            elif not right:
                self._route_pts.append((ix, iy))
            self.update()
            return
        if self._draw_mode == DrawMode.PLACE_OBST:
            (self.obstacle_remove_requested if right
             else self.obstacle_place_requested).emit(ix, iy)
            return
        if self._draw_mode == DrawMode.PLACE_CART and not right:
            self._place_from = (ix, iy)
            return

        if right and self._draw_mode == DrawMode.CALIBRATE:
            if self._calib and self._calib.point_count:
                self._calib.remove_last()           # undo a mistyped point
                self._calib.save()
                self.calib_point_removed.emit()
                self.update()
            return

        if right:
            self._cancel_wip()
            self.update()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self._draw_mode == DrawMode.CALIBRATE:
            self._handle_calib_click(event.pos())
            return

        if self._draw_mode in (DrawMode.RECT, DrawMode.ELLIPSE):
            self._anchor = (fx, fy)
        elif self._draw_mode == DrawMode.POLYGON:
            self._wip_pts.append((fx, fy))
        elif self._draw_mode == DrawMode.ERASE:
            self._erase_at(fx, fy)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._hover = self._to_frac(event.pos())
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._draw_mode == DrawMode.PLACE_CART and self._place_from is not None:
            (x0, y0), (x1, y1) = self._place_from, self._to_image(event.pos())
            self._place_from = None
            # Drag to aim: the cart will face from the press point toward the release
            heading = math.degrees(math.atan2(-(y1 - y0), x1 - x0)) \
                if math.hypot(x1 - x0, y1 - y0) > 8 else float("nan")
            self.cart_place_requested.emit(x0, y0, heading)
            self.update()
            return
        if self._draw_mode in (DrawMode.RECT, DrawMode.ELLIPSE) \
                and self._anchor is not None:
            fx, fy = self._to_frac(event.pos())
            ax, ay = self._anchor
            if abs(fx - ax) > 0.005 or abs(fy - ay) > 0.005:
                self._zones.append(ExclusionZone(
                    "rect" if self._draw_mode == DrawMode.RECT else "ellipse",
                    [(ax, ay), (fx, fy)],
                ))
            self._anchor = None
            self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._draw_mode == DrawMode.ROUTE:
            self._finish_route()
            return
        if self._draw_mode == DrawMode.POLYGON and len(self._wip_pts) >= 3:
            pts = self._wip_pts[:-1] if len(self._wip_pts) > 3 else self._wip_pts
            self._zones.append(ExclusionZone("poly", list(pts)))
            self._wip_pts.clear()
            self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._draw_mode == DrawMode.ROUTE and \
                event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish_route()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._cancel_wip()
            self.update()
        elif event.key() == Qt.Key.Key_Return and \
                self._draw_mode == DrawMode.POLYGON and len(self._wip_pts) >= 3:
            self._zones.append(ExclusionZone("poly", list(self._wip_pts)))
            self._wip_pts.clear()
            self.update()

    # ── Calibration click handler ─────────────────────────────────────────────

    def _handle_calib_click(self, pos: QPoint) -> None:
        if self._frame is None:
            return                                   # no camera image to click on
        ox, oy, sc, fw, fh = self._view()
        px = (pos.x() - ox) / sc
        py = (pos.y() - oy) / sc
        if not (0 <= px < fw and 0 <= py < fh):
            return                                   # outside the camera image

        rows, cols = self._grid_size
        n = self._calib.point_count if self._calib else 0
        # Points 1–4 default to the grid corners, clockwise from top-left: click
        # the floor's corners in that order and just press Enter twice each
        corners = [(0, 0), (0, cols - 1), (rows - 1, cols - 1), (rows - 1, 0)]
        d_row, d_col = corners[n] if n < 4 else (rows // 2, cols // 2)
        hint = ["top-left", "top-right", "bottom-right", "bottom-left"][n] \
            if n < 4 else "extra point"
        row, ok1 = QInputDialog.getInt(
            self, f"Calibration Point {n + 1} ({hint})",
            f"Grid row for this point (0 – {rows - 1}):", d_row, 0, rows - 1)
        if not ok1:
            return
        col, ok2 = QInputDialog.getInt(
            self, f"Calibration Point {n + 1} ({hint})",
            f"Grid column for this point (0 – {cols - 1}):", d_col, 0, cols - 1)
        if not ok2:
            return

        if self._calib:
            self._calib.add_point(px, py, row, col)
            self._calib.save()                       # partial sets survive a restart
        self.calib_point_added.emit(px, py, row, col)
        self.update()

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _finish_route(self) -> None:
        # A double-click also delivers a press: drop the duplicate last point
        pts = list(self._route_pts)
        if len(pts) >= 2 and math.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]) < 4:
            pts.pop()
        self._route_pts.clear()
        self.update()
        if pts:
            self.route_finished.emit(pts)

    def _to_image(self, pos: QPoint) -> Tuple[float, float]:
        ox, oy, sc = self._view()[:3]
        return (pos.x() - ox) / sc, (pos.y() - oy) / sc

    def _view(self) -> Tuple[float, float, float, float, float]:
        """Where the camera image sits in the widget, aspect ratio preserved:
        (offset x, offset y, scale, image width, image height)."""
        fw = self._frame.width()  if self._frame else self.width()
        fh = self._frame.height() if self._frame else self.height()
        sc = min(self.width() / max(fw, 1), self.height() / max(fh, 1))
        return ((self.width() - fw * sc) / 2, (self.height() - fh * sc) / 2,
                sc, float(fw), float(fh))

    def _to_frac(self, pos: QPoint) -> Tuple[float, float]:
        ox, oy, sc, fw, fh = self._view()
        return (pos.x() - ox) / max(1.0, fw * sc), (pos.y() - oy) / max(1.0, fh * sc)

    def _to_px(self, fx: float, fy: float) -> Tuple[float, float]:
        ox, oy, sc, fw, fh = self._view()
        return ox + fx * fw * sc, oy + fy * fh * sc

    def _frac_to_qptf(self, pts: List[Tuple[float, float]]) -> List[QPointF]:
        return [QPointF(*self._to_px(x, y)) for x, y in pts]

    def _grid_to_widget(self, row: float, col: float) -> Optional[Tuple[float, float]]:
        """Convert grid coords to widget pixel coords via calibration."""
        if self._calib is None or not self._calib.is_calibrated:
            return None
        cam_px, cam_py = self._calib.grid_to_pixel(row, col)
        # Scale from camera resolution to widget size
        if self._frame:
            ox, oy, sc = self._view()[:3]
            wx, wy = ox + cam_px * sc, oy + cam_py * sc
        else:
            wx, wy = cam_px, cam_py
        return (wx, wy)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cancel_wip(self) -> None:
        self._route_pts.clear()
        self._place_from = None
        self._wip_pts.clear()
        self._anchor = None

    def _erase_at(self, fx: float, fy: float) -> None:
        px, py = self._to_px(fx, fy)
        for i in range(len(self._zones) - 1, -1, -1):
            if self._zone_hit(self._zones[i], px, py):
                self._zones.pop(i)
                return

    def _zone_hit(self, zone: ExclusionZone, px: float, py: float) -> bool:
        if zone.shape == "rect" and len(zone.pts) == 2:
            x0, y0 = self._to_px(*zone.pts[0])
            x1, y1 = self._to_px(*zone.pts[1])
            return (min(x0, x1) <= px <= max(x0, x1) and
                    min(y0, y1) <= py <= max(y0, y1))
        if zone.shape == "ellipse" and len(zone.pts) == 2:
            x0, y0 = self._to_px(*zone.pts[0])
            x1, y1 = self._to_px(*zone.pts[1])
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx = abs(x1 - x0) / 2 or 1
            ry = abs(y1 - y0) / 2 or 1
            return ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2 <= 1
        if zone.shape == "poly" and len(zone.pts) >= 3:
            poly = QPolygonF(self._frac_to_qptf(zone.pts))
            return poly.containsPoint(QPointF(px, py), Qt.FillRule.OddEvenFill)
        return False

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._frame:
            # Keep the camera's aspect ratio (black bars), never stretch
            ox, oy, sc, fw, fh = self._view()
            p.fillRect(0, 0, self.width(), self.height(), QColor(0, 0, 0))
            if self.show_video:
                p.drawImage(QRectF(ox, oy, fw * sc, fh * sc), self._frame)
            else:
                # Same geometry as the image, so overlays land where they would
                p.setPen(QPen(_C_BACKDROP_RIM, 1))
                p.setBrush(QBrush(_C_BACKDROP))
                p.drawRect(QRectF(ox, oy, fw * sc, fh * sc))
        else:
            p.fillRect(0, 0, self.width(), self.height(), _C_NO_FEED)
            p.setPen(QPen(_C_NO_FEED_TEXT))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "No camera feed\nCheck connection and restart")

        if self.show_analysis:
            self._paint_zones(p)
        self._paint_preview(p)
        if self.show_analysis:
            self._paint_obstacles(p)
            self._paint_paths(p)
            self._paint_person_detections(p)
            self._paint_detections(p)
        if self.show_analysis or self._draw_mode == DrawMode.CALIBRATE:
            self._paint_calib_points(p)              # always visible while calibrating
        self._paint_routes(p, overlays=self.show_analysis)
        if self.show_analysis:
            self._paint_cart_status(p)
        p.end()

    # ── Routes (being drawn / of the selected cart) and cart placement ────────

    def _paint_routes(self, p: QPainter, overlays: bool = True) -> None:
        ox, oy, sc, fw, fh = self._view()
        hover = None if self._hover is None else (self._hover[0] * fw, self._hover[1] * fh)

        def draw(pts, colour: QColor, dashed: bool):
            w = [QPointF(ox + x * sc, oy + y * sc) for x, y in pts]
            pen = QPen(colour, 2, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            for a, b in zip(w, w[1:]):
                p.drawLine(a, b)
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            for i, pt in enumerate(w):
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(colour))
                p.drawEllipse(pt, 9, 9)
                p.setPen(QPen(QColor("#0b1020")))
                p.drawText(QRectF(pt.x() - 9, pt.y() - 9, 18, 18),
                           Qt.AlignmentFlag.AlignCenter, str(i + 1))

        for colour, pts in (self.route_overlay if overlays else []):
            if pts:
                draw(pts, QColor(colour), dashed=True)
        if self._route_pts:
            if hover is not None:                      # rubber band to the cursor
                (ax, ay), (bx, by) = self._route_pts[-1], hover
                p.setPen(QPen(QColor(250, 204, 21, 120), 1, Qt.PenStyle.DotLine))
                p.drawLine(QPointF(ox + ax * sc, oy + ay * sc), QPointF(ox + bx * sc, oy + by * sc))
            draw(self._route_pts, QColor(250, 204, 21), dashed=False)
        if self._draw_mode == DrawMode.PLACE_CART and self._place_from is not None \
                and hover is not None:                 # aiming arrow while dragging
            (x0, y0), (x1, y1) = self._place_from, hover
            a = QPointF(ox + x0 * sc, oy + y0 * sc)
            p.setPen(QPen(QColor(34, 211, 238), 3))
            p.drawLine(a, QPointF(ox + x1 * sc, oy + y1 * sc))
            p.drawEllipse(a, 6, 6)

    # ── Zone painting ─────────────────────────────────────────────────────────

    def _paint_zones(self, p: QPainter) -> None:
        for zone in self._zones:
            self._paint_zone(p, zone, _C_ZONE_FILL, _C_ZONE_BORDER, solid=True)

    def _paint_zone(self, p: QPainter, zone: ExclusionZone,
                    fill: QColor, border: QColor, solid: bool = True) -> None:
        pen_style = Qt.PenStyle.SolidLine if solid else Qt.PenStyle.DashLine
        p.setPen(QPen(border, 1.5, pen_style))
        p.setBrush(QBrush(fill))
        if zone.shape == "rect" and len(zone.pts) == 2:
            x0, y0 = self._to_px(*zone.pts[0])
            x1, y1 = self._to_px(*zone.pts[1])
            p.drawRect(QRectF(min(x0, x1), min(y0, y1),
                              abs(x1 - x0), abs(y1 - y0)))
        elif zone.shape == "ellipse" and len(zone.pts) == 2:
            x0, y0 = self._to_px(*zone.pts[0])
            x1, y1 = self._to_px(*zone.pts[1])
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
            p.drawEllipse(QRectF(cx - rx, cy - ry, 2 * rx, 2 * ry))
        elif zone.shape == "poly" and len(zone.pts) >= 3:
            p.drawPolygon(QPolygonF(self._frac_to_qptf(zone.pts)))

    # ── Preview while drawing ─────────────────────────────────────────────────

    def _paint_preview(self, p: QPainter) -> None:
        hover = self._hover
        if self._draw_mode in (DrawMode.RECT, DrawMode.ELLIPSE) \
                and self._anchor is not None and hover is not None:
            preview = ExclusionZone(
                "rect" if self._draw_mode == DrawMode.RECT else "ellipse",
                [self._anchor, hover],
            )
            self._paint_zone(p, preview, _C_PREV_FILL, _C_PREV_BORDER, solid=False)
        elif self._draw_mode == DrawMode.POLYGON and self._wip_pts:
            pts_px = self._frac_to_qptf(self._wip_pts)
            if hover:
                pts_px.append(QPointF(*self._to_px(*hover)))
            p.setPen(QPen(_C_PREV_BORDER, 1.5, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(pts_px) - 1):
                p.drawLine(pts_px[i], pts_px[i + 1])
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_C_VERTEX))
            for pt in pts_px[:-1]:
                p.drawEllipse(pt, 4, 4)
            if len(pts_px) >= 3:
                close_pen = QPen(QColor(251, 191, 36, 100), 1.0, Qt.PenStyle.DotLine)
                p.setPen(close_pen)
                p.drawLine(pts_px[-1], pts_px[0])

    # ── Path overlay (grid coords → widget pixels via homography) ─────────────

    def _paint_paths(self, p: QPainter) -> None:
        if not self._calib or not self._calib.is_calibrated:
            return
        pen = QPen(_C_PATH, 2, Qt.PenStyle.DashLine)
        pen.setDashPattern([6, 3])
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for path in self._paths.values():
            pts = [self._grid_to_widget(r, c) for r, c in path if
                   self._grid_to_widget(r, c) is not None]
            for i in range(len(pts) - 1):
                if pts[i] and pts[i + 1]:
                    p.drawLine(QPointF(*pts[i]), QPointF(*pts[i + 1]))

    # ── Obstacle overlay ──────────────────────────────────────────────────────

    def _paint_obstacles(self, p: QPainter) -> None:
        if not self.obstacles:
            return
        ox, oy, sc = self._view()[:3]
        def poly(pts):
            return QPolygonF([QPointF(ox + x * sc, oy + y * sc) for x, y in pts])
        for ob in self.obstacles:
            if ob.state == OB_MOVING:
                if ob.swept is not None:          # where it will be shortly
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QBrush(_C_MOVER_SWEEP))
                    p.drawPolygon(poly(ob.swept))
                p.setPen(QPen(_C_MOVER_EDGE, 1.5))
                p.setBrush(QBrush(_C_MOVER_FILL))
            elif ob.state == OB_REMEMBERED:       # not seen, but can't have left
                p.setPen(QPen(_C_OBST_EDGE, 1.5, Qt.PenStyle.DashLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
            else:
                p.setPen(QPen(_C_OBST_EDGE, 1.5))
                p.setBrush(QBrush(_C_OBST_FILL))
            p.drawPolygon(poly(ob.contour))

    # ── Person detection overlay ──────────────────────────────────────────────

    def _paint_person_detections(self, p: QPainter) -> None:
        if not self.person_detections:
            return
        ox, oy, sc = self._view()[:3]

        p.setPen(QPen(_C_PERSON_EDGE, 2))
        p.setBrush(QBrush(_C_PERSON_FILL))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

        for det in self.person_detections:
            x1, y1 = ox + det.x1 * sc, oy + det.y1 * sc
            x2, y2 = ox + det.x2 * sc, oy + det.y2 * sc
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            p.drawRect(rect)

            # Label above the box: "PERSON 0.87"
            label = f"PERSON  {det.confidence:.2f}"
            lw = max(x2 - x1, 80)
            label_rect = QRectF(x1, max(0.0, y1 - 18), lw, 18)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_C_PERSON_EDGE))
            p.drawRect(label_rect)
            p.setPen(QPen(_C_PERSON_LABEL))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawText(label_rect,
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       f"  {label}")
            p.setPen(QPen(_C_PERSON_EDGE, 2))
            p.setBrush(QBrush(_C_PERSON_FILL))

    # ── ArUco detection overlay ───────────────────────────────────────────────

    def _paint_detections(self, p: QPainter) -> None:
        if not self.detections:
            return
        # Scale factors from camera coords to widget coords
        ox, oy, sc = self._view()[:3]
        # Arrows show the direction of travel: marker angle minus how the
        # marker is mounted on the cart (measured by heading calibration)
        offsets = {c.marker_id: c.heading_offset for c in self.carts.values()
                   if c.marker_id is not None}

        for d in self.detections:
            cx = ox + d.cx * sc
            cy = oy + d.cy * sc
            # Scale corners
            corners = [(ox + pt[0] * sc, oy + pt[1] * sc) for pt in d.corners]
            poly_pts = [QPointF(x, y) for x, y in corners]

            # Solid = marker read this frame; dashed = body blob / prediction
            colour = _C_DETECT_WEAK if d.source in (SRC_PREDICTED, SRC_MEMORY) else _C_DETECT_BOX
            style  = Qt.PenStyle.DotLine if d.source == SRC_MEMORY else \
                Qt.PenStyle.DashLine if d.source in (SRC_BLOB, SRC_PREDICTED) \
                else Qt.PenStyle.SolidLine
            p.setPen(QPen(colour, 2, style))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolygon(QPolygonF(poly_pts + [poly_pts[0]]))

            # Orientation arrow (direction of travel)
            r = math.radians(d.angle_deg - offsets.get(d.marker_id, 0.0))
            arrow_len = 30
            ex = cx + arrow_len * math.cos(r)
            ey = cy - arrow_len * math.sin(r)
            p.setPen(QPen(_C_DETECT_ARROW, 2))
            p.drawLine(QPointF(cx, cy), QPointF(ex, ey))
            # Arrowhead
            head_len = 8
            for sign in (1, -1):
                ha = r + sign * math.radians(150)
                p.drawLine(QPointF(ex, ey),
                           QPointF(ex + head_len * math.cos(ha),
                                   ey - head_len * math.sin(ha)))

            # Marker ID label
            p.setPen(QPen(_C_DETECT_BOX))
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            p.drawText(QPointF(cx + 10, cy - 10), f"#{d.marker_id}")

    # ── Calibration point overlay ─────────────────────────────────────────────

    def _paint_calib_points(self, p: QPainter) -> None:
        if self._calib is None:
            return
        pts = self._calib.calibration_points()
        if not pts:
            return

        ox, oy, sc = self._view()[:3]

        colour = _C_CALIB_DONE if self._calib.is_calibrated else _C_CALIB_PT
        if self._calib.problem or self._calib.warning:
            colour = QColor("#fb923c")               # points need attention
        p.setPen(QPen(colour, 2))
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        for i, (px, py, row, col) in enumerate(pts):
            wx, wy = ox + px * sc, oy + py * sc
            r = 6
            p.drawLine(QPointF(wx - r, wy), QPointF(wx + r, wy))
            p.drawLine(QPointF(wx, wy - r), QPointF(wx, wy + r))
            p.drawText(QPointF(wx + 8, wy - 4), f"{i+1} ({row},{col})")

    # ── Cart status HUD (bottom-left) ─────────────────────────────────────────

    def _paint_cart_status(self, p: QPainter) -> None:
        if not self.carts:
            return
        margin, row_h, col_w = 10, 22, 140
        total_h = len(self.carts) * row_h + 10
        bx = margin
        by = self.height() - margin - total_h
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(6, 9, 18, 180)))
        p.drawRoundedRect(QRectF(bx, by, col_w + 10, total_h), 6, 6)

        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        state_sym = {CartState.IDLE: "·", CartState.MOVING: "▶",
                     CartState.WAITING: "⏳", CartState.BLOCKED: "✕"}
        state_col = {CartState.IDLE:    QColor("#94a3b8"),
                     CartState.MOVING:  QColor("#22d3ee"),
                     CartState.WAITING: QColor("#fb923c"),
                     CartState.BLOCKED: QColor("#ff3333")}
        for i, cart in enumerate(sorted(self.carts.values(), key=lambda c: c.name)):
            ry = by + 5 + i * row_h
            sym = state_sym.get(cart.state, "?")
            col = state_col.get(cart.state, QColor("#ffffff"))
            p.setPen(QPen(QColor(cart.color)))
            p.drawText(QRectF(bx + 6, ry, col_w - 6, row_h),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       cart.name[:12])
            p.setPen(QPen(col))
            p.drawText(QRectF(bx + 6, ry, col_w - 6, row_h),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                       sym)
