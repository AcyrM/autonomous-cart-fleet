"""
Left panel: cart list + routine list with add/edit/delete controls.
"""

from typing import Optional

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor, QIcon
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QFrame, QSizePolicy,
    QInputDialog, QMessageBox,
)

from core.cart import Cart, CartState
from core.routine import Routine
from core.routine_manager import RoutineManager


_STATE_ICONS = {
    CartState.IDLE:    "⬜",
    CartState.MOVING:  "🟢",
    CartState.WAITING: "🟡",
    CartState.BLOCKED: "🔴",
}


class CartPanel(QWidget):
    cart_selected    = pyqtSignal(str)         # cart_id
    cart_add_request = pyqtSignal(str)         # name
    cart_del_request = pyqtSignal(str)         # cart_id
    routine_add_request  = pyqtSignal(str)     # cart_id
    routine_edit_request = pyqtSignal(str)     # routine_id
    routine_del_request  = pyqtSignal(str)     # routine_id
    routine_run_request  = pyqtSignal(str)     # routine_id

    def __init__(self, routine_manager: RoutineManager, parent=None):
        super().__init__(parent)
        self.rm = routine_manager
        self._carts: dict[str, Cart] = {}
        self._selected_cart_id: Optional[str] = None
        self._selected_routine_id: Optional[str] = None
        self._build_ui()
        self.setFixedWidth(260)

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- Carts section ---
        layout.addWidget(self._section_label("CARTS"))

        self._cart_list = QListWidget()
        self._cart_list.setMaximumHeight(180)
        self._cart_list.currentItemChanged.connect(self._on_cart_selected)
        layout.addWidget(self._cart_list)

        cart_btns = QHBoxLayout()
        self._add_cart_btn = QPushButton("+ Cart")
        self._del_cart_btn = QPushButton("− Cart")
        self._del_cart_btn.setEnabled(False)
        self._add_cart_btn.clicked.connect(self._request_add_cart)
        self._del_cart_btn.clicked.connect(self._request_del_cart)
        cart_btns.addWidget(self._add_cart_btn)
        cart_btns.addWidget(self._del_cart_btn)
        layout.addLayout(cart_btns)

        layout.addWidget(self._divider())

        # --- Routines section ---
        layout.addWidget(self._section_label("ROUTINES"))

        self._routine_list = QListWidget()
        self._routine_list.currentItemChanged.connect(self._on_routine_selected)
        self._routine_list.itemDoubleClicked.connect(
            lambda _: self._request_edit_routine()
        )
        layout.addWidget(self._routine_list)

        r_btns = QHBoxLayout()
        self._add_routine_btn  = QPushButton("+ Routine")
        self._edit_routine_btn = QPushButton("Edit")
        self._del_routine_btn  = QPushButton("Delete")
        self._run_routine_btn  = QPushButton("▶ Now")

        for btn in (self._add_routine_btn, self._edit_routine_btn,
                    self._del_routine_btn, self._run_routine_btn):
            r_btns.addWidget(btn)

        self._add_routine_btn.clicked.connect(self._request_add_routine)
        self._edit_routine_btn.clicked.connect(self._request_edit_routine)
        self._del_routine_btn.clicked.connect(self._request_del_routine)
        self._run_routine_btn.clicked.connect(self._request_run_routine)

        self._set_routine_btns_enabled(False)
        layout.addLayout(r_btns)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Update from simulation
    # ------------------------------------------------------------------

    @property
    def selected_cart_id(self) -> Optional[str]:
        return self._selected_cart_id

    def refresh_carts(self, carts: dict) -> None:
        """Called each tick (or on structural change)."""
        selected_id = self._selected_cart_id
        self._carts = carts
        self._cart_list.blockSignals(True)
        self._cart_list.clear()
        for cart in sorted(carts.values(), key=lambda c: c.name):
            icon = _STATE_ICONS.get(cart.state, "⬜")
            item = QListWidgetItem(f"{icon} {cart.name}  [{cart.state.value}]")
            item.setData(Qt.ItemDataRole.UserRole, cart.id)
            item.setForeground(QColor(cart.color))
            self._cart_list.addItem(item)
            if cart.id == selected_id:
                self._cart_list.setCurrentItem(item)
        self._cart_list.blockSignals(False)

    def refresh_routines(self, cart_id: Optional[str] = None) -> None:
        cid = cart_id or self._selected_cart_id
        selected_rid = self._selected_routine_id
        self._routine_list.blockSignals(True)
        self._routine_list.clear()
        if cid:
            for r in self.rm.for_cart(cid):
                status = "✓" if r.enabled else "✗"
                sched = (r.time_of_day if r.time_of_day
                         else f"every {r.interval_seconds}s")
                text = f"{status} {r.name}  [{sched}]"
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, r.id)
                self._routine_list.addItem(item)
                if r.id == selected_rid:
                    self._routine_list.setCurrentItem(item)
        self._routine_list.blockSignals(False)
        self._set_routine_btns_enabled(self._routine_list.count() > 0)

    # ------------------------------------------------------------------
    # Signals → requests
    # ------------------------------------------------------------------

    def _on_cart_selected(self, item: QListWidgetItem) -> None:
        if item is None:
            self._selected_cart_id = None
            self._del_cart_btn.setEnabled(False)
            self.refresh_routines()
            return
        self._selected_cart_id = item.data(Qt.ItemDataRole.UserRole)
        self._del_cart_btn.setEnabled(True)
        self._add_routine_btn.setEnabled(True)
        self.refresh_routines()
        self.cart_selected.emit(self._selected_cart_id)

    def _on_routine_selected(self, item: QListWidgetItem) -> None:
        if item is None:
            self._selected_routine_id = None
            return
        self._selected_routine_id = item.data(Qt.ItemDataRole.UserRole)
        self._set_routine_btns_enabled(True)

    def _request_add_cart(self) -> None:
        name, ok = QInputDialog.getText(self, "New Cart", "Cart name:")
        if ok and name.strip():
            self.cart_add_request.emit(name.strip())

    def _request_del_cart(self) -> None:
        if self._selected_cart_id:
            cart = self._carts.get(self._selected_cart_id)
            name = cart.name if cart else "this cart"
            ans = QMessageBox.question(self, "Remove Cart",
                                       f"Remove '{name}' and all its routines?")
            if ans == QMessageBox.StandardButton.Yes:
                self.cart_del_request.emit(self._selected_cart_id)

    def _request_add_routine(self) -> None:
        if self._selected_cart_id:
            self.routine_add_request.emit(self._selected_cart_id)

    def _request_edit_routine(self) -> None:
        if self._selected_routine_id:
            self.routine_edit_request.emit(self._selected_routine_id)

    def _request_del_routine(self) -> None:
        if self._selected_routine_id:
            self.routine_del_request.emit(self._selected_routine_id)

    def _request_run_routine(self) -> None:
        if self._selected_routine_id:
            self.routine_run_request.emit(self._selected_routine_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_routine_btns_enabled(self, on: bool) -> None:
        for btn in (self._edit_routine_btn, self._del_routine_btn,
                    self._run_routine_btn):
            btn.setEnabled(on)

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: bold; color: #555; font-size: 11px;")
        return lbl

    @staticmethod
    def _divider() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet("color: #ccc;")
        return f
