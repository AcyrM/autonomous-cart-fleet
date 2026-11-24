"""Dialog to assign Bluetooth devices and ArUco marker IDs to carts."""

from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QSpinBox, QPushButton, QGroupBox,
    QProgressBar, QFrame, QListWidget,
)

from core.cart import Cart
from production.bt_controller import BTDevice, BTController


class _Scanner(QObject):
    """Worker that runs BT discovery in a background thread."""
    finished = pyqtSignal(list)  # list[BTDevice]

    def __init__(self, controller: BTController):
        super().__init__()
        self._ctrl = controller

    def run(self) -> None:
        # scan_async emits scan_result; we connect that to finished
        self._ctrl.scan_result.connect(self.finished)
        self._ctrl.scan_async()


class BTSelectorDialog(QDialog):
    def __init__(self, carts: Dict[str, Cart],
                 controller: BTController, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Bluetooth Devices & Markers")
        self.setMinimumWidth(480)
        self._carts      = carts
        self._controller = controller
        self._scanned:   List[BTDevice] = []

        # Combos and spinboxes keyed by cart_id
        self._dev_combos:    Dict[str, QComboBox] = {}
        self._marker_spins:  Dict[str, QSpinBox]  = {}

        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        lay.setContentsMargins(20, 18, 20, 18)

        title = QLabel("Bluetooth & Marker Assignment")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #60a5fa;")
        lay.addWidget(title)

        # ── Scan strip ────────────────────────────────────────────────────────
        scan_row = QHBoxLayout()
        self._scan_btn = QPushButton("🔍  Scan for Bluetooth devices")
        self._scan_btn.setFixedWidth(220)
        self._scan_btn.clicked.connect(self._start_scan)
        scan_row.addWidget(self._scan_btn)
        self._scan_progress = QProgressBar()
        self._scan_progress.setRange(0, 0)   # indeterminate
        self._scan_progress.setVisible(False)
        self._scan_progress.setFixedHeight(14)
        scan_row.addWidget(self._scan_progress, stretch=1)
        lay.addLayout(scan_row)

        self._scan_label = QLabel("Scan to discover nearby paired devices.")
        self._scan_label.setStyleSheet("color: #64748b; font-size: 10px;")
        lay.addWidget(self._scan_label)

        # Discovered devices — always visible, even when there are no carts
        self._dev_list = QListWidget()
        self._dev_list.setMaximumHeight(110)
        self._dev_list.setVisible(False)
        lay.addWidget(self._dev_list)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #1e3a5f;")
        lay.addWidget(sep)

        # ── Per-cart assignment ───────────────────────────────────────────────
        if self._carts:
            group = QGroupBox("Cart configuration")
            form = QFormLayout(group)
            form.setSpacing(8)

            for cart in sorted(self._carts.values(), key=lambda c: c.name):
                row_widget = QHBoxLayout()

                # BT device combo
                combo = QComboBox()
                combo.setMinimumWidth(200)
                combo.addItem("No device", None)
                existing = self._controller.device_for(cart.id)
                if existing:
                    combo.addItem(str(existing), existing)
                    combo.setCurrentIndex(1)
                self._dev_combos[cart.id] = combo
                row_widget.addWidget(combo, stretch=2)

                # Marker ID spinner
                spin = QSpinBox()
                spin.setRange(-1, 49)   # -1 = no marker, 0-49 for DICT_4X4_50
                spin.setSpecialValueText("No marker")
                spin.setValue(cart.marker_id if cart.marker_id is not None else -1)
                spin.setFixedWidth(100)
                spin.setToolTip("ArUco marker ID printed on top of this cart (0-49)")
                self._marker_spins[cart.id] = spin
                row_widget.addWidget(QLabel("Marker:"))
                row_widget.addWidget(spin)

                # Connection status indicator
                conn_lbl = QLabel("●")
                conn_lbl.setStyleSheet(
                    "color: #22d3ee;" if self._controller.is_connected(cart.id)
                    else "color: #475569;"
                )
                conn_lbl.setToolTip(
                    "Connected" if self._controller.is_connected(cart.id) else "Not connected"
                )
                row_widget.addWidget(conn_lbl)

                row_container = QGroupBox()
                row_container.setLayout(row_widget)
                row_container.setFlat(True)
                form.addRow(f"{cart.name}:", row_container)

            lay.addWidget(group)
        else:
            lay.addWidget(QLabel("  No carts added yet — add carts first."))

        # ── Bottom buttons ────────────────────────────────────────────────────
        note = QLabel(
            "After assigning devices, click 'Connect All' to open RFCOMM sockets.\n"
            "Marker IDs must match the ArUco DICT_4X4_50 marker on each cart."
        )
        note.setStyleSheet("color: #64748b; font-size: 10px;")
        lay.addWidget(note)

        btns = QHBoxLayout()
        connect_btn = QPushButton("🔗  Connect All")
        connect_btn.clicked.connect(self._connect_all)
        btns.addWidget(connect_btn)
        btns.addStretch()
        ok = QPushButton("Apply")
        ok.setDefault(True)
        ok.clicked.connect(self._apply)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        lay.addLayout(btns)

    # ── Scanning ──────────────────────────────────────────────────────────────

    def _start_scan(self) -> None:
        self._scan_btn.setEnabled(False)
        self._scan_progress.setVisible(True)
        self._scan_label.setText("Scanning… this takes ~8 seconds.")
        self._controller.scan_result.connect(self._on_scan_done)
        self._controller.scan_async()

    def _on_scan_done(self, devices: List[BTDevice]) -> None:
        try:
            self._controller.scan_result.disconnect(self._on_scan_done)
        except Exception:
            pass
        self._scanned = devices
        self._scan_progress.setVisible(False)
        self._scan_btn.setEnabled(True)
        self._dev_list.clear()
        for dev in devices:
            self._dev_list.addItem(str(dev))
        self._dev_list.setVisible(bool(devices))
        if devices:
            if self._dev_combos:
                self._scan_label.setText(
                    f"Found {len(devices)} device(s). Pick one for each cart in the dropdowns below.")
            else:
                self._scan_label.setText(
                    f"Found {len(devices)} device(s), but there are no carts to assign them to. "
                    "Add carts first, then reopen this dialog.")
            for combo in self._dev_combos.values():
                current = combo.currentData()
                # Keep existing items, add newly discovered ones
                existing_addrs = {combo.itemData(i).address
                                  for i in range(combo.count())
                                  if combo.itemData(i)}
                for dev in devices:
                    if dev.address not in existing_addrs:
                        combo.addItem(str(dev), dev)
                # Restore selection
                if current:
                    for i in range(combo.count()):
                        if combo.itemData(i) and combo.itemData(i).address == current.address:
                            combo.setCurrentIndex(i)
                            break
        else:
            self._scan_label.setText("No devices found. Make sure carts are powered on and paired.")

    # ── Apply / connect ───────────────────────────────────────────────────────

    def _connect_all(self) -> None:
        self._apply_assignments()
        for cart_id in self._carts:
            if self._controller.device_for(cart_id):
                self._controller.connect_cart(cart_id)

    def _apply(self) -> None:
        # Assigning a device means "use this cart": connect it too (a cart that
        # is assigned but not connected silently ignores every drive command)
        self._connect_all()
        self.accept()

    def _apply_assignments(self) -> None:
        for cart_id, combo in self._dev_combos.items():
            device: Optional[BTDevice] = combo.currentData()
            self._controller.assign(cart_id, device)

        for cart_id, spin in self._marker_spins.items():
            cart = self._carts.get(cart_id)
            if cart:
                val = spin.value()
                cart.marker_id = val if val >= 0 else None
