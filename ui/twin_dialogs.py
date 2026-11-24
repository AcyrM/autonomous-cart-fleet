"""Dialogs for the digital twin (virtual production mode)."""

from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel,
    QSpinBox, QVBoxLayout,
)

DRIVE_REF = 180      # the engine's cruise DRIVE value; speeds are shown at this command


class CartSetupDialog(QDialog):
    """Physics and marker mounting of one simulated cart. In the twin 1 px of
    the recorded image ≈ 1 mm, so speeds are shown in mm/s."""

    def __init__(self, name: str, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Cart set-up — {name}")
        self._params = dict(params)
        lay  = QVBoxLayout(self)
        form = QFormLayout()

        self._speed = QSpinBox()
        self._speed.setRange(30, 800)
        self._speed.setSuffix(" mm/s")
        self._speed.setValue(int(round(params["px_per_unit"] * DRIVE_REF)))
        form.addRow(f"Speed at DRIVE {DRIVE_REF}:", self._speed)

        self._tau = QSpinBox()
        self._tau.setRange(10, 1000)
        self._tau.setSuffix(" ms")
        self._tau.setValue(int(round(params["motor_tau"] * 1000)))
        form.addRow("Motor response time:", self._tau)

        self._dead = QSpinBox()
        self._dead.setRange(0, 200)
        self._dead.setValue(int(params["deadband"]))
        form.addRow("Motor dead band (DRIVE):", self._dead)

        self._gl = QDoubleSpinBox()
        self._gr = QDoubleSpinBox()
        for box, key in ((self._gl, "gain_l"), (self._gr, "gain_r")):
            box.setRange(50.0, 150.0)
            box.setDecimals(1)
            box.setSuffix(" %")
            box.setValue(params[key] * 100.0)
        form.addRow("Left wheel strength:", self._gl)
        form.addRow("Right wheel strength:", self._gr)

        self._mount = QComboBox()
        for deg in (0, 90, 180, -90):
            label = f"{deg:+d}°" + ("  (the recorded cart)" if deg == 90 else "")
            self._mount.addItem(label, float(deg))
        cur = (round(params["marker_offset"] / 90.0) * 90 + 180) % 360 - 180
        self._mount.setCurrentIndex(max(0, self._mount.findData(float(cur))))
        form.addRow("Marker mounted at (vs. front):", self._mount)

        lay.addLayout(form)
        note = QLabel("Changing the marker mounting re-runs the 🧭 heading calibration.")
        note.setStyleSheet("color: #94a3b8; font-size: 10px;")
        lay.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def params(self) -> dict:
        return {"px_per_unit": self._speed.value() / DRIVE_REF,
                "motor_tau": self._tau.value() / 1000.0,
                "deadband": self._dead.value(),
                "gain_l": self._gl.value() / 100.0,
                "gain_r": self._gr.value() / 100.0,
                "marker_offset": float(self._mount.currentData())}
