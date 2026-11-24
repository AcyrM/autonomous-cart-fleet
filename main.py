"""Entry point — mode selection → MainWindow."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QRadioButton, QPushButton,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# ── Dark blue stylesheet ────────────────────────────────────────────────
DARK_STYLE = """
* { font-family: "Segoe UI", "Inter", sans-serif; }

QMainWindow, QDialog, QWidget {
    background-color: #0d1220;
    color: #cbd5e1;
}

QListWidget, QTextEdit, QLineEdit, QSpinBox, QComboBox, QTimeEdit {
    background-color: #060912;
    color: #e2e8f0;
    border: 1px solid #1e3a5f;
    border-radius: 5px;
    padding: 2px 4px;
    selection-background-color: #1d4ed8;
}
QListWidget::item:selected {
    background-color: #1e3a5f;
    color: #e2e8f0;
}
QListWidget::item:hover { background-color: #132035; }

QPushButton {
    background-color: #1e3a5f;
    color: #93c5fd;
    border: 1px solid #2563eb;
    border-radius: 5px;
    padding: 4px 10px;
}
QPushButton:hover  { background-color: #2563eb; color: #ffffff; }
QPushButton:pressed { background-color: #1d4ed8; }
QPushButton:checked { background-color: #7c3aed; border-color: #a78bfa; color: #fff; }
QPushButton:disabled { background-color: #1a2535; color: #475569; border-color: #1e3a5f; }

QToolBar {
    background-color: #07101f;
    border-bottom: 1px solid #1e3a5f;
    spacing: 2px;
}
QToolBar QLabel { color: #94a3b8; font-size: 11px; }
QToolBar::separator { background: #1e3a5f; width: 1px; margin: 4px 6px; }

QStatusBar { background-color: #07101f; color: #475569; }

QGroupBox {
    border: 1px solid #1e3a5f;
    border-radius: 5px;
    margin-top: 10px;
    color: #60a5fa;
    font-size: 11px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}

QScrollArea { border: none; background: #060912; }
QScrollBar:vertical, QScrollBar:horizontal {
    background: #0d1220;
    width: 8px; height: 8px;
}
QScrollBar::handle { background: #1e3a5f; border-radius: 4px; }

QSplitter::handle { background: #1e3a5f; }

QLabel  { color: #cbd5e1; }
QCheckBox, QRadioButton { color: #cbd5e1; }
QCheckBox::indicator, QRadioButton::indicator {
    border: 1px solid #2563eb;
    border-radius: 3px;
    background: #060912;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background: #2563eb;
}

QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #0d1220;
    color: #e2e8f0;
    border: 1px solid #1e3a5f;
    selection-background-color: #1e3a5f;
}
"""


def _mode_dialog(app: QApplication) -> str | None:
    dlg = QDialog()
    dlg.setWindowTitle("Cart Automation System")
    dlg.setFixedSize(420, 270)

    layout = QVBoxLayout(dlg)
    layout.setSpacing(14)
    layout.setContentsMargins(28, 22, 28, 22)

    title = QLabel("Select Operation Mode")
    title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
    title.setStyleSheet("color: #60a5fa;")
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(title)

    sim_btn = QRadioButton(
        "🖥   Simulation  —  no hardware required\n"
        "       virtual carts, random moving obstacles"
    )
    prod_btn = QRadioButton(
        "📷   Production  —  real camera + Bluetooth\n"
        "       YOLO obstacle detection, physical carts"
    )
    virt_btn = QRadioButton(
        "🧪   Virtual production  —  digital twin\n"
        "       full production stack on the recorded floor footage"
    )
    sim_btn.setChecked(True)
    layout.addWidget(sim_btn)
    layout.addWidget(prod_btn)
    layout.addWidget(virt_btn)

    btn_row = QHBoxLayout()
    ok_btn   = QPushButton("Launch")
    ok_btn.setFixedWidth(90)
    ok_btn.setDefault(True)
    quit_btn = QPushButton("Exit")
    quit_btn.setFixedWidth(70)
    btn_row.addStretch()
    btn_row.addWidget(ok_btn)
    btn_row.addWidget(quit_btn)
    layout.addLayout(btn_row)

    ok_btn.clicked.connect(dlg.accept)
    quit_btn.clicked.connect(dlg.reject)

    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    if virt_btn.isChecked():
        return "virtual"
    return "simulation" if sim_btn.isChecked() else "production"


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Cart Automation System")
    ap.add_argument("--virtual", action="store_true",
                    help="start the digital twin directly (skip the mode dialog)")
    ap.add_argument("--carts", type=int, default=2, help="virtual mode: number of carts")
    ap.add_argument("--obstacles", type=int, default=5, help="virtual mode: obstacles")
    ap.add_argument("--marker-offset", type=float, default=90.0,
                    help="virtual mode: marker heading minus travel direction, "
                         "multiple of 90 (the recorded cart: 90)")
    ap.add_argument("--walkers", type=float, default=2.0,
                    help="virtual mode: people walking through per minute")
    ap.add_argument("--orientation", choices=["landscape", "portrait"], default="landscape",
                    help="virtual mode: camera view, 16:9 landscape or as recorded")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="virtual mode: render resolution relative to the footage")
    return ap.parse_known_args()[0]


def main() -> None:
    args = _parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("Cart Automation System")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)

    mode = "virtual" if args.virtual else _mode_dialog(app)
    if mode is None:
        sys.exit(0)

    camera_index = 0
    if mode == "production":
        try:
            import cv2  # noqa: F401
        except ImportError:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                None, "Missing dependency",
                "Production mode requires opencv-python.\n"
                "Install it with:  pip install opencv-python\n"
                "Then re-launch, or choose Simulation.",
            )
            sys.exit(1)

        from ui.production.camera_selector import CameraSelectorDialog
        cam_dlg = CameraSelectorDialog()
        if cam_dlg.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
        camera_index = cam_dlg.selected_index()

    twin_config = None
    if mode == "virtual":
        from simulation.digital_twin import TwinConfig
        twin_config = TwinConfig(n_carts=args.carts, n_obstacles=args.obstacles,
                                 marker_offset=args.marker_offset, scale=args.scale,
                                 walkers_per_min=args.walkers,
                                 orientation=args.orientation)

    from ui.main_window import MainWindow
    win = MainWindow(mode, camera_index=camera_index, twin_config=twin_config)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
