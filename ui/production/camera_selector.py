"""Dialog to select the input camera device."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton,
)

from production.camera_source import list_available_cameras


class CameraSelectorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Camera Input")
        self.setMinimumWidth(400)
        self._selected_index: int = 0
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        lay.setContentsMargins(20, 18, 20, 18)

        title = QLabel("Choose the camera input device")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #60a5fa;")
        lay.addWidget(title)

        lay.addWidget(QLabel(
            "The selected camera will provide the top-view feed for production mode."
        ))

        self._list = QListWidget()
        cameras = list_available_cameras()
        if cameras:
            for idx, label in cameras:
                item = QListWidgetItem(f"  📷  {label}")
                item.setData(Qt.ItemDataRole.UserRole, idx)
                self._list.addItem(item)
            self._list.setCurrentRow(0)
        else:
            self._list.addItem("  ⚠  No cameras detected — check connections")

        lay.addWidget(self._list)

        hint = QLabel("Tip: the top-view installation camera will appear here once connected.")
        hint.setStyleSheet("color: #64748b; font-size: 10px;")
        lay.addWidget(hint)

        btns = QHBoxLayout()
        ok = QPushButton("Use selected camera")
        ok.setDefault(True)
        ok.clicked.connect(self._accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        lay.addLayout(btns)

    def _accept(self) -> None:
        item = self._list.currentItem()
        if item:
            idx = item.data(Qt.ItemDataRole.UserRole)
            if idx is not None:
                self._selected_index = idx
        self.accept()

    def selected_index(self) -> int:
        return self._selected_index
