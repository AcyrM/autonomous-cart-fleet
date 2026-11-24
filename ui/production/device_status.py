"""
Device status panel: one row per cart showing its Bluetooth link at a glance.

  ● Cart-1                       [Ping]
    Cart-1 [E8:31:CD:C5:B8:5E]
    Connected · responding · 38 ms · 12/12 pongs

The dot is green (connected and answering PING/PONG), amber (connected but
silent, or connecting / retrying), red (disconnected) or grey (no device
assigned). Refreshed by the main window's panel timer.
"""

from typing import Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from core.cart import Cart

_GREEN, _AMBER, _RED, _GREY = "#4ade80", "#fbbf24", "#f87171", "#64748b"


class _Row(QFrame):
    def __init__(self, controller, cart_id: str, parent=None):
        super().__init__(parent)
        self._ctl, self._cid = controller, cart_id
        self.setStyleSheet("QFrame { background:#0b1424; border-radius:5px; }"
                           "QLabel { background:transparent; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 5, 6, 5)
        lay.setSpacing(1)

        top = QHBoxLayout()
        top.setSpacing(6)
        self._dot  = QLabel("●")
        self._name = QLabel()
        self._name.setStyleSheet("font-weight:bold; font-size:11px;")
        self._ping = QPushButton("Ping")
        self._ping.setFixedSize(44, 20)
        self._ping.setStyleSheet("font-size:10px; padding:0;")
        self._ping.setToolTip("Send a PING now and measure the round trip")
        self._ping.clicked.connect(lambda: controller.ping(cart_id))
        top.addWidget(self._dot)
        top.addWidget(self._name, stretch=1)
        top.addWidget(self._ping)
        lay.addLayout(top)

        self._device = QLabel()
        self._device.setStyleSheet("color:#64748b; font-size:9px;")
        self._state = QLabel()
        self._state.setStyleSheet("font-size:10px;")
        self._state.setWordWrap(True)
        lay.addWidget(self._device)
        lay.addWidget(self._state)

    def refresh(self, cart: Cart) -> None:
        self._name.setText(cart.name)
        self._name.setStyleSheet(f"font-weight:bold; font-size:11px; color:{cart.color};")
        st = self._ctl.status(self._cid)
        self._device.setText(st.device or "no Bluetooth device assigned")
        self._ping.setEnabled(st.state == "connected")

        if st.state == "connected":
            if st.responding:
                colour = _GREEN
                parts = ["Connected", "responding"]
            else:
                colour = _AMBER
                parts = ["Connected", "NOT responding" if st.pings else "waiting for reply"]
            parts.append(f"{st.rtt_ms:.0f} ms" if st.rtt_ms is not None else "— ms")
            parts.append(f"{st.pongs}/{st.pings} pongs")
            if st.last_reply_s is not None and st.last_reply_s > 3:
                parts.append(f"last reply {st.last_reply_s:.0f} s ago")
            text = " · ".join(parts)
            if st.uptime_s is not None:
                up = st.uptime_s
                up_txt = f"{up // 3600}h{up % 3600 // 60:02d}m" if up >= 3600 else \
                         f"{up // 60}m{up % 60:02d}s" if up >= 60 else f"{up} s"
                text += f"\nCart up {up_txt} · last reset: {st.reset_reason or '?'}"
                if st.reset_reason in ("BROWNOUT", "PANIC", "WATCHDOG"):
                    colour = _AMBER
        elif st.state == "connecting":
            colour, text = _AMBER, "Connecting…"
        elif st.state == "retrying":
            colour, text = _AMBER, "Link down — retrying"
        elif st.state == "disconnected":
            colour, text = _RED, "Disconnected"
        else:
            colour, text = _GREY, "Assign a device in 🔵 Bluetooth"
        if st.error and st.state != "connected":
            text += f"\n{st.error[:90]}"
        self._dot.setStyleSheet(f"color:{colour}; font-size:13px;")
        self._state.setText(text)
        self._state.setStyleSheet(f"font-size:10px; color:{colour};")


class DeviceStatusPanel(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self._ctl = controller
        self._rows: Dict[str, _Row] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 6)
        lay.setSpacing(4)
        title = QLabel("DEVICES")
        title.setStyleSheet("font-weight:bold; color:#555; font-size:11px;")
        lay.addWidget(title)
        self._rows_lay = QVBoxLayout()
        self._rows_lay.setSpacing(4)
        lay.addLayout(self._rows_lay)
        self._empty = QLabel("No carts yet")
        self._empty.setStyleSheet("color:#64748b; font-size:10px;")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(self._empty)

    def refresh(self, carts: Dict[str, Cart]) -> None:
        if not hasattr(self._ctl, "status"):          # virtual links: nothing to show
            self.setVisible(False)
            return
        for cid in list(self._rows):
            if cid not in carts:
                self._rows.pop(cid).deleteLater()
        for cart in sorted(carts.values(), key=lambda c: c.name):
            row = self._rows.get(cart.id)
            if row is None:
                row = self._rows[cart.id] = _Row(self._ctl, cart.id)
                self._rows_lay.addWidget(row)
            row.refresh(cart)
        self._empty.setVisible(not carts)
