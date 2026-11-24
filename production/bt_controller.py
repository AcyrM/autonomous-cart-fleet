"""
Bluetooth Classic SPP controller for physical carts.

Each cart is assigned a BT MAC address. Commands are sent as ASCII lines
over an RFCOMM socket (channel 1, standard SPP). Each cart gets its own
writer and reader threads so sends never block the UI.

Backends (no third-party packages needed on Windows):
  connect   — stdlib socket (AF_BLUETOOTH / BTPROTO_RFCOMM), Windows and Linux
  discovery — Windows: Win32 Bluetooth API (bthprops.cpl) via ctypes
              Linux:   PyBluez2, if installed
  pairing   — Windows: BluetoothAuthenticateDeviceEx before the first connect
              (silent for "Just Works" SSP, otherwise Windows shows its PIN prompt)

Command protocol (ESP32 interprets these):
  DRIVE:<left>:<right>\\n   — set motor speeds, each -255..255
  RAW:<left>:<right>\\n     — same, exact PWM without the firmware's breakaway
                              boost (start-PWM measurement in calibration)
  STOP\\n                  — cut power to both motors immediately
  PING\\n                   — keep-alive, answered with PONG
The cart sends HB\\n every 2 s while connected: a link that stays silent
means old firmware or a stalled connection (reported via `silent`).

A link that drops (or fails to open) is retried every RECONNECT_S for as long
as the cart is wanted connected; disconnect() ends that.
"""

import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

RFCOMM_CHANNEL   = 1      # ESP32 BluetoothSerial registers SPP on channel 1
SCAN_DURATION_S  = 8.0
RECONNECT_S      = 3.0    # retry interval for a wanted link that is down
SILENT_S         = 6.0    # no HB/PONG this long → cart not responding
PING_EVERY_S     = 2.0    # link test: PING interval while connected
PING_TIMEOUT_S   = 3.0    # a PING unanswered this long counts as lost
RECENT_BOOT_S    = 60     # cart uptime below this on connect = it just restarted


@dataclass
class LinkStatus:
    """Snapshot of one cart's Bluetooth link, for the UI."""
    state: str                       # unassigned | disconnected | connecting |
                                     # retrying | connected
    device: Optional[str] = None     # "name [address]"
    responding: bool = False         # replied within SILENT_S
    rtt_ms: Optional[float] = None   # last PING→PONG round trip
    last_reply_s: Optional[float] = None   # seconds since the cart last sent anything
    pings: int = 0                   # PINGs sent on this link
    pongs: int = 0                   # PONGs received on this link
    error: str = ""                  # last connect/link error
    uptime_s: Optional[int] = None   # cart's uptime (from its heartbeat)
    reset_reason: str = ""           # why the cart's ESP32 last restarted
_IS_WINDOWS      = sys.platform == "win32"
_SOCKET_BT       = hasattr(socket, "AF_BLUETOOTH") and hasattr(socket, "BTPROTO_RFCOMM")


@dataclass
class BTDevice:
    address: str
    name: str
    paired: bool = False

    def __str__(self) -> str:
        suffix = "  (paired)" if self.paired else ""
        return f"{self.name}  [{self.address}]{suffix}"


# ── Windows backend (Win32 Bluetooth API via ctypes) ──────────────────────────

if _IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _ERROR_SUCCESS       = 0
    _ERROR_NO_MORE_ITEMS = 259
    _ERROR_CANCELLED     = 1223
    _MITM_NOT_REQUIRED   = 0

    class _SYSTEMTIME(ctypes.Structure):
        _fields_ = [(n, wintypes.WORD) for n in (
            "wYear", "wMonth", "wDayOfWeek", "wDay",
            "wHour", "wMinute", "wSecond", "wMilliseconds")]

    class _BT_DEVICE_INFO(ctypes.Structure):
        _fields_ = [
            ("dwSize",          wintypes.DWORD),
            ("Address",         ctypes.c_ulonglong),   # BLUETOOTH_ADDRESS union
            ("ulClassofDevice", wintypes.ULONG),
            ("fConnected",      wintypes.BOOL),
            ("fRemembered",     wintypes.BOOL),
            ("fAuthenticated",  wintypes.BOOL),
            ("stLastSeen",      _SYSTEMTIME),
            ("stLastUsed",      _SYSTEMTIME),
            ("szName",          wintypes.WCHAR * 248),
        ]

    class _BT_SEARCH_PARAMS(ctypes.Structure):
        _fields_ = [
            ("dwSize",               wintypes.DWORD),
            ("fReturnAuthenticated", wintypes.BOOL),
            ("fReturnRemembered",    wintypes.BOOL),
            ("fReturnUnknown",       wintypes.BOOL),
            ("fReturnConnected",     wintypes.BOOL),
            ("fIssueInquiry",        wintypes.BOOL),
            ("cTimeoutMultiplier",   ctypes.c_ubyte),  # units of 1.28 s, max 48
            ("hRadio",               wintypes.HANDLE),
        ]

    class _BT_RADIO_PARAMS(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD)]

    _bthprops = None

    def _api():
        global _bthprops
        if _bthprops is None:
            dll = ctypes.WinDLL("bthprops.cpl", use_last_error=True)
            dll.BluetoothFindFirstRadio.argtypes  = [ctypes.POINTER(_BT_RADIO_PARAMS),
                                                     ctypes.POINTER(wintypes.HANDLE)]
            dll.BluetoothFindFirstRadio.restype   = wintypes.HANDLE
            dll.BluetoothFindRadioClose.argtypes  = [wintypes.HANDLE]
            dll.BluetoothFindFirstDevice.argtypes = [ctypes.POINTER(_BT_SEARCH_PARAMS),
                                                     ctypes.POINTER(_BT_DEVICE_INFO)]
            dll.BluetoothFindFirstDevice.restype  = wintypes.HANDLE
            dll.BluetoothFindNextDevice.argtypes  = [wintypes.HANDLE,
                                                     ctypes.POINTER(_BT_DEVICE_INFO)]
            dll.BluetoothFindNextDevice.restype   = wintypes.BOOL
            dll.BluetoothFindDeviceClose.argtypes = [wintypes.HANDLE]
            dll.BluetoothAuthenticateDeviceEx.argtypes = [
                wintypes.HWND, wintypes.HANDLE, ctypes.POINTER(_BT_DEVICE_INFO),
                ctypes.c_void_p, ctypes.c_int]
            dll.BluetoothAuthenticateDeviceEx.restype  = wintypes.DWORD
            _bthprops = dll
        return _bthprops

    def _win_has_radio() -> bool:
        params = _BT_RADIO_PARAMS(ctypes.sizeof(_BT_RADIO_PARAMS))
        h_radio = wintypes.HANDLE()
        h_find = _api().BluetoothFindFirstRadio(ctypes.byref(params), ctypes.byref(h_radio))
        if not h_find:
            return False
        _api().BluetoothFindRadioClose(h_find)
        ctypes.windll.kernel32.CloseHandle(h_radio)
        return True

    def _win_enum(inquiry: bool, duration_s: float = SCAN_DURATION_S) -> List[BTDevice]:
        """Paired/remembered devices, plus nearby ones when inquiry=True."""
        params = _BT_SEARCH_PARAMS()
        params.dwSize               = ctypes.sizeof(_BT_SEARCH_PARAMS)
        params.fReturnAuthenticated = True
        params.fReturnRemembered    = True
        params.fReturnUnknown       = inquiry
        params.fReturnConnected     = True
        params.fIssueInquiry        = inquiry
        params.cTimeoutMultiplier   = max(1, min(48, round(duration_s / 1.28)))
        params.hRadio               = None

        info = _BT_DEVICE_INFO()
        info.dwSize = ctypes.sizeof(_BT_DEVICE_INFO)
        h_find = _api().BluetoothFindFirstDevice(ctypes.byref(params), ctypes.byref(info))
        if not h_find:
            err = ctypes.get_last_error()
            if err == _ERROR_NO_MORE_ITEMS:
                return []
            raise OSError(err, ctypes.FormatError(err))

        devices: Dict[str, BTDevice] = {}
        try:
            while True:
                addr = _fmt_addr(info.Address)
                devices[addr] = BTDevice(addr, info.szName or addr,
                                         bool(info.fAuthenticated or info.fRemembered))
                info = _BT_DEVICE_INFO()
                info.dwSize = ctypes.sizeof(_BT_DEVICE_INFO)
                if not _api().BluetoothFindNextDevice(h_find, ctypes.byref(info)):
                    break
        finally:
            _api().BluetoothFindDeviceClose(h_find)
        return list(devices.values())

    def _win_pair(address: str) -> None:
        """Pair with a device. Returns quietly if it is already paired."""
        info = _BT_DEVICE_INFO()
        info.dwSize  = ctypes.sizeof(_BT_DEVICE_INFO)
        info.Address = _parse_addr(address)
        err = _api().BluetoothAuthenticateDeviceEx(None, None, ctypes.byref(info),
                                                   None, _MITM_NOT_REQUIRED)
        if err in (_ERROR_SUCCESS, _ERROR_NO_MORE_ITEMS):   # NO_MORE_ITEMS = already paired
            return
        if err == _ERROR_CANCELLED:
            raise OSError(err, "pairing cancelled")
        raise OSError(err, ctypes.FormatError(err))


def _fmt_addr(value: int) -> str:
    return ":".join(f"{(value >> s) & 0xFF:02X}" for s in range(40, -1, -8))


def _parse_addr(address: str) -> int:
    return int(address.replace(":", "").replace("-", ""), 16)


def _discover(duration_s: float) -> List[BTDevice]:
    if _IS_WINDOWS:
        if not _win_has_radio():
            raise OSError("No Bluetooth adapter found — plug in or enable Bluetooth")
        return _win_enum(inquiry=True, duration_s=duration_s)
    try:
        import bluetooth  # PyBluez2 (Linux/macOS discovery only)
    except (ImportError, AttributeError):
        raise OSError("Device discovery needs PyBluez2 on this OS — run: pip install PyBluez2")
    found = bluetooth.discover_devices(lookup_names=True, duration=int(duration_s),
                                       lookup_class=False)
    return [BTDevice(addr, name or addr) for addr, name in found]


# ── Per-cart link ─────────────────────────────────────────────────────────────

class _Link:
    """One open RFCOMM socket with its writer and reader threads.

    Outgoing commands are coalesced: the writer always sends the most recent
    command, since each DRIVE/STOP fully replaces the previous motor state.
    PINGs travel outside that slot, so a link test never replaces a STOP."""

    def __init__(self, cart_id: str, sock: socket.socket):
        self.cart_id   = cart_id
        self.sock      = sock
        self.cond      = threading.Condition()
        self.send_lock = threading.Lock()
        self.pending:  Optional[bytes] = None
        self.closing   = False
        self.opened    = time.monotonic()
        self.last_rx   = 0.0          # monotonic time of the last line from the cart
        self.replied   = False        # any line received on this link
        self.silent    = False        # `silent` emitted, waiting for traffic again
        self.rx_buf    = b""
        self.ping_now  = True         # send a PING at once (firmware check)
        self.ping_sent: Optional[float] = None   # outstanding PING time
        self.last_ping = 0.0
        self.rtt_ms:   Optional[float] = None
        self.pings     = 0
        self.pongs     = 0
        self.uptime_s: Optional[int] = None    # from HB:<uptime>:<reset reason>
        self.uptime_at = 0.0
        self.reset_reason = ""
        self.boot_reported = False


class BTController(QObject):
    connected    = pyqtSignal(str)       # cart_id
    disconnected = pyqtSignal(str)       # cart_id
    error        = pyqtSignal(str, str)  # cart_id, message
    scan_result  = pyqtSignal(list)      # list of BTDevice
    responding   = pyqtSignal(str)       # cart_id: first reply on a link (firmware OK)
    silent       = pyqtSignal(str, bool) # cart_id, had answered before: no reply for SILENT_S
    restarted    = pyqtSignal(str, str, int)  # cart_id, reset reason, uptime s:
                                              # the cart rebooted shortly before connecting
    stalled      = pyqtSignal(str, int, int)  # cart_id, new stalls, longest ms: the cart's
                                              # main loop froze (its Bluetooth was blocked)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._devices:    Dict[str, BTDevice] = {}   # cart_id -> device
        self._links:      Dict[str, _Link]    = {}
        self._connecting: Set[str]            = set()
        self._wanted:     Set[str]            = set()   # keep these connected
        self._failing:    Set[str]            = set()   # connect failure already logged
        self._last_error: Dict[str, str]      = {}
        self._stalls_seen: Dict[str, int]     = {}   # cart's stall counter, last seen
        self._state_lock = threading.Lock()
        self.error.connect(self._remember_error)

    def _remember_error(self, cart_id: str, msg: str) -> None:
        if cart_id:
            self._last_error[cart_id] = msg

    # ── Device assignment ─────────────────────────────────────────────────────

    def assign(self, cart_id: str, device: Optional[BTDevice]) -> None:
        if device is None:
            self._devices.pop(cart_id, None)
        else:
            self._devices[cart_id] = device

    def unassign(self, cart_id: str) -> None:
        self.disconnect(cart_id)
        self._devices.pop(cart_id, None)

    def device_for(self, cart_id: str) -> Optional[BTDevice]:
        return self._devices.get(cart_id)

    def is_connected(self, cart_id: str) -> bool:
        return cart_id in self._links

    def is_responding(self, cart_id: str) -> bool:
        """Connected and the cart has replied within SILENT_S."""
        link = self._links.get(cart_id)
        return link is not None and link.replied and not link.silent

    def status(self, cart_id: str) -> LinkStatus:
        dev = self._devices.get(cart_id)
        if dev is None:
            return LinkStatus("unassigned")
        name = f"{dev.name} [{dev.address}]"
        link = self._links.get(cart_id)
        if link is None:
            if cart_id in self._connecting:
                state = "connecting"
            elif cart_id in self._wanted:
                state = "retrying"
            else:
                state = "disconnected"
            return LinkStatus(state, name, error=self._last_error.get(cart_id, ""))
        now = time.monotonic()
        uptime = None if link.uptime_s is None else \
            int(link.uptime_s + (now - link.uptime_at))
        return LinkStatus(
            "connected", name,
            responding=link.replied and not link.silent,
            rtt_ms=link.rtt_ms,
            last_reply_s=(now - link.last_rx) if link.last_rx else None,
            pings=link.pings, pongs=link.pongs,
            uptime_s=uptime, reset_reason=link.reset_reason)

    def ping(self, cart_id: str) -> None:
        """Send a link test now (the result shows up in status().rtt_ms)."""
        link = self._links.get(cart_id)
        if link is None:
            return
        with link.cond:
            link.ping_now = True
            link.cond.notify()

    # ── Scanning ──────────────────────────────────────────────────────────────

    def scan_async(self) -> None:
        """Discover nearby Bluetooth devices in a background thread.
        Emits scan_result(list[BTDevice]) when done."""
        threading.Thread(target=self._scan_thread, daemon=True).start()

    def _scan_thread(self) -> None:
        try:
            devices = _discover(SCAN_DURATION_S)
            self.scan_result.emit(devices)
        except Exception as e:
            self.scan_result.emit([])
            self.error.emit("", f"Scan failed: {e}")

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect_cart(self, cart_id: str) -> None:
        """Open RFCOMM connection to a cart's assigned device (background thread)."""
        if cart_id not in self._devices:
            return
        with self._state_lock:
            self._wanted.add(cart_id)
            if cart_id in self._links or cart_id in self._connecting:
                return
            self._connecting.add(cart_id)
        threading.Thread(target=self._connect_thread,
                         args=(cart_id,), daemon=True).start()

    def _retry_later(self, cart_id: str) -> None:
        """Reconnect a wanted cart after RECONNECT_S (power cycle, out of range…)."""
        def retry():
            if cart_id in self._wanted and cart_id not in self._links:
                self.connect_cart(cart_id)
        t = threading.Timer(RECONNECT_S, retry)
        t.daemon = True
        t.start()

    def _connect_thread(self, cart_id: str) -> None:
        try:
            device = self._devices.get(cart_id)
            if device is None:
                return
            sock, err = self._open_socket(device)
            if sock is None:
                if cart_id not in self._failing:          # log once per outage
                    self._failing.add(cart_id)
                    self.error.emit(cart_id, f"Connect failed: {err} — is the cart on, "
                                             f"in range and not connected to another "
                                             f"program? Retrying every {RECONNECT_S:g} s")
                if cart_id in self._wanted:
                    self._retry_later(cart_id)
                return
            self._failing.discard(cart_id)
            self._last_error.pop(cart_id, None)
            link = _Link(cart_id, sock)
            with self._state_lock:
                self._links[cart_id] = link
            threading.Thread(target=self._writer_loop, args=(link,), daemon=True).start()
            threading.Thread(target=self._reader_loop, args=(link,), daemon=True).start()
            self.connected.emit(cart_id)                  # writer PINGs at once
        finally:
            with self._state_lock:
                self._connecting.discard(cart_id)

    def _open_socket(self, device: BTDevice) -> Tuple[Optional[socket.socket], str]:
        if not _SOCKET_BT:
            return None, "this Python build has no Bluetooth socket support"
        if _IS_WINDOWS:
            try:
                if not _win_has_radio():
                    return None, "no Bluetooth adapter found — plug in or enable Bluetooth"
                paired = {d.address for d in _win_enum(inquiry=False) if d.paired}
                if device.address.upper() not in paired:
                    _win_pair(device.address)
            except OSError as e:
                return None, f"pairing with {device.address} failed: {e}"
        sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        try:
            sock.connect((device.address, RFCOMM_CHANNEL))
        except OSError as e:
            sock.close()
            return None, str(e)
        return sock, ""

    def disconnect(self, cart_id: str) -> None:
        with self._state_lock:
            self._wanted.discard(cart_id)                 # user's choice: no reconnect
            self._failing.discard(cart_id)
            link = self._links.pop(cart_id, None)
        if link is None:
            return
        # Flush the last queued command (typically STOP) before closing
        with link.cond:
            link.closing = True
            data, link.pending = link.pending, None
            link.cond.notify()
        if data:
            with link.send_lock:
                try:
                    link.sock.sendall(data)
                except OSError:
                    pass
        self._close_socket(link)
        self.disconnected.emit(cart_id)

    def disconnect_all(self) -> None:
        for cid in list(self._links.keys()):
            self.disconnect(cid)

    def _drop(self, link: _Link, reason: str) -> None:
        """Tear down a link that failed underneath us (no-op if already closed)."""
        with self._state_lock:
            if self._links.get(link.cart_id) is not link:
                return
            del self._links[link.cart_id]
        with link.cond:
            link.closing = True
            link.cond.notify()
        self._close_socket(link)
        self.disconnected.emit(link.cart_id)
        retry = link.cart_id in self._wanted
        self.error.emit(link.cart_id, reason + (f" — reconnecting every {RECONNECT_S:g} s"
                                                if retry else ""))
        if retry:
            self._retry_later(link.cart_id)

    @staticmethod
    def _close_socket(link: _Link) -> None:
        try:
            link.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            link.sock.close()
        except OSError:
            pass

    # ── Link threads ──────────────────────────────────────────────────────────

    def _writer_loop(self, link: _Link) -> None:
        while True:
            with link.cond:
                while link.pending is None and not link.closing \
                        and not self._ping_due(link):
                    link.cond.wait(0.5)
                    self._check_silence(link)
                if link.closing:
                    return
                data, link.pending = link.pending, None
                if self._ping_due(link):
                    now = time.monotonic()
                    if link.ping_sent is not None:         # previous one went unanswered
                        link.rtt_ms = None
                    link.ping_now, link.ping_sent, link.last_ping = False, now, now
                    link.pings += 1
                    data = (data or b"") + b"PING\n"       # after the command: STOP first
            with link.send_lock:
                try:
                    link.sock.sendall(data)
                except OSError as e:
                    self._drop(link, f"Send failed: {e}")
                    return

    @staticmethod
    def _ping_due(link: _Link) -> bool:
        now = time.monotonic()
        if link.ping_now:
            return True
        if link.ping_sent is not None and now - link.ping_sent < PING_TIMEOUT_S:
            return False                                   # still waiting for PONG
        return now - link.last_ping >= PING_EVERY_S

    def _check_silence(self, link: _Link) -> None:
        """Called from the writer thread about twice a second. A cart that had
        been answering and goes quiet is gone (typically its ESP32 reset when
        the motors started): Windows can take 10–20 s to notice, so drop the
        link now and let the reconnect loop bring it back."""
        now  = time.monotonic()
        last = link.last_rx or link.opened
        if link.silent or link.closing or now - last <= SILENT_S:
            return
        link.silent = True
        self.silent.emit(link.cart_id, link.replied)
        if link.replied:
            self._drop(link, f"Cart stopped answering for {SILENT_S:g} s (it may have "
                             f"restarted — see its reset reason after reconnecting)")

    def _reader_loop(self, link: _Link) -> None:
        """Read the cart's lines (HB / PONG) and detect the link going down."""
        while True:
            try:
                chunk = link.sock.recv(1024)
            except OSError as e:
                if not link.closing:
                    self._drop(link, f"Connection lost: {e}")
                return
            if not chunk:
                if not link.closing:
                    self._drop(link, "Connection closed by cart")
                return
            link.rx_buf = (link.rx_buf + chunk)[-512:]
            if b"\n" not in link.rx_buf:
                continue
            *lines, link.rx_buf = link.rx_buf.split(b"\n")
            now = time.monotonic()
            for line in lines:
                line = line.strip()
                if line == b"PONG" and link.ping_sent is not None:
                    link.rtt_ms    = (now - link.ping_sent) * 1000.0
                    link.ping_sent = None
                    link.pongs    += 1
                elif line.startswith(b"HB:"):
                    self._parse_heartbeat(link, line, now)
            link.last_rx = now
            link.silent  = False
            if not link.replied:
                link.replied = True
                self.responding.emit(link.cart_id)

    def _parse_heartbeat(self, link: _Link, line: bytes, now: float) -> None:
        """HB:<uptime s>:<reset reason>. The first one on a link tells whether
        the cart rebooted just before (re)connecting, and why."""
        parts = line.decode(errors="replace").split(":")
        try:
            link.uptime_s, link.uptime_at = int(parts[1]), now
        except (IndexError, ValueError):
            return
        link.reset_reason = parts[2] if len(parts) > 2 else ""
        # HB:...:<stalls>:<longest ms> (newer firmware): times the cart's main
        # loop froze because a Bluetooth call blocked — survives reconnects,
        # so the stall that caused a drop shows up on the first heartbeat after
        try:
            stalls, longest = int(parts[3]), int(parts[4])
        except (IndexError, ValueError):
            stalls = None
        if stalls is not None:
            seen = self._stalls_seen.get(link.cart_id, 0)
            if stalls < seen:
                seen = 0                                   # counter restarted with the cart
            if stalls > seen:
                self.stalled.emit(link.cart_id, stalls - seen, longest)
            self._stalls_seen[link.cart_id] = stalls
        if not link.boot_reported:
            link.boot_reported = True
            if link.uptime_s < RECENT_BOOT_S:
                self.restarted.emit(link.cart_id, link.reset_reason, link.uptime_s)

    # ── Commands ──────────────────────────────────────────────────────────────

    def send(self, cart_id: str, command: str) -> None:
        """Queue a raw command string (newline appended automatically).
        Never blocks; a newer command replaces one not yet sent."""
        link = self._links.get(cart_id)
        if link is None:
            return
        with link.cond:
            if link.closing:
                return
            link.pending = (command + "\n").encode()
            link.cond.notify()

    def drive(self, cart_id: str, left: int, right: int) -> None:
        left  = max(-255, min(255, int(left)))
        right = max(-255, min(255, int(right)))
        self.send(cart_id, f"DRIVE:{left}:{right}")

    def drive_raw(self, cart_id: str, left: int, right: int) -> None:
        """Exact PWM, no breakaway boost in the firmware."""
        left  = max(-255, min(255, int(left)))
        right = max(-255, min(255, int(right)))
        self.send(cart_id, f"RAW:{left}:{right}")

    def stop(self, cart_id: str) -> None:
        self.send(cart_id, "STOP")

    def stop_all(self) -> None:
        for cid in list(self._links.keys()):
            self.stop(cid)

    # ── Legacy shim (kept for UI compatibility) ───────────────────────────────

    def assignments(self) -> Dict[str, Optional[BTDevice]]:
        return {cid: self._devices.get(cid) for cid in self._devices}
