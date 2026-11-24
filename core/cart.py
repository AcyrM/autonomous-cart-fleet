import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Tuple, Optional

from config import CART_SIZE

# Start PWM assumed until heading calibration has measured the cart's own
# (the value the firmware used to have hard-coded)
DEFAULT_MIN_PWM = 170


class CartState(Enum):
    IDLE    = "idle"
    MOVING  = "moving"
    WAITING = "waiting"   # collision ahead, counting toward replan
    BLOCKED = "blocked"   # no path found


@dataclass
class Cart:
    id: str
    name: str
    color: str
    row: int               # integer grid cell (top-left corner)
    col: int
    home_row: int
    home_col: int
    size: Tuple[int, int] = field(default_factory=lambda: CART_SIZE)
    state: CartState = CartState.IDLE

    # Theta* path — list of integer (row, col) nodes
    path: List[Tuple[int, int]] = field(default_factory=list)
    path_index: int = 0

    # High-level waypoints from the active routine
    waypoints: List[Tuple[int, int]] = field(default_factory=list)
    waypoint_index: int = 0

    active_routine_id: Optional[str] = None
    blocked_since: Optional[int] = None   # tick when blocking started
    _move_tick: int = 0                   # counts up to CART_MOVE_EVERY
    marker_id: Optional[int] = None       # ArUco marker ID for production mode
    # Marker heading minus the direction the cart drives when told to go
    # forward (deg). 0 when the marker's top edge faces the front of the cart;
    # measured by ProdEngine.calibrate_heading().
    heading_offset: float = 0.0
    # Motor wiring as found by heading calibration, compensated on the PC:
    # invert_right — the firmware's RIGHT motor group turns opposite to its
    #                LEFT group for the same command (cart spins on DRIVE:x:x)
    # swap_sides   — the firmware's LEFT group is on the cart's right-hand side
    #                (a left turn command turns it right)
    invert_right: bool = False
    swap_sides: bool = False
    # Lowest PWM that starts the cart from rest, measured by heading
    # calibration (0 = not measured yet → DEFAULT_MIN_PWM). Turning in place
    # usually needs more than driving straight, so both are kept. Depends on
    # the motors, driver, battery and load — never hard-coded in the firmware.
    min_pwm_drive: int = 0
    min_pwm_turn: int = 0

    def lift(self, left: int, right: int) -> Tuple[int, int]:
        """Planner speeds (1..255 per side) → PWM that actually moves this cart
        (min..255), keeping 0 as stop. Opposite signs = turning in place."""
        drive = self.min_pwm_drive or DEFAULT_MIN_PWM
        lo = (self.min_pwm_turn or drive) if left * right < 0 else drive
        lo = max(0, min(255, lo))

        def f(v: int) -> int:
            if v == 0:
                return 0
            s = round(lo + (min(255, abs(v)) - 1) * (255 - lo) / 254)
            return s if v > 0 else -s
        return f(left), f(right)

    def wheel_command(self, left: int, right: int) -> Tuple[int, int]:
        """Planner's (left, right) wheel speeds → the firmware's DRIVE values."""
        a, b = (right, left) if self.swap_sides else (left, right)
        return a, (-b if self.invert_right else b)

    @staticmethod
    def make(name: str, color: str, row: int, col: int) -> "Cart":
        return Cart(
            id=str(uuid.uuid4())[:8],
            name=name, color=color,
            row=row, col=col,
            home_row=row, home_col=col,
        )

    @property
    def grid_pos(self) -> Tuple[int, int]:
        return (self.row, self.col)

    def all_cells(self) -> Set[Tuple[int, int]]:
        h, w = self.size
        return {(self.row + dr, self.col + dc)
                for dr in range(h) for dc in range(w)}

    def assign_routine(self, routine_id: str,
                       waypoints: List[Tuple[int, int]]) -> None:
        self.active_routine_id = routine_id
        self.waypoints = list(waypoints)
        self.waypoint_index = 0
        self.path = []
        self.path_index = 0
        self.blocked_since = None
        self._move_tick = 0
        self.state = CartState.MOVING

    def clear_task(self) -> None:
        self.path = []
        self.path_index = 0
        self.waypoints = []
        self.waypoint_index = 0
        self.active_routine_id = None
        self.blocked_since = None
        self._move_tick = 0
        self.state = CartState.IDLE

    def current_target_waypoint(self) -> Optional[Tuple[int, int]]:
        if self.waypoint_index < len(self.waypoints):
            return self.waypoints[self.waypoint_index]
        return None

    def to_dict(self) -> dict:
        d = {"id": self.id, "name": self.name, "color": self.color,
             "home_row": self.home_row, "home_col": self.home_col}
        if self.marker_id is not None:
            d["marker_id"] = self.marker_id
        if self.heading_offset:
            d["heading_offset"] = self.heading_offset
        if self.invert_right:
            d["invert_right"] = True
        if self.swap_sides:
            d["swap_sides"] = True
        if self.min_pwm_drive:
            d["min_pwm_drive"] = self.min_pwm_drive
        if self.min_pwm_turn:
            d["min_pwm_turn"] = self.min_pwm_turn
        return d

    @staticmethod
    def from_dict(d: dict) -> "Cart":
        cart = Cart(id=d["id"], name=d["name"], color=d["color"],
                    row=d["home_row"], col=d["home_col"],
                    home_row=d["home_row"], home_col=d["home_col"])
        cart.marker_id = d.get("marker_id")
        cart.heading_offset = float(d.get("heading_offset", 0.0))
        cart.invert_right = bool(d.get("invert_right", False))
        cart.swap_sides = bool(d.get("swap_sides", False))
        cart.min_pwm_drive = int(d.get("min_pwm_drive", 0))
        cart.min_pwm_turn = int(d.get("min_pwm_turn", 0))
        return cart
