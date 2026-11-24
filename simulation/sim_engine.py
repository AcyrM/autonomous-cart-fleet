"""
Simulation engine — integer-cell occupancy model.

Every entity (cart and obstacle) occupies a fixed set of integer grid cells.
No sub-cell fractions, no AABB.  Before any entity moves, the destination cells
are checked against a global occupancy set; if any cell is taken the move is
rejected.  This makes it physically impossible for two entities to share a cell.

Cart movement:
  - Theta* plans a path as a list of integer (row, col) waypoints.
  - Each CART_MOVE_EVERY ticks the cart takes one Chebyshev step toward the
    current path node (8-directional, 1 cell at a time).
  - On collision: enters WAITING and counts ticks; after BLOCK_REPLAN_TICKS
    a fresh Theta* is computed around the current blockers.

Obstacle movement:
  - Handled by SimObstacle.tick() which updates the shared occupancy set.
"""

import math
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from config import (
    SIM_TICK_MS, CART_MOVE_EVERY, NUM_OBSTACLES,
    BLOCK_REPLAN_TICKS, GRID_ROWS, GRID_COLS, CART_SIZE,
)
from core.grid import Grid
from core.cart import Cart, CartState
from core.routine import Routine
from core.routine_manager import RoutineManager
from pathfinding.theta_star import theta_star
from simulation.sim_obstacles import SimObstacle


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sign(x: int) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _make_passable(grid: Grid, cart_size: Tuple[int, int],
                   extra_blocked: Set[Tuple[int, int]]) -> Callable[[int, int], bool]:
    """
    Returns passable(r, c) → bool that checks the full cart footprint.
    The cart's top-left corner is the planning position.
    """
    ch, cw = cart_size
    rows, cols = grid.rows, grid.cols

    def passable(r: int, c: int) -> bool:
        if r + ch > rows or c + cw > cols:
            return False
        for dr in range(ch):
            for dc in range(cw):
                if grid.is_wall(r + dr, c + dc):
                    return False
                if (r + dr, c + dc) in extra_blocked:
                    return False
        return True

    return passable


def _build_occupancy(carts: Dict[str, Cart],
                     obstacles: List[SimObstacle]) -> Set[Tuple[int, int]]:
    cells: Set[Tuple[int, int]] = set()
    for cart in carts.values():
        cells |= cart.all_cells()
    for obs in obstacles:
        cells |= obs.all_cells()
    return cells


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

class SimEngine(QObject):
    state_updated = pyqtSignal()
    log_message   = pyqtSignal(str)

    def __init__(self, grid: Grid, routine_manager: RoutineManager):
        super().__init__()
        self.grid = grid
        self.routine_manager = routine_manager
        self.carts: Dict[str, Cart] = {}
        self.obstacles: List[SimObstacle] = []

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._speed  = 1
        self._tick_count = 0
        self._running = False

    # ── Public API ──────────────────────────────────────────────────────────

    def add_cart(self, cart: Cart) -> None:
        self.carts[cart.id] = cart

    def remove_cart(self, cart_id: str) -> None:
        self.carts.pop(cart_id, None)

    def spawn_obstacles(self, n: int = NUM_OBSTACLES) -> None:
        self.obstacles.clear()
        for _ in range(n):
            self.obstacles.append(SimObstacle.spawn_random(self.grid))

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._timer.start(max(10, SIM_TICK_MS // self._speed))

    def stop(self) -> None:
        self._running = False
        self._timer.stop()

    def set_speed(self, factor: int) -> None:
        self._speed = max(1, factor)
        if self._running:
            self._timer.start(max(10, SIM_TICK_MS // self._speed))

    def trigger_routine(self, routine: Routine) -> None:
        cart = self.carts.get(routine.cart_id)
        if cart and routine.waypoints:
            cart.assign_routine(routine.id, routine.waypoints)
            self._plan_path(cart)
            self.log_message.emit(
                f"▶ '{routine.name}' → {cart.name}"
            )

    @property
    def running(self) -> bool:
        return self._running

    # ── Tick ────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._tick_count += 1

        # Build fresh occupancy set (all cells taken right now)
        occupied = _build_occupancy(self.carts, self.obstacles)

        # 1. Move obstacles (each updates `occupied` in-place)
        for obs in self.obstacles:
            obs.tick(self.grid, occupied)

        # 2. Schedule check (~once per real second)
        if self._tick_count % 10 == 0:
            self._check_schedules()

        # 3. Advance each active cart — rebuild occupancy fresh so each cart
        #    sees the updated positions of all other carts that already moved.
        for cart in self.carts.values():
            if cart.state not in (CartState.IDLE, CartState.BLOCKED):
                occupied = _build_occupancy(self.carts, self.obstacles)
                self._step_cart(cart, occupied)

        self.state_updated.emit()

    def _check_schedules(self) -> None:
        now = datetime.now()
        busy = {c.id for c in self.carts.values()
                if c.state not in (CartState.IDLE, CartState.BLOCKED)}
        for routine in self.routine_manager.check_due(now, busy):
            self.routine_manager.mark_triggered(routine, now)
            self.trigger_routine(routine)

    # ── Cart stepping ────────────────────────────────────────────────────────

    def _step_cart(self, cart: Cart,
                   occupied: Set[Tuple[int, int]]) -> None:
        # Rate-limit movement
        cart._move_tick += 1
        if cart._move_tick < CART_MOVE_EVERY:
            return
        cart._move_tick = 0

        # Need a (new) path?
        if not cart.path or cart.path_index >= len(cart.path):
            self._plan_path(cart, occupied)
            return

        # Current target node on the Theta* path
        tr, tc = cart.path[cart.path_index]

        # Already at this node → advance
        if (cart.row, cart.col) == (tr, tc):
            cart.path_index += 1
            if cart.path_index >= len(cart.path):
                self._on_segment_done(cart, occupied)
            return

        # One Chebyshev step toward (tr, tc)
        nr = cart.row + _sign(tr - cart.row)
        nc = cart.col + _sign(tc - cart.col)

        if self._can_place(cart, nr, nc, occupied):
            # Commit move
            occupied -= cart.all_cells()
            cart.row, cart.col = nr, nc
            occupied |= cart.all_cells()
            cart.blocked_since = None
            cart.state = CartState.MOVING

            # Reached path node?
            if (cart.row, cart.col) == (tr, tc):
                cart.path_index += 1
                if cart.path_index >= len(cart.path):
                    self._on_segment_done(cart, occupied)
        else:
            # Blocked — count toward replan
            cart.state = CartState.WAITING
            if cart.blocked_since is None:
                cart.blocked_since = self._tick_count
            elif self._tick_count - cart.blocked_since >= BLOCK_REPLAN_TICKS:
                self.log_message.emit(
                    f"↺ {cart.name} replanning (stuck {BLOCK_REPLAN_TICKS} ticks)"
                )
                self._plan_path(cart, occupied)
                cart.blocked_since = None

    def _can_place(self, cart: Cart, nr: int, nc: int,
                   occupied: Set[Tuple[int, int]]) -> bool:
        """
        True if the cart can move to (nr, nc).
        Checks every cell the cart WOULD occupy that it doesn't already occupy
        (cells it vacates are not considered blocking).
        """
        ch, cw = cart.size
        current = cart.all_cells()
        for dr in range(ch):
            for dc in range(cw):
                cell = (nr + dr, nc + dc)
                if cell in current:
                    continue          # the cart is leaving this cell
                if not self.grid.is_passable(nr + dr, nc + dc):
                    return False
                if cell in occupied:
                    return False      # another entity is there
        return True

    def _on_segment_done(self, cart: Cart,
                         occupied: Set[Tuple[int, int]]) -> None:
        """Called when the cart reaches the end of its current Theta* path."""
        target = cart.current_target_waypoint()
        if target and (cart.row, cart.col) == target:
            cart.waypoint_index += 1

        if cart.waypoint_index < len(cart.waypoints):
            self._plan_path(cart, occupied)
        else:
            routine = self.routine_manager.get(cart.active_routine_id or "")
            if routine and routine.repeat:
                cart.waypoint_index = 0
                self._plan_path(cart, occupied)
                self.log_message.emit(f"↻ {cart.name} repeating '{routine.name}'")
            else:
                self.log_message.emit(f"✓ {cart.name} done")
                cart.clear_task()

    # ── Path planning ────────────────────────────────────────────────────────

    def _plan_path(self, cart: Cart,
                   occupied: Optional[Set[Tuple[int, int]]] = None) -> None:
        if occupied is None:
            occupied = _build_occupancy(self.carts, self.obstacles)

        target = cart.current_target_waypoint()
        if target is None:
            return

        start = cart.grid_pos
        if start == target:
            cart.waypoint_index += 1
            self._plan_path(cart, occupied)
            return

        ch, cw = cart.size

        # Cells taken by everything *except* this cart
        others = occupied - cart.all_cells()
        # Don't treat the goal footprint as blocked (allow arrival)
        goal_cells = {(target[0] + dr, target[1] + dc)
                      for dr in range(ch) for dc in range(cw)}
        others -= goal_cells

        # Attempt 1: avoid all other entities
        passable = _make_passable(self.grid, cart.size, others)
        path = theta_star(passable, GRID_ROWS, GRID_COLS, start, target)

        if not path:
            # Attempt 2: ignore dynamic obstacles (static walls only)
            obs_cells: Set[Tuple[int, int]] = set()
            for obs in self.obstacles:
                obs_cells |= obs.all_cells()
            passable2 = _make_passable(self.grid, cart.size, obs_cells - goal_cells)
            path = theta_star(passable2, GRID_ROWS, GRID_COLS, start, target)

        if path:
            cart.path = path
            cart.path_index = 0
            cart._move_tick = 0
            cart.state = CartState.MOVING
        else:
            cart.state = CartState.BLOCKED
            self.log_message.emit(f"⚠ {cart.name}: no path to {target}")
