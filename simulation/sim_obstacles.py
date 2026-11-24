"""
Randomly wandering multi-cell obstacles.

Movement model:
  - Integer (row, col) positions — no sub-cell fractions.
  - Each OBS_MOVE_EVERY ticks, try to step 1 cell in the preferred direction.
  - On collision: rotate 90° CCW or CW (random), try again; if still stuck, reverse.
  - Every WANDER_CHANGE ticks: spontaneously pick a new random direction so
    obstacles explore the whole corridor network instead of just bouncing.
"""

import random
from dataclasses import dataclass, field
from typing import Set, Tuple

from config import GRID_ROWS, GRID_COLS, OBSTACLE_SIZES, OBS_MOVE_EVERY

# Cardinal directions only — predictable corridor movement
_DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]   # E S W N

WANDER_CHANGE = 35   # ticks between spontaneous direction changes


@dataclass
class SimObstacle:
    row: int
    col: int
    height: int
    width: int
    _dir_idx: int = field(default_factory=lambda: random.randint(0, 3))
    _move_tick: int = 0
    _wander_tick: int = field(default_factory=lambda: random.randint(0, WANDER_CHANGE))

    @staticmethod
    def spawn_random(grid) -> "SimObstacle":
        h, w = random.choice(OBSTACLE_SIZES)
        for _ in range(400):
            r = random.randint(1, GRID_ROWS - h - 1)
            c = random.randint(1, GRID_COLS - w - 1)
            if all(grid.is_passable(r + dr, c + dc)
                   for dr in range(h) for dc in range(w)):
                return SimObstacle(row=r, col=c, height=h, width=w)
        return SimObstacle(row=1, col=1, height=2, width=2)

    def tick(self, grid,
             occupied: Set[Tuple[int, int]]) -> None:
        """Advance one simulation tick; update `occupied` in-place."""
        self._move_tick += 1
        self._wander_tick += 1

        # Spontaneous direction change
        if self._wander_tick >= WANDER_CHANGE:
            self._wander_tick = 0
            self._dir_idx = random.randint(0, 3)

        if self._move_tick < OBS_MOVE_EVERY:
            return
        self._move_tick = 0

        # Try preferred direction, then rotations, then reverse
        tried = set()
        for _ in range(4):
            dr, dc = _DIRS[self._dir_idx % 4]
            nr, nc = self.row + dr, self.col + dc

            if self._can_place(nr, nc, grid, occupied):
                # Move
                occupied -= self.all_cells()
                self.row, self.col = nr, nc
                occupied |= self.all_cells()
                return

            tried.add(self._dir_idx % 4)
            # Rotate 90° (random CW or CCW)
            self._dir_idx = (self._dir_idx + random.choice([1, 3])) % 4

        # Completely stuck — reverse
        self._dir_idx = (self._dir_idx + 2) % 4

    def _can_place(self, r: int, c: int,
                   grid, occupied: Set[Tuple[int, int]]) -> bool:
        """True if the obstacle can be placed at (r, c) without collision."""
        my_cells = self.all_cells()
        for dr in range(self.height):
            for dc in range(self.width):
                nr, nc = r + dr, c + dc
                if not grid.is_passable(nr, nc):
                    return False
                if (nr, nc) in occupied and (nr, nc) not in my_cells:
                    return False
        return True

    def all_cells(self) -> Set[Tuple[int, int]]:
        return {(self.row + dr, self.col + dc)
                for dr in range(self.height) for dc in range(self.width)}
