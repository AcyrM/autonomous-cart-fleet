import numpy as np
from config import GRID_ROWS, GRID_COLS, EMPTY, WALL


class Grid:
    def __init__(self, rows: int = GRID_ROWS, cols: int = GRID_COLS):
        self.rows = rows
        self.cols = cols
        self._cells = np.zeros((rows, cols), dtype=np.int8)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_passable(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self._cells[row][col] == EMPTY

    def set_wall(self, row: int, col: int, on: bool = True) -> None:
        if self.in_bounds(row, col):
            self._cells[row][col] = WALL if on else EMPTY

    def toggle_wall(self, row: int, col: int) -> None:
        if self.in_bounds(row, col):
            self._cells[row][col] = EMPTY if self._cells[row][col] == WALL else WALL

    def is_wall(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self._cells[row][col] == WALL

    def cells(self) -> np.ndarray:
        return self._cells

    def to_list(self) -> list:
        return self._cells.tolist()

    def from_list(self, data: list) -> None:
        self._cells = np.array(data, dtype=np.int8)
