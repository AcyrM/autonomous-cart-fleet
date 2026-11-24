"""
Default factory floor for simulation — 192 cols × 108 rows.

Layout: outer border walls + scattered rectangular stand groups.
All interior starts passable; named zones place solid wall rectangles.

Zone A (rows  4–15): 6 top storage racks
Zone B (rows 22–37): 7 heavy machinery blocks
Zone C (rows 44–57): 5 large processing machines
Zone D (rows 64–67): 4 conveyor line segments
Zone E (rows 74–85): 6 middle storage racks
Zone F (rows 92–99): 10 workstations

Every zone is separated by ≥6 clear rows; horizontal gaps between stands are
≥7 cells so 2×2 carts can pass comfortably in both directions.
"""

from config import GRID_ROWS, GRID_COLS

# (row_start, row_end, col_start, col_end) — inclusive, filled as wall
_STANDS = [
    # ── Zone A: top storage racks (rows 4-15, 12 tall × 24 wide) ─────
    ( 4, 15,   4,  27), ( 4, 15,  36,  59), ( 4, 15,  68,  91),
    ( 4, 15, 100, 123), ( 4, 15, 132, 155), ( 4, 15, 164, 187),

    # ── Zone B: heavy machinery (rows 22-37, 16 tall × 20 wide) ──────
    (22, 37,   4,  23), (22, 37,  31,  50), (22, 37,  58,  77),
    (22, 37,  85, 104), (22, 37, 112, 131), (22, 37, 139, 158),
    (22, 37, 166, 185),

    # ── Zone C: processing machines (rows 44-57, 14 tall × 28 wide) ──
    (44, 57,   4,  31), (44, 57,  43,  70), (44, 57,  82, 109),
    (44, 57, 121, 148), (44, 57, 160, 187),

    # ── Zone D: conveyor segments (rows 64-67, 4 tall × 40 wide) ─────
    (64, 67,   4,  43), (64, 67,  52,  91),
    (64, 67, 100, 139), (64, 67, 148, 187),

    # ── Zone E: middle storage racks (rows 74-85, 12 tall × 24 wide) ─
    (74, 85,   4,  27), (74, 85,  36,  59), (74, 85,  68,  91),
    (74, 85, 100, 123), (74, 85, 132, 155), (74, 85, 164, 187),

    # ── Zone F: workstations (rows 92-99, 8 tall × 14 wide) ──────────
    (92, 99,   4,  17), (92, 99,  23,  36), (92, 99,  42,  55),
    (92, 99,  61,  74), (92, 99,  80,  93), (92, 99,  99, 112),
    (92, 99, 118, 131), (92, 99, 137, 150), (92, 99, 156, 169),
    (92, 99, 175, 188),
]


def generate_default_map(grid) -> None:
    """Fill grid with the factory floor layout."""
    grid._cells[:] = 0  # all passable

    # Outer border
    grid._cells[0, :] = 1
    grid._cells[GRID_ROWS - 1, :] = 1
    grid._cells[:, 0] = 1
    grid._cells[:, GRID_COLS - 1] = 1

    for r0, r1, c0, c1 in _STANDS:
        grid._cells[r0:r1 + 1, c0:c1 + 1] = 1
