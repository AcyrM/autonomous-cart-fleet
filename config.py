GRID_COLS = 192
GRID_ROWS = 108
CELL_SIZE = 10           # nominal px per cell (reference; canvas scales dynamically)
MINI_CELL_SIZE = 3       # px per cell in routine dialog mini-map (192×3=576, 108×3=324)

# Simulation timing
SIM_TICK_MS = 80         # ms per tick  ≈ 12.5 Hz base clock
CART_MOVE_EVERY = 2      # ticks between 1-cell cart moves
OBS_MOVE_EVERY  = 3      # ticks between 1-cell obstacle moves

# Entity sizes (height, width) in grid cells
CART_SIZE = (2, 2)
OBSTACLE_SIZES = [(3, 3), (4, 4), (3, 4), (4, 3), (2, 4), (4, 2)]
NUM_OBSTACLES = 8

# Collision recovery
BLOCK_REPLAN_TICKS = 25  # ticks stuck before replanning (~2 s at 1×)

ROUTINES_FILE = "data/routines.json"

CART_COLORS = [
    "#3b82f6",  # blue
    "#06b6d4",  # cyan
    "#10b981",  # emerald
    "#a78bfa",  # violet
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#6366f1",  # indigo
    "#84cc16",  # lime
    "#f97316",  # orange
    "#38bdf8",  # sky
]

EMPTY = 0
WALL  = 1
