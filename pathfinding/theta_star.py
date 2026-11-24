"""
Theta* any-angle pathfinding on a grid.

Theta* extends A* by checking line-of-sight (LOS) to allow parent pointers
to skip across multiple cells, producing smoother, shorter paths than A*.
"""

import heapq
import math
from typing import Callable, List, Optional, Set, Tuple

Pos = Tuple[int, int]


def line_of_sight(
    passable: Callable[[int, int], bool],
    blocked: Set[Pos],
    s: Pos,
    e: Pos,
) -> bool:
    """
    Supercover grid traversal: returns True when the straight line from s to e
    does not pass through any non-passable cell.

    Uses the integer supercover DDA so every grid cell the segment crosses is
    checked, avoiding the floating-point aliasing issues of naive lerp methods.
    """
    r0, c0 = s
    r1, c1 = e
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    r, c = r0, c0

    # Total steps = sum of manhattan components (supercover visits all cells)
    steps = dr + dc
    err = dr - dc
    dr2 = dr * 2
    dc2 = dc * 2

    for _ in range(steps + 1):
        if (r, c) != s:
            if not passable(r, c) or (r, c) in blocked:
                return False
        if err > 0:
            r += sr
            err -= dc2
        elif err < 0:
            c += sc
            err += dr2
        else:
            # Diagonal step: check both corner cells to prevent corner cutting
            if not passable(r + sr, c) or (r + sr, c) in blocked:
                return False
            if not passable(r, c + sc) or (r, c + sc) in blocked:
                return False
            r += sr
            c += sc

    return True


def theta_star(
    passable: Callable[[int, int], bool],
    rows: int,
    cols: int,
    start: Pos,
    goal: Pos,
    blocked: Optional[Set[Pos]] = None,
) -> Optional[List[Pos]]:
    """
    Returns a list of (row, col) positions from start to goal (inclusive),
    or None if no path exists.

    Parameters
    ----------
    passable : callable (row, col) -> bool — True when the cell can be entered.
    rows, cols : grid dimensions for bounds checking.
    blocked : extra cells to treat as impassable (e.g., other carts).
    """
    if blocked is None:
        blocked = set()

    def ok(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and passable(r, c) and (r, c) not in blocked

    if not ok(*start) or not ok(*goal):
        return None
    if start == goal:
        return [start]

    def h(pos: Pos) -> float:
        return math.hypot(goal[0] - pos[0], goal[1] - pos[1])

    # (f, g, pos)
    open_heap: List[Tuple[float, float, Pos]] = []
    g: dict[Pos, float] = {start: 0.0}
    parent: dict[Pos, Pos] = {start: start}
    closed: Set[Pos] = set()

    heapq.heappush(open_heap, (h(start), 0.0, start))

    _DIRS = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0,  1),
             (1, -1),  (1,  0), (1,  1)]

    while open_heap:
        _, g_cur, cur = heapq.heappop(open_heap)

        if cur in closed:
            continue
        closed.add(cur)

        if cur == goal:
            path: List[Pos] = []
            node = goal
            while node != parent[node]:
                path.append(node)
                node = parent[node]
            path.append(start)
            return list(reversed(path))

        par = parent[cur]

        for dr, dc in _DIRS:
            nb = (cur[0] + dr, cur[1] + dc)
            if not ok(*nb) or nb in closed:
                continue

            # Path 2 — try to connect nb directly to grandparent via LOS
            if line_of_sight(passable, blocked, par, nb):
                new_g = g[par] + math.hypot(nb[0] - par[0], nb[1] - par[1])
                if new_g < g.get(nb, math.inf):
                    g[nb] = new_g
                    parent[nb] = par
                    heapq.heappush(open_heap, (new_g + h(nb), new_g, nb))
            else:
                # Path 1 — standard A* update through current node
                new_g = g_cur + math.hypot(nb[0] - cur[0], nb[1] - cur[1])
                if new_g < g.get(nb, math.inf):
                    g[nb] = new_g
                    parent[nb] = cur
                    heapq.heappush(open_heap, (new_g + h(nb), new_g, nb))

    return None
