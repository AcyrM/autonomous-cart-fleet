"""
Ready-made fleet and scenario templates for the digital twin.

Writes them to data/virtual/templates/ (overwriting templates with the same
name). Positions are floor fractions: u left→right, v top→bottom of the
floor area, laid out for the default landscape view.

Usage:
    python tools/make_templates.py
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CART_COLORS                       # noqa: E402
from core.routine import SCHEDULE_INTERVAL           # noqa: E402
from simulation import templates                     # noqa: E402

# Obstacle sprites cut from the recordings
CARD, CARDBOARD, TOOLBOX, PHONE, WALLET, PHONE2, WALLET2 = range(7)

REAL_MOUNTING = 90.0           # the recorded cart: marker at +90° to its front
DRIVE_REF = 180                # speeds below are in mm/s at DRIVE 180 (1 px ≈ 1 mm)

# Lamp hotspot in the landscape view, as a floor fraction
HOTSPOT = (0.50, 0.61)


# ── Helpers ───────────────────────────────────────────────────────────────────

def route(name, points, every_s=5, enabled=True):
    """A route that repeats: fires again `every_s` seconds after it last ran,
    as soon as its cart is free."""
    return {"name": name, "schedule_type": SCHEDULE_INTERVAL,
            "waypoints_norm": [list(p) for p in points],
            "time_of_day": None, "interval_seconds": every_s,
            "repeat": True, "enabled": enabled, "last_triggered": None}


def cart(i, pos, facing, routes, speed=200, response=0.12, deadband=40,
         left=1.0, right=1.0, mounting=REAL_MOUNTING, calibrated=True):
    """`facing` = direction it points, deg (0 = right, 90 = up). `calibrated`:
    store the 🧭 heading offset (= mounting) so no calibration is needed."""
    return {"name": f"V{i}", "color": CART_COLORS[(i - 1) % len(CART_COLORS)],
            "marker_id": i, "pos_norm": list(pos), "heading": float(facing),
            "heading_offset": mounting if calibrated else 0.0,
            "params": {"marker_offset": mounting, "px_per_unit": speed / DRIVE_REF,
                       "motor_tau": response, "deadband": deadband,
                       "gain_l": left, "gain_r": right},
            "routines": routes}


def obstacle(kind, pos, angle=0.0):
    return {"kind": kind, "pos_norm": list(pos), "angle": float(angle)}


def loop_from(corners, start):
    """The loop's corners in order, starting after corner `start` and ending on it."""
    n = len(corners)
    return [corners[(start + k) % n] for k in range(1, n + 1)]


# ── Scenarios ─────────────────────────────────────────────────────────────────

SCENARIOS = {
    "Empty floor": {"obstacles": [], "walkers_per_min": 0.0},

    # A row of objects across the middle splits the floor into two corridors
    "Divider": {"obstacles": [
        obstacle(WALLET,  (0.41, 0.50), 0), obstacle(PHONE, (0.50, 0.50), 90),
        obstacle(WALLET2, (0.59, 0.50), 0),
    ], "walkers_per_min": 0.0},

    # Objects in the way of the usual routes: carts must plan detours
    "Scattered objects": {"obstacles": [
        obstacle(TOOLBOX,   (0.50, 0.56), 30),
        obstacle(CARD,      (0.27, 0.86), -20),
        obstacle(WALLET,    (0.74, 0.86), 60),
    ], "walkers_per_min": 0.0},

    # Objects inside the lamp's reflection, incl. floor-coloured cardboard:
    # the hardest case for obstacle detection and permanence
    "Glare zone": {"obstacles": [
        obstacle(CARDBOARD, (HOTSPOT[0] - 0.02, HOTSPOT[1] + 0.03), 20),
        obstacle(CARD,      (HOTSPOT[0] + 0.13, HOTSPOT[1] + 0.02), -35),
        obstacle(PHONE2,    (HOTSPOT[0] - 0.12, HOTSPOT[1] + 0.07), 90),
    ], "walkers_per_min": 0.0},

    # A few objects and people walking through, stopping, carrying things
    "Rush hour": {"obstacles": [
        obstacle(WALLET2, (0.30, 0.09), 0),  obstacle(PHONE,  (0.70, 0.09), 0),
        obstacle(CARD,    (0.30, 0.91), 90), obstacle(WALLET, (0.70, 0.91), 0),
    ], "walkers_per_min": 8.0},
}


# ── Fleets ────────────────────────────────────────────────────────────────────

LANE = 0.5                                     # middle of the floor
LOOP = [(0.14, 0.25), (0.86, 0.25), (0.86, 0.75), (0.14, 0.75)]   # clockwise
# Carts start halfway along the loop's edges, so every corner is free
MIDS = [((LOOP[i][0] + LOOP[(i + 1) % 4][0]) / 2, (LOOP[i][1] + LOOP[(i + 1) % 4][1]) / 2)
        for i in range(4)]

FLEETS = {
    "One cart - patrol": [
        cart(1, (0.12, LANE), 0, [route("Patrol", [(0.88, LANE), (0.12, LANE)])]),
    ],

    # Same line, opposite directions: one of them has to give way
    "Two carts - head-on": [
        cart(1, (0.10, LANE), 0,   [route("Across and back", [(0.74, LANE), (0.10, LANE)])]),
        cart(2, (0.90, LANE), 180, [route("Across and back", [(0.26, LANE), (0.90, LANE)])]),
    ],

    # Two carts half a lap apart on a rectangular loop, chasing each other
    "Two carts - circuit": [
        cart(1, MIDS[3], 90,  [route("Circuit", loop_from(LOOP, 3) + [MIDS[3]])]),
        cart(2, MIDS[1], -90, [route("Circuit", loop_from(LOOP, 1) + [MIDS[1]])]),
    ],

    # Different hardware: slow, fast, and one with a weak left motor
    "Mixed fleet": [
        cart(1, (0.12, 0.28), 0, [route("Upper lane", [(0.88, 0.28), (0.12, 0.28)])],
             speed=110),
        cart(2, (0.88, 0.72), 180, [route("Lower lane", [(0.12, 0.72), (0.88, 0.72)])],
             speed=320, response=0.08),
        cart(3, (0.12, 0.50), 0, [route("Across the middle", [(0.88, 0.50), (0.12, 0.50)])],
             left=0.85, deadband=60),
    ],

    # Marker mounted facing forward and not calibrated yet: run 🧭 first,
    # then start its route with ▶ Now (it doesn't start by itself)
    "Marker at 0 degrees": [
        cart(1, (0.20, LANE), 0, [route("Back and forth", [(0.80, LANE), (0.20, LANE)],
                                        enabled=False)],
             mounting=0.0, calibrated=False),
    ],
}


def main() -> None:
    for name, data in SCENARIOS.items():
        print("scenario:", templates.save("scenario", name, {"kind": "scenario", **data}))
    for name, carts in FLEETS.items():
        for c in carts:
            for r in c["routines"]:
                r["id"] = str(uuid.uuid4())[:12]
        print("fleet:   ", templates.save("fleet", name, {"kind": "fleet", "carts": carts}))


if __name__ == "__main__":
    main()
