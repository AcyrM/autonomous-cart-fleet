"""
Saved set-ups for the digital twin (virtual production mode), kept as JSON:

  fleet    — the carts: name, colour, marker id, where they stand and face,
             their physics (speed, motor response, dead band, wheel gains,
             marker mounting), the measured heading offset (🧭), and their
             routes (routines)
  scenario — the floor: every obstacle (which object, where, rotated how) and
             the foot traffic (people per minute)

Positions are stored as fractions of the floor area (0–1), so a template made
in landscape also loads in portrait. Route waypoints are stored the same way
and converted back to grid cells through the calibration on load.

Files live in data/virtual/templates/<fleet|scenario>/<name>.json.
"""

import json
import re
import uuid
from pathlib import Path
from typing import List

TEMPLATE_DIR = Path("data/virtual/templates")
KINDS = ("fleet", "scenario")


def _path(kind: str, name: str) -> Path:
    assert kind in KINDS
    safe = re.sub(r"[^\w\- ]+", "_", name).strip() or "template"
    return TEMPLATE_DIR / kind / f"{safe}.json"


def list_templates(kind: str) -> List[str]:
    d = TEMPLATE_DIR / kind
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


def save(kind: str, name: str, data: dict) -> Path:
    path = _path(kind, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    return path


def load(kind: str, name: str) -> dict:
    return json.loads(_path(kind, name).read_text())


# ── Fleet ─────────────────────────────────────────────────────────────────────

def fleet_snapshot(world, engine, rm, calib) -> dict:
    """The carts as they stand now, with their routes."""
    s, carts = world.cfg.scale, []
    for cart in engine.carts.values():
        vc = world.carts.get(cart.marker_id)
        if vc is None:
            continue
        u, v = world.to_norm(vc.x, vc.y)
        routes = []
        for r in rm.for_cart(cart.id):
            d = r.to_dict()
            d.pop("id", None)
            d.pop("cart_id", None)
            d.pop("waypoints", None)
            d["waypoints_norm"] = [
                list(world.to_norm(*(c / s for c in calib.grid_to_pixel(row, col))))
                for row, col in r.waypoints]
            routes.append(d)
        carts.append({"name": cart.name, "color": cart.color, "marker_id": cart.marker_id,
                      "pos_norm": [u, v], "heading": vc.heading,
                      "heading_offset": cart.heading_offset,
                      "params": world.cart_params(cart.marker_id),
                      "routines": routes})
    return {"kind": "fleet", "carts": carts}


def fleet_specs(data: dict, world, calib) -> List[dict]:
    """Template carts in world coordinates, routes as grid-cell routines
    (Routine dicts without cart_id — set it once the cart exists)."""
    s, out = world.cfg.scale, []
    for c in data.get("carts", []):
        x, y = world.from_norm(*c["pos_norm"])
        routines = []
        for d in c.get("routines", []):
            d = dict(d)
            cells = []
            for u, v in d.pop("waypoints_norm", []):
                px, py = world.from_norm(u, v)
                row, col = calib.pixel_to_grid(px * s, py * s)
                cells.append([int(round(row)), int(round(col))])
            d["waypoints"] = cells
            d["id"] = str(uuid.uuid4())[:12]
            routines.append(d)
        out.append({**c, "x": x, "y": y, "routines": routines})
    return out


# ── Scenario ──────────────────────────────────────────────────────────────────

def scenario_snapshot(world) -> dict:
    obs = []
    for ob in world.obstacles:
        u, v = world.to_norm(ob.x, ob.y)
        obs.append({"kind": ob.sprite_idx, "pos_norm": [u, v], "angle": ob.angle})
    return {"kind": "scenario", "obstacles": obs,
            "walkers_per_min": world.cfg.walkers_per_min}


def apply_scenario(data: dict, world) -> int:
    """Replace the floor's obstacles with the template's. Returns how many were placed."""
    world.clear_obstacles()
    n = 0
    for o in data.get("obstacles", []):
        x, y = world.from_norm(*o["pos_norm"])
        n += world.place_obstacle(int(o["kind"]), x, y, float(o["angle"]), check=False)
    world.cfg.walkers_per_min = float(data.get("walkers_per_min", world.cfg.walkers_per_min))
    return n
