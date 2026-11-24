"""
Synthetic test videos with exact ground truth, built from the real recordings.

Every frame is a real empty-floor frame ("Background 1", played back and forth,
so sensor noise, the lamp hotspot and the camera's own drift are real) with
real sprites composited on top:
  • cart  — cut from "Just cart 1"; extra carts get their marker re-rendered
            with another ArUco ID
  • obstacles — toolbox, phones, wallets, card, cardboard cut from "Setup 1"
            and "Background and objects 1"

Effects modelled on what the recordings show:
  • local lighting — sprites are re-lit by the floor's illumination map, so
    they brighten under the hotspot
  • glossy marker washout — inside the hotspot the marker's black cells fade
    to light grey, as the printed paper does on camera
  • motion blur along the cart's velocity, camera drift (smooth random
    similarity transform), exposure drift, sensor noise

Carts drive like the real ones (turn in place, then drive) between random
waypoints that keep clear of the obstacles.

Usage:
    python tools/synth_videos.py --out videos/synthetic
    python tools/synth_videos.py --out videos/synthetic --scenarios glare stress --count 3 --seconds 30

Each video <name>.mp4 is written with <name>.json holding, per frame, every
cart's marker id / centre / heading / corners and every obstacle polygon (all
in output-frame pixels). Evaluate with:
    python tools/eval_videos.py videos/synthetic/*.mp4 --background "videos/Background 1.mp4"
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulation.photoreal import (                                   # noqa: E402
    FLOOR_BOTTOM, MARKER_MINUS_TRAVEL, VIDEOS, CameraDrift, Sprite,
    _alpha_polygon, build_assets, composite, place, remark,
)

FPS = 30
CART_GAP_PX = 190            # simulated carts keep at least this far apart


# ── Motion ────────────────────────────────────────────────────────────────────

@dataclass
class CartSim:
    marker_id: int
    sprite: Sprite
    gloss: np.ndarray                   # sprite-sized marker gloss mask
    x: float
    y: float
    heading: float                      # travel direction (deg, math convention)
    speed: float                        # px/frame cruise speed
    turn_rate: float                    # deg/frame
    target: Optional[Tuple[float, float]] = None
    pause: int = 0
    v: float = 0.0                      # current speed
    vx: float = 0.0
    vy: float = 0.0
    route: List[Tuple[float, float]] = field(default_factory=list)  # forced waypoints
    waited: int = 0                     # frames spent yielding to another cart

    def step(self, rng, pick_target, others: List["CartSim"] = ()) -> None:
        if self.pause > 0:
            self.pause -= 1
            self.v = self.vx = self.vy = 0.0
            return
        if self.target is None:
            self.target = self.route.pop(0) if self.route else pick_target(self)
        tx, ty = self.target
        dist    = math.hypot(tx - self.x, ty - self.y)
        if dist < max(self.speed, 6):
            self.target = None
            if rng.random() < 0.25:
                self.pause = int(rng.integers(10, 45))
            return
        bearing = math.degrees(math.atan2(-(ty - self.y), tx - self.x))
        err     = (bearing - self.heading + 180) % 360 - 180
        if abs(err) > 12:                      # turn in place, like the real control
            self.heading += math.copysign(min(abs(err), self.turn_rate), err)
            self.v = max(0.0, self.v - 1.5)
        else:
            self.heading += 0.3 * err + rng.normal(0, 0.4)
            self.v = min(self.speed, self.v + 0.8)
        self.heading = (self.heading + 180) % 360 - 180
        r = math.radians(self.heading)
        # Yield to other carts: stop if the next step gets too close; after a
        # while give up on this target
        nx, ny = self.x + self.v * math.cos(r), self.y - self.v * math.sin(r)
        if any(math.hypot(nx - o.x, ny - o.y) < CART_GAP_PX and
               math.hypot(nx - o.x, ny - o.y) < math.hypot(self.x - o.x, self.y - o.y)
               for o in others if o is not self):
            self.v = self.vx = self.vy = 0.0
            self.waited += 1
            if self.waited > 45:
                self.target, self.waited = None, 0
            return
        self.waited = 0
        self.vx, self.vy = self.v * math.cos(r), -self.v * math.sin(r)
        self.x += self.vx
        self.y += self.vy


# ── Scenarios ─────────────────────────────────────────────────────────────────

SCENARIOS = {
    # name: carts, obstacles, gloss, drift px, drift deg, speed range, noise σ,
    #       exposure amplitude, blur factor, special
    "basic":    dict(carts=1, obstacles=0, gloss=0.8, drift=8,  rot=0.0, speed=(4, 9),
                     noise=1.5, exposure=0.03, blur=0.5),
    "glare":    dict(carts=1, obstacles=0, gloss=1.0, drift=8,  rot=0.0, speed=(4, 9),
                     noise=1.5, exposure=0.03, blur=0.5, special="glare"),
    "obstacles": dict(carts=1, obstacles=6, gloss=0.8, drift=10, rot=0.1, speed=(4, 9),
                      noise=1.5, exposure=0.04, blur=0.5),
    "multi":    dict(carts=3, obstacles=5, gloss=0.8, drift=10, rot=0.1, speed=(4, 9),
                     noise=1.5, exposure=0.04, blur=0.5),
    "dynamic":  dict(carts=1, obstacles=5, gloss=0.8, drift=10, rot=0.1, speed=(4, 9),
                     noise=1.5, exposure=0.04, blur=0.5, special="dynamic"),
    "stress":   dict(carts=2, obstacles=6, gloss=1.0, drift=35, rot=0.6, speed=(8, 16),
                     noise=4.0, exposure=0.10, blur=1.0),
}


@dataclass
class PlacedObstacle:
    patch: np.ndarray
    x0: int
    y0: int
    polygon: np.ndarray          # frame px (before camera drift)
    src_illum: float
    appears: int = 0             # frame index range in which it is present
    disappears: int = 10 ** 9


def generate(name: str, cfg: dict, seconds: float, seed: int, out_dir: Path,
             cart: Sprite, obstacle_sprites: List[Sprite], illum: np.ndarray,
             floors: List[np.ndarray]) -> Path:
    rng    = np.random.default_rng(seed)
    H, W   = illum.shape
    n_fr   = int(seconds * FPS)
    floor_y1 = FLOOR_BOTTOM * H - 40
    margin = 70

    # Obstacles: random poses, not overlapping each other
    placed: List[PlacedObstacle] = []
    occ = np.zeros((H, W), np.uint8)
    for k in range(cfg["obstacles"] * 20):
        if len(placed) >= cfg["obstacles"]:
            break
        spr = obstacle_sprites[int(rng.integers(len(obstacle_sprites)))]
        x, y = rng.uniform(margin + 60, W - margin - 60), rng.uniform(margin + 60, floor_y1 - 60)
        patch, x0, y0, _ = place(spr, x, y, float(rng.uniform(-180, 180)), (H, W))
        poly = _alpha_polygon(patch, x0, y0)
        m = np.zeros_like(occ)
        cv2.fillPoly(m, [poly.astype(np.int32)], 1)
        if poly[:, 1].max() > floor_y1 or (cv2.dilate(m, np.ones((61, 61), np.uint8)) & occ).any():
            continue
        occ |= m
        placed.append(PlacedObstacle(patch, x0, y0, poly, spr.illum))
    if cfg.get("special") == "dynamic" and len(placed) >= 2:
        placed[0].appears    = n_fr // 3          # dropped in
        placed[1].disappears = 2 * n_fr // 3      # picked up

    clearance = cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (181, 181)))

    def free(x, y) -> bool:
        return margin <= x <= W - margin and margin <= y <= floor_y1 and not clearance[int(y), int(x)]

    def pick_target(c: CartSim) -> Tuple[float, float]:
        for _ in range(200):
            tx, ty = rng.uniform(margin, W - margin), rng.uniform(margin, floor_y1)
            if not free(tx, ty) or math.hypot(tx - c.x, ty - c.y) < 150:
                continue
            ts = np.linspace(0, 1, 40)
            if all(free(c.x + (tx - c.x) * t, c.y + (ty - c.y) * t) for t in ts):
                return tx, ty
        return c.x, c.y

    # Carts
    hot = np.unravel_index(np.argmax(cv2.GaussianBlur(illum, (0, 0), 30)), illum.shape)
    carts: List[CartSim] = []
    for i in range(cfg["carts"]):
        mid = 1 + i
        spr = remark(cart, mid)
        gloss = np.zeros(spr.rgba.shape[:2], np.float32)
        cv2.fillConvexPoly(gloss, spr.marker.astype(np.int32), 1.0)
        gloss = cv2.GaussianBlur(gloss, (7, 7), 0)
        for _ in range(500):
            x, y = rng.uniform(margin, W - margin), rng.uniform(margin, floor_y1)
            if free(x, y) and all(math.hypot(x - c.x, y - c.y) > 250 for c in carts):
                break
        c = CartSim(mid, spr, gloss, x, y, float(rng.uniform(-180, 180)),
                    float(rng.uniform(*cfg["speed"])), float(rng.uniform(3, 7)))
        if cfg.get("special") == "glare":
            hy, hx = hot
            # Cross the hotspot repeatedly from varying directions
            for _ in range(6):
                c.route.append((float(hx + rng.normal(0, 40)), float(hy + rng.normal(0, 40))))
                c.route.append(pick_target(c))
        carts.append(c)

    drift = CameraDrift(rng, cfg["drift"], cfg["rot"])
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{name}.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gt_frames = []
    exposure_phase = rng.uniform(0, 2 * math.pi)
    n_floor = len(floors)

    for f in range(n_fr):
        # Floor frames played back and forth to keep real noise and drift
        j     = f % (2 * n_floor - 2)
        frame = cv2.imdecode(floors[j if j < n_floor else 2 * n_floor - 2 - j],
                             cv2.IMREAD_COLOR)

        gt_obs = []
        for ob in placed:
            if ob.appears <= f < ob.disappears:
                composite(frame, ob.patch, ob.x0, ob.y0, illum, ob.src_illum)
                gt_obs.append(ob.polygon)

        gt_carts = []
        for c in carts:
            c.step(rng, pick_target, carts)
            rot = c.heading + MARKER_MINUS_TRAVEL - c.sprite.marker_angle
            patch, x0, y0, M = place(c.sprite, c.x, c.y, rot, (H, W))
            gl, _, _, _ = place(Sprite(np.dstack([c.gloss] * 4), c.sprite.anchor, 0),
                                c.x, c.y, rot, (H, W))
            speed = math.hypot(c.vx, c.vy)
            composite(frame, patch, x0, y0, illum, c.sprite.illum, gl[..., 0],
                      cfg["gloss"], (speed * cfg["blur"], c.heading))
            corners = cv2.transform(c.sprite.marker.reshape(-1, 1, 2), M).reshape(-1, 2)
            gt_carts.append((c.marker_id, corners, c.heading, c.v))

        # Camera: drift, exposure, sensor noise
        Mc = drift.step((W / 2, H / 2))
        if cfg["drift"] > 0 or cfg["rot"] > 0:
            frame = cv2.warpAffine(frame, Mc, (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
        gain  = 1 + cfg["exposure"] * math.sin(2 * math.pi * f / (FPS * 7) + exposure_phase)
        noisy = frame.astype(np.float32) * gain + rng.normal(0, cfg["noise"], frame.shape)
        frame = np.clip(noisy, 0, 255).astype(np.uint8)
        writer.write(frame)

        def tf(pts):
            return cv2.transform(np.asarray(pts, np.float32).reshape(-1, 1, 2), Mc).reshape(-1, 2)
        rec = {"carts": [], "obstacles": []}
        for mid, corners, heading, v in gt_carts:
            cc = tf(corners)
            cx, cy = cc.mean(axis=0)
            travel = heading + Mc_angle(Mc)
            rec["carts"].append({
                "id": mid, "cx": round(float(cx), 2), "cy": round(float(cy), 2),
                # angle = marker heading, as CartDetector measures it
                "angle": round(_wrap(travel + MARKER_MINUS_TRAVEL), 2),
                "travel_angle": round(_wrap(travel), 2),
                "speed": round(float(v), 2),
                "corners": np.round(cc, 1).tolist(),
                "in_view": bool(0 <= cx < W and 0 <= cy < H and
                                (cc[:, 0] >= 0).all() and (cc[:, 0] < W).all() and
                                (cc[:, 1] >= 0).all() and (cc[:, 1] < H).all()),
            })
        rec["obstacles"] = [np.round(tf(p), 1).tolist() for p in gt_obs]
        gt_frames.append(rec)

    writer.release()
    (out_dir / f"{name}.json").write_text(json.dumps({
        "fps": FPS, "width": W, "height": H, "scenario": cfg, "seed": seed,
        "floor_bottom": FLOOR_BOTTOM, "frames": gt_frames}))
    return video


def _wrap(a: float) -> float:
    return float((a + 180.0) % 360.0 - 180.0)


def Mc_angle(M: np.ndarray) -> float:
    """Image-rotation angle of a similarity matrix, CCW positive (math convention)."""
    return math.degrees(math.atan2(-M[1, 0], M[0, 0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=VIDEOS / "synthetic")
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS),
                    choices=list(SCENARIOS))
    ap.add_argument("--count", type=int, default=1, help="videos per scenario")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cart, obstacles, illum, floors = build_assets(args.out / "assets")
    print(f"assets: cart + {len(obstacles)} obstacle sprites, {len(floors)} floor frames")
    for s_i, name in enumerate(args.scenarios):
        for k in range(args.count):
            seed  = args.seed + 1000 * s_i + k
            label = name if args.count == 1 else f"{name}_{k:02d}"
            path  = generate(label, SCENARIOS[name], args.seconds, seed, args.out,
                             cart, obstacles, illum, floors)
            print(f"  wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()
