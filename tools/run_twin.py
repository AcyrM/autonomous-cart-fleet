"""
Headless closed-loop run of the full production stack in the digital twin.

The real ProdEngine (tracking, obstacle detection, Theta* planning, steering)
drives simulated carts through VirtualCamera / VirtualBTController frames
(simulation/digital_twin.py). Carts get random missions on the free floor;
the run reports how many succeed and how well everything is tracked, using
the simulator's ground truth.

Usage:
    python tools/run_twin.py --minutes 3
    python tools/run_twin.py --minutes 2 --carts 3 --obstacles 6 --dynamic --record twin.mp4
    python tools/run_twin.py --no-calibrate-heading   # what happens without 🧭 calibration

Simulated time runs as fast as the PC allows (not real time).
"""

import argparse
import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2                                                        # noqa: E402
import numpy as np                                                # noqa: E402
from PyQt6.QtWidgets import QApplication                          # noqa: E402

from config import GRID_COLS, GRID_ROWS                           # noqa: E402
from core.cart import Cart, CartState                             # noqa: E402
from core.grid import Grid                                        # noqa: E402
from core.routine import SCHEDULE_TIME, Routine                   # noqa: E402
from simulation import templates                                  # noqa: E402
from core.routine_manager import RoutineManager                   # noqa: E402
from production.calibration import FloorCalibration               # noqa: E402
from production.cart_detector import CartDetector                 # noqa: E402
from production.obstacle_detector import ObstacleDetector         # noqa: E402
from production.prod_engine import ProdEngine                     # noqa: E402
from simulation.digital_twin import (                             # noqa: E402
    TwinConfig, VirtualBTController, VirtualWorld, bgr_to_qimage,
    calibrate_to_world, mission_goal,
)

MISSION_TIMEOUT_S = 60.0
GHOST_GRACE_S     = 2.0
CART_COLORS_BGR = [(246, 130, 59), (212, 182, 6), (129, 185, 16), (250, 139, 167)]


def build_system(cfg: TwinConfig, workdir: Path, fleet: dict = None):
    """The production stack wired to a VirtualWorld, as MainWindow does in
    virtual mode. With a fleet template, its carts (and their routes, in the
    engine's routine manager) replace cfg.n_carts random ones.
    Returns (world, engine, bt, calibration, carts by marker)."""
    if fleet is not None:
        cfg.n_carts = 0
    world = VirtualWorld(cfg)
    bt    = VirtualBTController(world)
    grid  = Grid(GRID_ROWS, GRID_COLS)
    calib = FloorCalibration(workdir / "calibration.json")
    calibrate_to_world(world, calib, grid)
    od  = ObstacleDetector(workdir / "floor_background.npz")
    rm  = RoutineManager(None)
    eng = ProdEngine(grid, rm, bt,
                     CartDetector(), calib, person_detector=None, obstacle_detector=od)
    carts = {}
    if fleet is not None:
        for spec in templates.fleet_specs(fleet, world, calib):
            mid = spec["marker_id"]
            world.add_cart(mid, at=(spec["x"], spec["y"]), heading=spec["heading"],
                           params=spec["params"])
            cart = Cart.make(spec["name"], spec.get("color", "#3b82f6"), 0, 0)
            cart.marker_id = mid
            cart.heading_offset = spec.get("heading_offset", 0.0)
            eng.add_cart(cart)
            bt.assign(cart.id, bt.device_for_marker(mid))
            bt.connect_cart(cart.id)
            carts[mid] = cart
            for d in spec["routines"]:
                rm.add(Routine.from_dict({**d, "cart_id": cart.id}))
        return world, eng, bt, calib, carts
    for mid in sorted(world.carts):
        # Grid position is a placeholder, as in the app: the engine localises
        # carts from the image only
        cart = Cart.make(f"V{mid}", "#3b82f6", 0, 0)
        cart.marker_id = mid
        eng.add_cart(cart)
        bt.assign(cart.id, bt.device_for_marker(mid))
        bt.connect_cart(cart.id)
        carts[mid] = cart
    return world, eng, bt, calib, carts


class Mission:
    def __init__(self, cart: Cart, goal_px, goal_cell, t0: float):
        self.cart, self.goal_px, self.goal_cell, self.t0 = cart, goal_px, goal_cell, t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=2.0, help="simulated time")
    ap.add_argument("--carts", type=int, default=2)
    ap.add_argument("--obstacles", type=int, default=5)
    ap.add_argument("--dynamic", action="store_true",
                    help="every 20 s a person comes in and sets an object down or picks one up")
    ap.add_argument("--walkers", type=float, default=None,
                    help="people walking through per minute (some stop, drop or pick up); "
                         "default 0, or the scenario template's value")
    ap.add_argument("--fleet", help="fleet template name: its carts run their routes "
                                    "(instead of random missions)")
    ap.add_argument("--scenario", help="scenario template name: its objects and foot "
                                       "traffic (instead of --obstacles random ones)")
    ap.add_argument("--hide-cart", type=float, default=None, metavar="SECONDS",
                    help="at this time cart V1 stops appearing in the image (it is still "
                         "physically there) — tests that it isn't forgotten")
    ap.add_argument("--marker-offset", type=float, default=90.0,
                    help="marker heading minus travel direction, multiple of 90 "
                         "(the recorded cart: 90)")
    ap.add_argument("--drift", type=float, default=0.0, help="camera drift, px")
    ap.add_argument("--no-calibrate-heading", dest="calibrate_heading", action="store_false",
                    help="skip the engine's heading calibration before the missions "
                         "(as an operator who forgot to press 🧭)")
    ap.add_argument("--calib-noise", type=float, default=2.0,
                    help="calibration click error σ, px")
    ap.add_argument("--scale", type=float, default=0.5, help="render resolution")
    ap.add_argument("--orientation", choices=["landscape", "portrait"], default="landscape",
                    help="camera view: 16:9 landscape (default) or as recorded")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", type=Path, help="write an annotated .mp4")
    ap.add_argument("--log", type=Path, help="write the engine's event log (with sim time)")
    args = ap.parse_args()

    _app = QApplication.instance() or QApplication(sys.argv)   # Qt objects need one
    cfg = TwinConfig(n_carts=args.carts, n_obstacles=args.obstacles, scale=args.scale,
                     marker_offset=args.marker_offset, drift_px=args.drift,
                     calib_noise_px=args.calib_noise, walkers_per_min=args.walkers or 0.0,
                     orientation=args.orientation,
                     seed=args.seed)
    workdir = Path(tempfile.mkdtemp(prefix="twin_"))
    fleet    = templates.load("fleet", args.fleet) if args.fleet else None
    scenario = templates.load("scenario", args.scenario) if args.scenario else None
    world, eng, bt, calib, carts = build_system(cfg, workdir, fleet)
    routes   = {mid: eng._rm.for_cart(c.id) for mid, c in carts.items()}
    route_mode = any(routes.values())
    next_route = {mid: 0 for mid in carts}
    rng  = np.random.default_rng(args.seed + 1)
    logs = []
    eng.log_message.connect(lambda m: logs.append((world.t, m)))

    dt       = 1.0 / cfg.fps
    n_frames = int(args.minutes * 60 * cfg.fps)
    missions = {}                     # marker id → Mission
    results  = {"ok": 0, "timeout": 0, "unplannable": 0, "times": [], "goal_err": []}
    pos_err, ang_err, untracked = [], [], 0
    writer = None
    obstacles_placed = False
    cal_retry_at = 0.0
    calibrated = set()
    if fleet is not None:
        # Carts whose template carries the 🧭 heading offset need no calibration
        calibrated = {c.id for c in carts.values() if c.heading_offset}
    perm = {"pairs": 0, "kept": 0, "blocked": 0, "ghost_frames": 0, "checks": 0,
            "moving_frames": 0}
    t_wall = time.perf_counter()

    for f in range(n_frames):
        world.step(dt)
        frame = world.render()
        eng.on_frame(bgr_to_qimage(frame))

        if f == 3:
            eng.capture_floor()
        # Operator set-up on the empty floor: heading calibration (🧭) first;
        # a cart refused (no room) is retried every 3 s and gets no missions
        if args.calibrate_heading and eng.floor_captured and not eng._calibrating and \
                world.t >= cal_retry_at:
            cal_retry_at = world.t + 3.0
            for cart in carts.values():
                if cart.id not in calibrated:
                    eng.calibrate_heading(cart.id)
        if len(calibrated) < len(carts):
            for cart in carts.values():
                if cart.id not in calibrated and any(
                        f"✓ {cart.name}: heading offset" in m for _, m in logs):
                    calibrated.add(cart.id)
        if eng.floor_captured and not obstacles_placed and not eng._calibrating and \
                (not args.calibrate_heading or calibrated or world.t > 15):
            if scenario is not None:
                templates.apply_scenario(scenario, world)
                if args.walkers is not None:
                    world.cfg.walkers_per_min = args.walkers
            else:
                for _ in range(cfg.n_obstacles):
                    world.add_obstacle()
            obstacles_placed = True
            world.walkers_enabled = True
        if args.dynamic and obstacles_placed and f % (20 * cfg.fps) == 0 and f > 0:
            # Objects only come and go with a person carrying them
            kind = "drop" if (rng.random() < 0.5 or not world.obstacles) else "pickup"
            world.spawn_walker(kind)
        if args.hide_cart is not None and world.t >= args.hide_cart and 1 in world.carts \
                and not world.carts[1].hidden:
            world.carts[1].hidden = True
            logs.append((world.t, "[harness] V1 disappears from the image (still on the floor)"))
        if obstacles_placed and f % 5 == 0:
            check_permanence(world, eng, perm)

        # Tracking accuracy against ground truth
        truth = world.truth()
        dets  = {d.marker_id: d for d in eng.detections if d.source != "memory"}
        for mid, (tx, ty, ta) in truth.items():
            d = dets.get(mid)
            if d is None:
                untracked += 1
            else:
                pos_err.append(math.hypot(d.cx - tx, d.cy - ty))
                ang_err.append(abs((d.angle_deg - ta + 180) % 360 - 180))

        calibrating = bool(eng._calibrating)

        # Missions: one at a time per cart, random free goals
        if obstacles_placed and f > 3 * cfg.fps and not calibrating:
            for mid, cart in carts.items():
                m = missions.get(mid)
                if m is not None:
                    done = cart.state == CartState.IDLE and cart.id not in eng._paths
                    if done:
                        tx, ty, _ = truth[mid]
                        results["ok"] += 1
                        results["times"].append(world.t - m.t0)
                        results["goal_err"].append(math.hypot(tx - m.goal_px[0], ty - m.goal_px[1]))
                        del missions[mid]
                    elif world.t - m.t0 > MISSION_TIMEOUT_S:
                        results["timeout"] += 1
                        logs.append((world.t, f"[harness] {cart.name} mission timed out"))
                        eng._paths.pop(cart.id, None)
                        cart.clear_task()
                        bt.stop(cart.id)
                        del missions[mid]
                    continue
                if args.calibrate_heading and cart.id not in calibrated:
                    continue          # an operator wouldn't dispatch an uncalibrated cart
                if not any(d.marker_id == mid and d.source != "memory" for d in eng.detections):
                    continue          # an operator can't dispatch a cart the camera doesn't see
                if route_mode:
                    # The fleet's own routes, in turn (an operator pressing ▶ Now)
                    if not routes[mid]:
                        continue
                    rt = routes[mid][next_route[mid] % len(routes[mid])]
                    next_route[mid] += 1
                    cell = tuple(rt.waypoints[-1])
                    goal_px = calib.grid_to_pixel(*cell)
                else:
                    seen = [(d.cx, d.cy) for d in eng.detections if d.marker_id != mid]
                    goal = mission_goal(world, calib,
                                        seen + [mm.goal_px for mm in missions.values()])
                    if goal is None:
                        continue
                    goal_px, cell = goal
                    rt = Routine.make(cart.id, f"mission {f}", SCHEDULE_TIME)
                    rt.waypoints = [cell]
                eng.trigger_routine(rt)
                if cart.id in eng._paths:
                    missions[mid] = Mission(cart, goal_px, cell, world.t)
                else:
                    results["unplannable"] += 1

        if args.record:
            vis = annotate(frame, eng, calib, missions, cfg)
            if writer is None:
                args.record.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"),
                                         cfg.fps, (vis.shape[1], vis.shape[0]))
            writer.write(vis)

    if writer is not None:
        writer.release()
    for t, ev in world.walker_log:
        logs.append((t, f"[world] {ev}"))
    for t, mid, what in world.collision_log:
        logs.append((t, f"[harness] V{mid} COLLISION with {what}"))
    logs.sort(key=lambda e: e[0])
    if args.log:
        args.log.write_text("".join(f"{t:7.1f}s  {m}\n" for t, m in logs))
    wall = time.perf_counter() - t_wall
    collisions = sum(c.collisions for c in world.carts.values())
    lost  = sum(1 for _, m in logs if "lost from view" in m)
    stops = sum(1 for _, m in logs if "stopped —" in m)
    replans = sum(1 for _, m in logs if "replanned" in m)
    n_m = results["ok"] + results["timeout"]

    print(f"\nDigital twin — {args.minutes:g} min simulated in {wall:.0f} s "
          f"({n_frames / wall:.1f} frames/s), {cfg.n_carts} carts, "
          f"{len(world.obstacles)} obstacles, marker offset {cfg.marker_offset:g}°"
          + (f", fleet '{args.fleet}'" if args.fleet else "")
          + (f", scenario '{args.scenario}'" if args.scenario else ""))
    print(f"  missions     : {results['ok']}/{n_m} reached"
          + (f" ({results['ok'] / n_m:.0%})" if n_m else "")
          + f", {results['timeout']} timed out, {len(missions)} unfinished at the end, "
          f"{results['unplannable']} goals with no path")
    offsets = {c.name: c.heading_offset for c in carts.values() if c.heading_offset}
    if args.calibrate_heading:
        print(f"  heading cal. : {offsets or 'no offset measured'} "
              f"(true marker offset {cfg.marker_offset:g}°)")
    if results["times"]:
        print(f"  mission time : median {np.median(results['times']):.1f} s, "
              f"final distance to goal median {np.median(results['goal_err']):.1f} px "
              f"(cart radius {world.cart_radius * cfg.scale:.0f} px)")
    kinds = {}
    for _, _, what in world.collision_log:      # incl. "person (cart braking)", not counted
        kinds[what] = kinds.get(what, 0) + 1
    print(f"  collisions   : {collisions} {kinds or ''}   stops for blockage: {stops}   "
          f"replans: {replans}   lost-from-view stops: {lost}")
    if perm["pairs"]:
        n_people = sum(1 for _, e in world.walker_log if e.startswith("person enters"))
        print(f"  obstacles    : {perm['kept'] / perm['pairs']:.1%} of real obstacles kept "
              f"as static (seen or remembered), {perm['blocked'] / perm['pairs']:.1%} blocked "
              f"at all (incl. merged with a person); ghosts (>{GHOST_GRACE_S:g} s) in {perm['ghost_frames']}/"
              f"{perm['checks']} checks; people: {n_people}, seen moving in "
              f"{perm['moving_frames']} checks")
    if args.hide_cart is not None:
        mem = sum(1 for _, m in logs if "assumed still at its last position" in m)
        print(f"  hidden cart  : 'assumed still there' events {mem}; collisions with it: "
              f"{sum(1 for _, mid, w in world.collision_log if w == 'cart')}")
    if pos_err:
        tot = len(pos_err) + untracked
        print(f"  tracking     : {len(pos_err) / tot:.1%} of cart-frames, position error "
              f"median {np.median(pos_err):.1f} px / p95 {np.percentile(pos_err, 95):.1f} px, "
              f"heading error median {np.median(ang_err):.1f}° / p95 {np.percentile(ang_err, 95):.1f}°")
    if args.record:
        print(f"  video        : {args.record}")


def check_permanence(world: VirtualWorld, eng: ProdEngine, perm: dict) -> None:
    """Compare the engine's static obstacles (seen or remembered) with the
    obstacles really on the floor: kept = real ones ≥50 % covered; ghosts =
    remembered ones that cover nothing real."""
    s, sh = world.cfg.scale, (world.render_size[1] // 4, world.render_size[0] // 4)
    real = np.zeros(sh, np.uint8)
    polys = []
    for ob in world.obstacles:
        p = cv2.transform(ob.polygon.reshape(-1, 1, 2).astype(np.float32), world._Mc)
        polys.append(np.round(p.reshape(-1, 2) * s / 4).astype(np.int32))
    if polys:
        cv2.fillPoly(real, polys, 1)
    static = np.zeros(sh, np.uint8)
    anything = np.zeros(sh, np.uint8)
    remembered = []
    for ob in eng.obstacles:
        cv2.fillPoly(anything, [np.round(ob.contour / 4).astype(np.int32)], 1)
        if ob.state == "moving":
            perm["moving_frames"] += 1
            continue
        q = np.round(ob.contour / 4).astype(np.int32)
        cv2.fillPoly(static, [q], 1)
        if ob.state == "remembered":
            remembered.append((ob.track_id, q))
    perm["checks"] += 1
    for q in polys:
        m = np.zeros(sh, np.uint8)
        cv2.fillPoly(m, [q], 1)
        perm["pairs"] += 1
        perm["kept"] += int((m & static).sum() >= 0.5 * max(m.sum(), 1))
        perm["blocked"] += int((m & anything).sum() >= 0.5 * max(m.sum(), 1))
    # A remembered obstacle with nothing real under it is expected for a moment
    # (someone picked it up and is still standing there); only count ghosts
    # that persist for more than GHOST_GRACE_S
    since = perm.setdefault("ghost_since", {})
    flagged = set()
    for ob_id, q in remembered:
        m = np.zeros(sh, np.uint8)
        cv2.fillPoly(m, [q], 1)
        if (m & real).sum() < 0.2 * max(m.sum(), 1):
            flagged.add(ob_id)
            since.setdefault(ob_id, world.t)
    for ob_id in list(since):
        if ob_id not in flagged:
            del since[ob_id]
    if any(world.t - since[i] > GHOST_GRACE_S for i in flagged):
        perm["ghost_frames"] += 1


def annotate(frame, eng: ProdEngine, calib: FloorCalibration, missions, cfg) -> np.ndarray:
    vis = frame.copy()
    colours = {"visible": (255, 0, 255), "moving": (0, 140, 255), "remembered": (180, 120, 255)}
    for ob in eng.obstacles:
        col = colours.get(ob.state, (255, 0, 255))
        if ob.swept is not None:
            cv2.polylines(vis, [ob.swept.astype(np.int32)], True, col, 1)
        cv2.polylines(vis, [ob.contour.astype(np.int32)], True, col,
                      1 if ob.state == "remembered" else 2)
    for i, (cid, path) in enumerate(eng._paths.items()):
        pts = [calib.grid_to_pixel(r, c) for r, c in path]
        idx = eng._path_idx.get(cid, 0)
        for a, b in zip(pts[max(idx - 1, 0):], pts[max(idx, 1):]):
            cv2.line(vis, tuple(map(int, a)), tuple(map(int, b)), (0, 200, 255), 2)
    for m in missions.values():
        gx, gy = map(int, m.goal_px)
        cv2.drawMarker(vis, (gx, gy), (0, 200, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
    for d in eng.detections:
        col = (0, 255, 0) if d.source.startswith("marker") else (255, 160, 0)
        cv2.polylines(vis, [d.corners.astype(np.int32)], True, col, 2)
    for cart in eng.carts.values():
        if cart.state != CartState.IDLE:
            d = next((x for x in eng.detections if x.marker_id == cart.marker_id), None)
            if d is not None:
                cv2.putText(vis, cart.state.value, (int(d.cx) + 20, int(d.cy) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return vis


if __name__ == "__main__":
    main()
