"""
Offline evaluation of cart tracking + obstacle detection on recorded videos.

Usage:
    python tools/eval_videos.py "videos/Setup 1.mp4" --background "videos/Background 1.mp4"
    python tools/eval_videos.py videos/*.mp4 --background "videos/Background 1.mp4" \
        --ids 1 --out eval_out

For each video prints how often the cart was located and by which source,
the false marker IDs seen, obstacle counts and per-frame cost. With --out,
writes an annotated (half-resolution) .mp4 per video plus snapshot JPEGs.

If <video>.json ground truth exists next to the video (written by
tools/synth_videos.py), it is scored too: cart recall, position / heading
error, wrong detections, and obstacle recall / precision. The tracked IDs and
the floor ROI then default to those in the ground truth.

The background clip must show the empty floor from (roughly) the same
camera position; camera drift between clips is compensated by alignment.
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production.cart_detector import CartDetector, CartTracker   # noqa: E402
from production.obstacle_detector import ObstacleDetector        # noqa: E402
from production.prod_engine import CART_FOOTPRINT_SCALE           # noqa: E402

BG_FRAMES = 20
WARMUP_FRAMES = 10        # obstacle filter settling time, not scored
GT_SCALE = 0.25           # rasterisation scale for obstacle overlap tests


class GroundTruthScore:
    """Accumulates per-frame comparisons against synthetic ground truth."""

    def __init__(self, gt: dict):
        self.gt     = gt
        self.shape  = (int(gt["height"] * GT_SCALE), int(gt["width"] * GT_SCALE))
        self.cart_frames = 0
        self.hits   = collections.Counter()      # source → matched cart-frames
        self.pos_err: list = []
        self.ang_err: list = []
        self.wrong  = 0                          # detections matching no visible cart
        self.ob_gt = self.ob_found = 0
        self.ob_det = self.ob_true = self.ob_on_cart = 0

    def _mask(self, polys) -> np.ndarray:
        m = np.zeros(self.shape, np.uint8)
        if len(polys):
            cv2.fillPoly(m, [np.round(np.asarray(p) * GT_SCALE).astype(np.int32)
                             for p in polys], 1)
        return m

    def add(self, i: int, dets, obstacles, roi_y1: float) -> None:
        fr    = self.gt["frames"][i]
        carts = {c["id"]: c for c in fr["carts"]}
        used  = set()
        for c in fr["carts"]:
            if not c["in_view"]:
                continue
            self.cart_frames += 1
            side = float(np.mean(np.linalg.norm(
                np.diff(np.asarray(c["corners"] + c["corners"][:1]), axis=0), axis=1)))
            for k, d in enumerate(dets):
                err = float(np.hypot(d.cx - c["cx"], d.cy - c["cy"]))
                if d.marker_id == c["id"] and err < 0.5 * side:
                    used.add(k)
                    self.hits[d.source] += 1
                    self.pos_err.append(err)
                    self.ang_err.append(abs((d.angle_deg - c["angle"] + 180) % 360 - 180))
                    break
        for k, d in enumerate(dets):
            c = carts.get(d.marker_id)
            if k not in used and (c is None or c["in_view"]):
                self.wrong += 1      # unknown id, or far from its cart while it's in view

        if i < WARMUP_FRAMES:
            return
        # Recall: obstacles centred on the floor area; precision: against all
        # of them (one straddling the ROI edge is still a real object)
        h_lim  = roi_y1 * self.gt["height"]
        all_ob = [np.asarray(p) for p in fr["obstacles"]]
        gt_obs = [p for p in all_ob if p[:, 1].mean() <= h_lim]
        det_m  = self._mask([o.contour for o in obstacles])
        gt_m   = self._mask(all_ob)
        cart_m = self._mask([c["corners"] for c in fr["carts"]])
        cart_m = cv2.dilate(cart_m, np.ones((15, 15), np.uint8))   # whole body, not just marker
        for p in gt_obs:
            m = self._mask([p])
            self.ob_gt += 1
            self.ob_found += int((m & det_m).sum() >= 0.5 * max(m.sum(), 1))
        for o in obstacles:
            m = self._mask([o.contour])
            area = max(int(m.sum()), 1)
            self.ob_det += 1
            if (m & gt_m).sum() >= 0.5 * area:
                self.ob_true += 1
            elif (m & cart_m).sum() >= 0.5 * area:
                self.ob_on_cart += 1

    def report(self) -> dict:
        hit = sum(self.hits.values())
        r = {"cart_recall": hit / max(self.cart_frames, 1),
             "pos_med": float(np.median(self.pos_err)) if self.pos_err else float("nan"),
             "pos_p95": float(np.percentile(self.pos_err, 95)) if self.pos_err else float("nan"),
             "ang_med": float(np.median(self.ang_err)) if self.ang_err else float("nan"),
             "ang_p95": float(np.percentile(self.ang_err, 95)) if self.ang_err else float("nan"),
             "wrong": self.wrong,
             "ob_recall": self.ob_found / self.ob_gt if self.ob_gt else float("nan"),
             "ob_precision": self.ob_true / self.ob_det if self.ob_det else float("nan"),
             "ob_on_cart": self.ob_on_cart}
        print(f"  GT carts     : recall {r['cart_recall']:6.1%} of {self.cart_frames} "
              f"in-view cart-frames  (" +
              "  ".join(f"{k}={v / max(self.cart_frames, 1):.1%}" for k, v in self.hits.most_common()) + ")")
        print(f"                 position error median {r['pos_med']:.1f} px, p95 {r['pos_p95']:.1f} px; "
              f"heading error median {r['ang_med']:.1f}°, p95 {r['ang_p95']:.1f}°; "
              f"wrong detections {self.wrong}")
        if self.ob_gt or self.ob_det:
            n_false = self.ob_det - self.ob_true - self.ob_on_cart
            print(f"  GT obstacles : recall {r['ob_recall']:6.1%}  precision {r['ob_precision']:6.1%}  "
                  f"(false obstacles: {n_false}, cart parts reported as obstacles: "
                  f"{self.ob_on_cart}, over {len(self.gt['frames']) - WARMUP_FRAMES} frames)")
        return r


def build_detector(background: str, roi, frame_shape) -> ObstacleDetector:
    od  = ObstacleDetector()
    if roi:
        h, w = frame_shape[:2]
        x0, y0, x1, y1 = roi
        od.set_roi([(x0 * w, y0 * h), (x1 * w, y0 * h), (x1 * w, y1 * h), (x0 * w, y1 * h)])
    cap = cv2.VideoCapture(background)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in np.linspace(0, n - 1, BG_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            od.capture_frame(frame)
    od.finish_capture()
    return od


def annotate(frame, dets, obstacles, scale):
    out = CartDetector().annotate(frame, dets)
    for ob in obstacles:
        cv2.polylines(out, [ob.contour.astype(np.int32)], True, (255, 0, 255), 3)
    return cv2.resize(out, None, fx=scale, fy=scale)


def evaluate(path: str, background: str, ids, out_dir: Path | None, roi) -> dict | None:
    cap     = cv2.VideoCapture(path)
    shape   = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    gt_path = Path(path).with_suffix(".json")
    score   = None
    if gt_path.exists():
        gt    = json.loads(gt_path.read_text())
        score = GroundTruthScore(gt)
        if ids is None:
            ids = sorted({c["id"] for f in gt["frames"] for c in f["carts"]})
        if roi is None:
            roi = (0, 0, 1, gt.get("floor_bottom", 1.0))
    od      = build_detector(background, roi, shape) if background else None
    tracker = CartTracker()
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30
    writer  = None
    sources = collections.Counter()
    raw_ids = collections.Counter()
    n_obs   = []
    t_track = t_obs = 0.0
    n_memory = 0
    n = 0
    snapshots = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        t0 = time.perf_counter()
        fg = od.update(frame) if od else None
        t1 = time.perf_counter()
        dets = tracker.update(frame, ids=ids, fg_mask=fg,
                              fg_scale=od.scale if od else 1.0)
        # Remembered carts (lost mid-floor) are not detections: score without them
        n_memory += sum(1 for d in dets if d.source == "memory")
        footprints = [d.footprint(CART_FOOTPRINT_SCALE) for d in dets]
        dets = [d for d in dets if d.source != "memory"]
        t2 = time.perf_counter()
        obstacles = od.obstacles(footprints) if od else []
        t_obs   += (t1 - t0) + (time.perf_counter() - t2)
        t_track += t2 - t1

        for d in tracker.raw:
            raw_ids[d.marker_id] += 1
        sources[dets[0].source if dets else "none"] += 1
        n_obs.append(len(obstacles))
        if score is not None:
            score.add(n - 1, dets, obstacles, roi[3] if roi else 1.0)

        if out_dir is not None:
            vis = annotate(frame, dets, obstacles, 0.5)
            if writer is None:
                out_dir.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out_dir / f"{Path(path).stem}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (vis.shape[1], vis.shape[0]))
            writer.write(vis)
            if n % 150 == 0:
                snapshots.append(cv2.resize(vis, None, fx=0.5, fy=0.5))

    if writer is not None:
        writer.release()
        if snapshots:
            rows = [np.hstack(snapshots[i:i + 6]) for i in range(0, len(snapshots), 6)]
            width = rows[0].shape[1]
            rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
            cv2.imwrite(str(out_dir / f"{Path(path).stem}_snapshots.jpg"), np.vstack(rows))

    located = n - sources["none"]
    print(f"\n{path}  ({n} frames)")
    print(f"  cart located : {located / n:6.1%}   " +
          "  ".join(f"{k}={v / n:.1%}" for k, v in sources.most_common() if k != "none"))
    print(f"  marker IDs   : {dict(raw_ids)}" +
          (f"   (remembered, not seen: {n_memory} cart-frames)" if n_memory else ""))
    if od:
        print(f"  obstacles    : median {int(np.median(n_obs))}, max {max(n_obs)}")
    print(f"  cost         : tracking {1000 * t_track / n:.1f} ms/frame, "
          f"obstacles {1000 * t_obs / n:.1f} ms/frame")
    return score.report() if score is not None else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--background", help="empty-floor video for obstacle detection")
    ap.add_argument("--ids", type=int, nargs="*", default=None,
                    help="marker IDs to track (default: all)")
    ap.add_argument("--out", type=Path, help="write annotated videos/snapshots here")
    ap.add_argument("--roi", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="floor area as image fractions, e.g. 0 0 1 0.85 "
                         "(the app uses the calibrated floor instead)")
    args = ap.parse_args()
    results = {}
    for v in args.videos:
        r = evaluate(v, args.background, args.ids, args.out, args.roi)
        if r is not None:
            results[Path(v).stem] = r
    if len(results) > 1:
        print(f"\n{'video':16s} {'cart rec':>8s} {'pos med':>7s} {'pos p95':>7s} "
              f"{'ang med':>7s} {'wrong':>5s} {'obs rec':>7s} {'obs prec':>8s}")
        for k, r in results.items():
            print(f"{k:16s} {r['cart_recall']:8.1%} {r['pos_med']:7.1f} {r['pos_p95']:7.1f} "
                  f"{r['ang_med']:7.1f} {r['wrong']:5d} {r['ob_recall']:7.1%} {r['ob_precision']:8.1%}")


if __name__ == "__main__":
    main()
