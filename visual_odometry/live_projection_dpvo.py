#!/usr/bin/env python3
"""
DPVO live projection with cumulative timestamps saved to CSV (timestamp, tx, tz)
and final projected image saved.
"""

import os
import sys
from multiprocessing import Process, Queue
from pathlib import Path
import cv2
import numpy as np
import torch
import time
import csv  # for CSV writing

# --- Ensure DPVO package root is on path ---
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dpvo import lietorch
from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.stream import image_stream, video_stream
from dpvo.utils import Timer

# ==============================
# 1️⃣ GLOBAL CONFIG
# ==============================
GLOBAL_IMAGE_PATH = Path("/home/kabs_d/global_20_lr.png")
RESULTS_FOLDER = Path("/home/kabs_d/DPVO/saved_trajectories")
OUTPUT_IMAGE_PATH = RESULTS_FOLDER / "live_projected_path.png"
CSV_OUTPUT_PATH = RESULTS_FOLDER / "vo_positions.csv"

# --- Camera intrinsics ---
fx, fy = 692.200439453125, 691.7953491210938
cx, cy = 644.2618408203125, 361.5069274902344
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.array([
    0.0036264462396502495,
    -0.04444902762770653,
    -0.000880336039699614,
    0.00022572405578102916,
    0.030637802556157112
], dtype=np.float32).reshape(-1, 1)

# --- +20° upward rotation (for projection) ---
angle_deg = -20.0
a = np.radians(angle_deg)
R_up = np.array([
    [1, 0, 0],
    [0, np.cos(a), -np.sin(a)],
    [0, np.sin(a), np.cos(a)]
], dtype=np.float32)

rvec = np.zeros((3, 1), dtype=np.float32)
tvec = np.zeros((3, 1), dtype=np.float32)

# ==============================
# 2️⃣ MAIN DPVO LIVE LOOP
# ==============================
@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False):
    slam = None
    queue = Queue(maxsize=8)

    # Load global image
    global_image = cv2.imread(str(GLOBAL_IMAGE_PATH))
    if global_image is None:
        raise FileNotFoundError(f"❌ Could not load {GLOBAL_IMAGE_PATH}")
    h, w = global_image.shape[:2]
    print(f"🖼️ Live projection on global image: {w}x{h}")

    # Reader setup
    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))
    reader.start()

    # Timing
    start_time = time.time()
    prev_time = start_time
    elapsed_time = 0.0
    frame_counter = 0

    poses_accum = []
    csv_data = []

    frame = None

    while True:
        (t, image, intrinsics) = queue.get()
        if t < 0:
            break

        image = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)

        # Extract pose
        if slam.n > 0:
            last_idx = int(slam.n - 1)
            try:
                se3 = lietorch.SE3(slam.pg.poses_[last_idx])
                cam_pose = se3.inv().data.cpu().numpy().ravel()
                tx, ty, tz = cam_pose[:3]

                poses_accum.append([float(t), float(tx), float(ty), float(tz), 0, 0, 0, 1])
                csv_data.append([elapsed_time, float(tx), float(tz)])
            except Exception as e:
                print(f"⚠️ pose extraction failed: {e}")
        else:
            print("⚠️ DPVO poses_ not yet initialized")

        # Not enough points yet
        if len(poses_accum) < 2:
            frame = global_image.copy()
            cv2.putText(frame, "Waiting for DPVO poses...", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Live Projection", frame)
            cv2.waitKey(1)
            prev_time = time.time()
            continue

        # ===========================
        # PROJECTION OF TRAJECTORY
        # ===========================
        trajectory_data = np.array(poses_accum, dtype=np.float32)
        object_points_pose = trajectory_data[:, 1:4]

        object_points_pose = (R_up @ object_points_pose.T).T
        y0 = object_points_pose[0, 1]
        object_points_pose[:, 1] = y0 + 0.75

        image_points, _ = cv2.projectPoints(object_points_pose, rvec, tvec, camera_matrix, dist_coeffs)

        # integer path points for the polyline (unchanged)
        int_pts = image_points.reshape(-1, 2).astype(np.int32)
        mask = (
            (int_pts[:, 0] >= 0) & (int_pts[:, 0] < w) &
            (int_pts[:, 1] >= 0) & (int_pts[:, 1] < h)
        )
        path_points = int_pts[mask]

        frame = global_image.copy()

        # Draw polyline
        if len(path_points) >= 2:
            cv2.polylines(frame, [path_points], False, (0, 255, 0), 2)

        # ===========================
        # 🚀 Replace blue dot with BLUE ARROW (Option A)
        # ===========================

        proj_pts = image_points.reshape(-1, 2)
        valid_mask = (
            (proj_pts[:, 0] >= 0) & (proj_pts[:, 0] < w) &
            (proj_pts[:, 1] >= 0) & (proj_pts[:, 1] < h)
        )
        valid_idx = np.nonzero(valid_mask)[0]

        if len(valid_idx) >= 2:
            i1, i2 = valid_idx[-2], valid_idx[-1]
            p1 = proj_pts[i1].astype(np.float64)
            p2 = proj_pts[i2].astype(np.float64)

            v = p2 - p1
            norm = np.linalg.norm(v)

            if norm < 0.005:
                cv2.circle(frame, (int(round(p2[0])), int(round(p2[1]))), 4, (255, 0, 0), -1)
            else:
                u = v / norm
                shaft_len = min(norm, 60.0)
                head_len = max(12.0, min(25.0, norm * 0.3))
                head_width = max(10.0, min(30.0, norm * 0.12))

                tail = p2 - u * shaft_len
                tip = p2

                perp = np.array([-u[1], u[0]])

                left = tip - u * head_len + perp * (head_width / 2.0)
                right = tip - u * head_len - perp * (head_width / 2.0)

                tail_i = (int(round(tail[0])), int(round(tail[1])))
                tip_i = (int(round(tip[0])), int(round(tip[1])))
                left_i = (int(round(left[0])), int(round(left[1])))
                right_i = (int(round(right[0])), int(round(right[1])))

                cv2.line(frame, tail_i, tip_i, (255, 0, 0), 4, lineType=cv2.LINE_AA)
                pts = np.array([left_i, tip_i, right_i], np.int32)
                cv2.fillConvexPoly(frame, pts, (255, 0, 0))

            # Red origin dot stays
            cv2.circle(frame, tuple(path_points[0]), 5, (0, 0, 255), -1)

        # FPS + display
        now = time.time()
        delta_t = now - prev_time
        elapsed_time += delta_t
        fps = 1.0 / delta_t if delta_t > 1e-6 else 0.0
        prev_time = now
        frame_counter += 1

        cv2.putText(frame, f"Frame {frame_counter} | FPS: {fps:.2f}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        cv2.putText(frame, f"Collected poses: {len(poses_accum)}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        cv2.imshow("Live Projection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.join()
    cv2.destroyAllWindows()

    print(f"✅ Finished live projection with {len(poses_accum)} poses.")

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    if frame is not None:
        cv2.imwrite(str(OUTPUT_IMAGE_PATH), frame)
        print(f"✅ Saved final projection to {OUTPUT_IMAGE_PATH}")

    with open(CSV_OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'tx', 'tz'])
        writer.writerows(csv_data)

    print(f"✅ Saved CSV to {CSV_OUTPUT_PATH}")


# ==============================
# 3️⃣ ENTRY POINT
# ==============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default="dpvo.pth")
    parser.add_argument("--imagedir", type=str)
    parser.add_argument("--calib", type=str)
    parser.add_argument("--name", type=str, default="result")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--timeit", action="store_true")
    parser.add_argument("--viz", action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.KEYFRAME_THRESH = 15.0
    cfg.OPTIMIZATION_WINDOW = 10
    cfg.PATCH_LIFETIME = 8
    cfg.REMOVAL_WINDOW = 17

    print("Running with config...\n", cfg)
    run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip, args.viz, args.timeit)
