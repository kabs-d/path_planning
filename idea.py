#!/usr/bin/env python3
"""
Interactive OpenCV click + PyTorch refinement of camera intrinsics & extrinsics.

Now includes plotting of loss (RMSE) vs iterations.
"""

import cv2
import numpy as np
import torch
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt   ### NEW ###

# ----------------- USER SETTINGS -----------------
IMAGE_PATH = "/home/kabs_d/mast3r/dust3r/croco/assets/floor2.jpg"

WORLD_POINTS = np.array([
    [0,   0,    0],
    [54,  0,    0],
    [54,  56.5, 0],
    [0,   56.5, 0],
    [0, -56.5,  0],
    [54, -56.8, 0]
], dtype=np.float64)

K0 = {
    "fx": 462.24111498,
    "fy": 468.12031394,
    "cx": 309.30907629,
    "cy": 274.08438126
}

OPTIMIZE_DISTORTION = False
LR = 1e-2
N_ITERS = 100000
PRINT_EVERY = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------------------------------

clicked_points = []
def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < WORLD_POINTS.shape[0]:
            clicked_points.append([int(x), int(y)])
            print(f"Clicked point {len(clicked_points)}: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if clicked_points:
            removed = clicked_points.pop()
            print(f"Removed last clicked point: {removed}")

def rodrigues_torch(rvec):
    theta = torch.norm(rvec)
    if theta.item() < 1e-12:
        return torch.eye(3, device=rvec.device, dtype=rvec.dtype)
    k = rvec / theta
    K = torch.tensor([[0., -k[2], k[1]],
                      [k[2], 0., -k[0]],
                      [-k[1], k[0], 0.]], device=rvec.device, dtype=rvec.dtype)
    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R

def project_points_torch(world_pts, rvec, tvec, fx, fy, cx, cy, k1=None, k2=None):
    R = rodrigues_torch(rvec)
    Xc = (R @ world_pts.T).T + tvec
    X = Xc[:, 0]; Y = Xc[:, 1]; Z = Xc[:, 2]
    x = X / Z
    y = Y / Z
    if k1 is not None:
        r2 = x*x + y*y
        radial = 1 + k1 * r2
        if k2 is not None:
            radial = radial + k2 * (r2*r2)
        x_d = x * radial
        y_d = y * radial
    else:
        x_d = x; y_d = y
    u = fx * x_d + cx
    v = fy * y_d + cy
    return torch.stack([u, v], dim=1)

def run_refinement(image_path, world_points, K0):
    global clicked_points
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"Image loaded: {image_path} (w={w}, h={h})")
    n_required = world_points.shape[0]

    if n_required < 4:
        raise ValueError("Provide at least 4 world points for stability.")

    win_name = "Click image points (left-click add, right-click undo, ESC to exit)"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback)

    print(f"Click {n_required} image points in the SAME ORDER as world points.")
    print("Right-click to undo. Press ESC to abort.")

    while True:
        disp = img.copy()
        for i, pt in enumerate(clicked_points):
            cv2.circle(disp, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(disp, str(i+1), (pt[0]+6, pt[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow(win_name, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            print("Aborted by user.")
            cv2.destroyAllWindows()
            sys.exit(0)
        if len(clicked_points) == n_required:
            cv2.waitKey(250)
            break

    cv2.destroyAllWindows()
    img_pts = np.array(clicked_points, dtype=np.float64).reshape(-1,1,2)
    world_pts_cv = world_points.reshape(-1,1,3).astype(np.float64)

    K_cv = np.array([[K0["fx"], 0, K0["cx"]],
                     [0, K0["fy"], K0["cy"]],
                     [0,0,1]], dtype=np.float64)

    success, rvec0, tvec0 = cv2.solvePnP(world_pts_cv, img_pts, K_cv, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise RuntimeError("solvePnP failed")
    rvec0 = rvec0.reshape(3).astype(np.float64)
    tvec0 = tvec0.reshape(3).astype(np.float64)

    img_reproj_cv, _ = cv2.projectPoints(world_pts_cv, rvec0.reshape(3,1), tvec0.reshape(3,1), K_cv, None)
    img_reproj_cv = img_reproj_cv.reshape(-1,2)
    init_rmse = np.sqrt(np.mean(np.sum((img_reproj_cv - img_pts.reshape(-1,2))**2, axis=1)))
    print(f"Initial reprojection RMSE (px): {init_rmse:.4f}")

    dtype = torch.float64
    world_t = torch.from_numpy(world_points).to(DEVICE).to(dtype)
    target_uv = torch.from_numpy(np.array(clicked_points, dtype=np.float64)).to(DEVICE).to(dtype)

    log_fx = torch.tensor(np.log(K0["fx"]), requires_grad=True, device=DEVICE, dtype=dtype)
    log_fy = torch.tensor(np.log(K0["fy"]), requires_grad=True, device=DEVICE, dtype=dtype)
    cx = torch.tensor(K0["cx"], requires_grad=True, device=DEVICE, dtype=dtype)
    cy = torch.tensor(K0["cy"], requires_grad=True, device=DEVICE, dtype=dtype)

    rvec_t = torch.tensor(rvec0, requires_grad=True, device=DEVICE, dtype=dtype)
    tvec_t = torch.tensor(tvec0, requires_grad=True, device=DEVICE, dtype=dtype)

    if OPTIMIZE_DISTORTION:
        k1 = torch.tensor(0.0, requires_grad=True, device=DEVICE, dtype=dtype)
        k2 = torch.tensor(0.0, requires_grad=True, device=DEVICE, dtype=dtype)
    else:
        k1 = None; k2 = None

    params = [log_fx, log_fy, cx, cy, rvec_t, tvec_t]
    if OPTIMIZE_DISTORTION:
        params += [k1, k2]

    optimizer = optim.Adam(params, lr=LR)

    best = None
    losses = []   ### NEW ###

    for it in range(1, N_ITERS+1):
        optimizer.zero_grad()
        fx_t = torch.exp(log_fx)
        fy_t = torch.exp(log_fy)

        proj = project_points_torch(world_t, rvec_t, tvec_t, fx_t, fy_t, cx, cy,
                                    k1=k1 if OPTIMIZE_DISTORTION else None,
                                    k2=k2 if OPTIMIZE_DISTORTION else None)
        err = proj - target_uv
        loss = torch.mean(torch.sum(err**2, dim=1))

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cx.clamp_(0.0, float(w))
            cy.clamp_(0.0, float(h))
            if OPTIMIZE_DISTORTION:
                k1.clamp_(-1.0, 1.0)
                k2.clamp_(-1.0, 1.0)

        cur_rmse = float(torch.sqrt(loss).item())
        losses.append(cur_rmse)   ### NEW ###

        if it % PRINT_EVERY == 0 or it==1 or it==N_ITERS:
            print(f"Iter {it:4d} | RMSE(px) = {cur_rmse:.6f}")
            if best is None or cur_rmse < best[0]:
                best = (cur_rmse,
                        float(torch.exp(log_fx).item()), float(torch.exp(log_fy).item()),
                        float(cx.item()), float(cy.item()),
                        rvec_t.detach().cpu().numpy().copy(), tvec_t.detach().cpu().numpy().copy(),
                        None if not OPTIMIZE_DISTORTION else (float(k1.item()), float(k2.item()))
                       )

    final_rmse, fx_fin, fy_fin, cx_fin, cy_fin, rvec_fin, tvec_fin, k_fin = best
    print("\nRefined camera parameters:")
    print(f"fx={fx_fin:.4f}, fy={fy_fin:.4f}, cx={cx_fin:.4f}, cy={cy_fin:.4f}")
    print("rvec:", rvec_fin.tolist())
    print("tvec:", tvec_fin.tolist())
    if k_fin:
        print("k1,k2:", k_fin)
    print(f"Final best RMSE (px): {final_rmse:.6f}")

    # --- Plot RMSE vs iterations ---   ### NEW ###
    plt.figure(figsize=(6,4))
    plt.plot(losses, label="RMSE (px)")
    plt.xlabel("Iteration")
    plt.ylabel("Reprojection Error (px)")
    plt.title("Loss vs Iterations")
    plt.grid(True)
    plt.legend()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()
    print("Saved loss curve to loss_curve.png")

    # --- Visualize reprojection before/after ---
    K_ref = np.array([[fx_fin, 0, cx_fin],[0, fy_fin, cy_fin],[0,0,1]], dtype=np.float64)
    rvec_ref = rvec_fin.reshape(3,1)
    tvec_ref = tvec_fin.reshape(3,1)

    after_reproj_cv, _ = cv2.projectPoints(world_pts_cv, rvec_ref, tvec_ref, K_ref, distCoeffs=None)
    after_reproj_cv = after_reproj_cv.reshape(-1,2)

    vis = img.copy()
    for p in np.array(clicked_points, dtype=int):
        cv2.circle(vis, (int(p[0]), int(p[1])), 6, (255,0,0), -1)
    for p in img_reproj_cv.astype(int):
        cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0,255,0), 2)
    for p in after_reproj_cv.astype(int):
        cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0,0,255), 2)

    out_name = "reprojection_refinement.png"
    cv2.imwrite(out_name, vis)
    print(f"Saved reprojection visualization to {out_name}")

if __name__ == "__main__":
    run_refinement(IMAGE_PATH, WORLD_POINTS, K0)
