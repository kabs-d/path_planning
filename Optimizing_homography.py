#!/usr/bin/env python3
"""
Homography refinement via direct optimization of matrix parameters using PyTorch.

- The 8 free parameters of the homography matrix are treated as learnable
  tensors and are directly refined by an Adam optimizer to minimize
  reprojection error.
- This version uses a hard-coded list of point correspondences to compuet RMSE Reprojection Error.
"""

import cv2
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# ----------------- USER SETTINGS -----------------
# Image Paths
IMG1_PATH = "/home/kabs_d/mast3r/floor2_with_fitted_grid_and_extra_columns_v4.jpg"
IMG2_PATH = "/home/kabs_d/mast3r/sec.jpg"

# Starting homography (user-provided initial guess)
H0 = np.array([[ 4.82022100e-01 ,-1.80439926e+00 , 8.99994220e+02],
 [-8.72638433e-03, 6.42153559e-01, -3.25468825e+02],
 [-2.34114875e-05 ,-1.30960024e-03 , 1.00000000e+00]], dtype=np.float64)

# Optimization Settings
LR = 1e-5
N_ITERS = 100000
PRINT_EVERY = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------------

# --- Direct Optimization Function ---
def optimize_homography_directly(pts1, pts2, H0, n_iters=N_ITERS, lr=LR, print_every=PRINT_EVERY):
    """
    Refines a homography matrix H by directly optimizing its parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    pts1_t = torch.tensor(pts1, device=device, dtype=dtype)
    pts2_t = torch.tensor(pts2, device=device, dtype=dtype)
    ones = torch.ones((pts1_t.shape[0], 1), device=device, dtype=dtype)
    pts1_h = torch.cat([pts1_t, ones], dim=1)

    H_t = torch.tensor(H0, device=device, dtype=dtype, requires_grad=True)
    optimizer = optim.Adam([H_t], lr=lr)

    min_rmse = float('inf')
    best_H = H0.copy()
    losses = []

    print("\n--- Training: Direct Homography Optimization ---")
    for it in range(1, n_iters + 1):
        optimizer.zero_grad()

        proj_h = pts1_h @ H_t.T
        proj = proj_h[:, :2] / proj_h[:, 2:3]

        err = proj - pts2_t
        loss = torch.mean(torch.sum(err ** 2, dim=1))

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            scalar = H_t.data[2, 2].item()
            if abs(scalar) > 1e-9: # Avoid division by zero
                H_t.data /= scalar

        cur_rmse = float(torch.sqrt(loss).item())
        losses.append(cur_rmse)

        if cur_rmse < min_rmse:
            min_rmse = cur_rmse
            best_H = H_t.detach().cpu().numpy().copy()

        if it % print_every == 0 or it == 1 or it == n_iters:
            print(f"Iter {it:5d} | RMSE(px): {cur_rmse:.6f} | Min RMSE: {min_rmse:.6f}")

    print(f"\nOptimization complete. Best RMSE(px): {min_rmse:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("RMSE vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Root Mean Square Error (pixels)")
    plt.grid(True)
    plt.savefig("loss_curve_direct_h.png")
    print("✅ Loss curve saved to 'loss_curve_direct_h.png'")

    return best_H


if __name__ == "__main__":
    img1 = cv2.imread(IMG1_PATH)
    img2 = cv2.imread(IMG2_PATH)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load one of the images")

    # --- MODIFICATION: Use hard-coded points instead of clicking ---
    pts1_refine = np.array([
        [843, 679], [795, 657], [756, 638], [725, 624], [702, 613],
        [685, 603], [666, 596], [653, 589], [641, 583], [698, 678],
        [667, 658], [645, 640], [626, 625], [612, 614], [601, 605],
        [591, 596], [583, 589], [527, 638], [522, 626], [517, 612],
        [515, 604], [511, 595], [507, 589], [429, 613], [433, 603],
        [437, 596], [439, 588], [441, 583], [359, 596], [368, 589],
        [376, 583], [381, 579], [386, 574], [308, 584], [316, 578],
        [326, 574], [331, 570], [338, 567]
    ], dtype=np.float64)

    pts2_refine = np.array([
        [882, 1153], [800, 741], [765, 537], [747, 422], [730, 345],
        [717, 260], [708, 217], [704, 193], [700, 175], [296, 728],
        [385, 534], [442, 418], [484, 343], [506, 295], [526, 252],
        [546, 223], [555, 193], [157, 416], [237, 343], [298, 293],
        [344, 255], [378, 220], [407, 199], [112, 287], [175, 258],
        [226, 224], [269, 197], [306, 174], [73, 222], [121, 203],
        [169, 179], [222, 157], [245, 146], [40, 184], [92, 163],
        [129, 147], [174, 131], [211, 119]
    ], dtype=np.float64)
    # --- END MODIFICATION ---

    print(f"\n✅ Loaded {len(pts1_refine)} corresponding points.")
    print("\nUsing the user-provided H0 as the starting point...")
    print("Initial Homography (H0):\n", H0)

    H_refined = optimize_homography_directly(pts1_refine, pts2_refine, H0)

    np.save("refined_H_direct.npy", H_refined)
    print("\nRefined homography saved as refined_H_direct.npy")
    print("Final Matrix:\n", H_refined)

    vis = img2.copy()
    pts1_h_np = np.hstack([pts1_refine, np.ones((pts1_refine.shape[0], 1))])
    proj_pts_h = (H_refined @ pts1_h_np.T).T
    proj_pts = proj_pts_h[:, :2] / proj_pts_h[:, 2:3]

    for p_actual, p_proj in zip(pts2_refine.astype(int), proj_pts.astype(int)):
        cv2.circle(vis, tuple(p_actual), 8, (0, 255, 0), -1)
        cv2.circle(vis, tuple(p_proj), 8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(vis, tuple(p_actual), tuple(p_proj), (0, 255, 255), 2)

    cv2.imwrite("homography_visualization_direct.jpg", vis)
    print("✅ Successfully saved visualization: homography_visualization_direct.jpg")

    warped = cv2.warpPerspective(img1, H_refined, (img2.shape[1], img2.shape[0]))
    cv2.imwrite("warped_image_direct.jpg", warped)
    print("✅ Successfully saved warped image: warped_image_direct.jpg")

    cv2.imshow("Refinement Error Visualization", vis)
    cv2.imshow("Final Warped Image", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
