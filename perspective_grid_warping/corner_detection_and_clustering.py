import cv2
import numpy as np
import math

from sklearn.cluster import KMeans


# -----------------------
# Pipeline 2: Shi–Tomasi
# -----------------------
def detect_corners_shitomasi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Initial loose detection (dense everywhere)
    corners = cv2.goodFeaturesToTrack(
        gray, 
        maxCorners=2000,        # detect a lot, we will filter
        qualityLevel=0.1, 
        minDistance=15          # allow dense before filtering
    )
    if corners is None:
        return []
    corners = corners.reshape(-1, 2)

    # Perspective-aware filtering
    kept = []
    for (x, y) in corners:
        # Normalize Y position (0=top, 1=bottom)
        y_norm = y / h  

        # More aggressive scaling: cubic growth + higher base
        min_dist = int(8 + 1000 * (y_norm ** 4))  # near base ≈ 88 px

        # Keep if far enough from already kept points
        if all(np.hypot(x - px, y - py) >= min_dist for (px, py) in kept):
            kept.append((x, y))

    return [(int(x), int(y)) for (x, y) in kept]


from sklearn.neighbors import NearestNeighbors

def filter_outlier_corners(corners, k=5, distance_thresh=2.5):
    """
    corners: np.array of shape (N,2)
    k: number of nearest neighbors to consider
    distance_thresh: multiplier for outlier detection
    """
    if len(corners) < k + 1:
        return corners  # too few points, skip

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(corners)
    distances, _ = nbrs.kneighbors(corners)  # distances shape: (N, k+1)
    # ignore distance to self (first column)
    mean_dist = distances[:, 1:].mean(axis=1)
    median_dist = np.median(mean_dist)
    
    # Keep points within distance_thresh * median
    filtered = corners[mean_dist < distance_thresh * median_dist]
    return filtered

def draw_corners(img, corners, color=(0,0,255), radius=4):
    """
    Draw corners on the image.
    corners: np.array of shape (N,2)
    """
    out = img.copy()
    for (x, y) in corners.astype(int):
        cv2.circle(out, (x, y), radius, color, -1)
    return out
import numpy as np
from sklearn.cluster import DBSCAN

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

def estimate_vp_slope(corners, min_segments=5, slope_eps=0.1):
    pts = corners.copy()
    segments = []

    # Create segments to nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=2).fit(pts)
    _, idxs = nbrs.kneighbors(pts)
    
    for i, neighbors in enumerate(idxs):
        p1 = pts[i]
        for j in neighbors[1:]:
            p2 = pts[j]
            dy = p2[1] - p1[1]
            dx = p2[0] - p1[0]
            if abs(dy) < 1e-3: continue
            slope = dx/dy
            segments.append((p1, p2, slope))
    
    if not segments: # Added check in case no segments are created
        print("Warning: No segments found to estimate vanishing point.")
        # Return a default VP at the top center of the image
        # This requires image dimensions, which we don't have here.
        # A high-level generic default can be an array of zeros or a specific point.
        # For now, let's return a point far away. A better approach would be passing image width.
        return np.array([500, -1000]) # A generic distant point

    slopes = np.array([s for _,_,s in segments]).reshape(-1,1)
    clustering = DBSCAN(eps=slope_eps, min_samples=min_segments).fit(slopes)
    labels = clustering.labels_
    
    # --- MODIFIED SECTION START ---
    valid_clusters = set(labels) - {-1}
    if not valid_clusters:
        print("Warning: DBSCAN found no valid clusters for the vanishing point.")
        # Return a default VP (e.g., top center, though we don't have image width)
        # We can calculate an approximate center from the corners themselves
        x_mean = np.mean(corners[:, 0])
        return np.array([x_mean, 0]) # Default to top-center of the corner cluster
    # --- MODIFIED SECTION END ---
    
    # Select largest cluster
    best_cluster = max(valid_clusters, key=lambda l: np.sum(labels==l))
    cluster_segments = [seg for seg, l in zip(segments, labels) if l == best_cluster]
    
    # Solve intersection of cluster segments
    A, B = [], []
    for p1, p2, _ in cluster_segments:
        a = p2[1]-p1[1]; b = p1[0]-p2[0]; c = a*p1[0]+b*p1[1]
        if a!=0 or b!=0:
            A.append([a,b])
            B.append([c])
    
    if not A: # Added another check in case cluster_segments is empty
        print("Warning: No valid lines in the best cluster to estimate vanishing point.")
        x_mean = np.mean(corners[:, 0])
        return np.array([x_mean, 0])

    A = np.array(A); B = np.array(B)
    vp, _, _, _ = np.linalg.lstsq(A,B,rcond=None)
    return vp.flatten()


def draw_perspective_grid_slope(img, corners, n_rows=11, color=(0,255,0), thickness=2):
    """
    Draw perspective grid using slope-based vanishing point estimation.
    Horizontal lines: clustered by y-coordinate (KMeans)
    Vertical lines: converge toward vanishing point estimated by slope clustering
    """
    img_out = img.copy()
    h, w = img.shape[:2]

    # --- ADDED THIS CHECK ---
    # Ensure we don't ask for more clusters than we have points
    if len(corners) < n_rows:
        print(f"Warning: Only found {len(corners)} corners, reducing n_rows from {n_rows} to {len(corners)}")
        n_rows = len(corners)
    
    # If there are too few points to form lines, return the image
    if n_rows < 2:
        print("Warning: Too few corners to draw a grid. Skipping.")
        return img_out
    # --- END OF CHECK ---

    # ---- Horizontal lines ----
    y_coords = corners[:, 1].reshape(-1, 1)
    # The 'n_init' argument is added to avoid a future warning in scikit-learn
    kmeans_rows = KMeans(n_clusters=n_rows, random_state=0, n_init=10).fit(y_coords)
    row_centers = kmeans_rows.cluster_centers_.flatten()
    sorted_row_indices = np.argsort(row_centers)

    for r_idx in sorted_row_indices:
        pts = corners[kmeans_rows.labels_ == r_idx]
        if len(pts) < 2:
            continue
        vx, vy, x0, y0 = cv2.fitLine(
            pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        )
        vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]
        if abs(vx) < 1e-6: vx = 1e-6  # avoid divide-by-zero
        left_y = int((-x0*vy/vx) + y0)
        right_y = int(((w-1 - x0)*vy/vx) + y0)
        cv2.line(img_out, (0, left_y), (w-1, right_y), color, thickness)

    # ---- Vertical lines (perspective-aware using slope-based VP) ----
    vp = estimate_vp_slope(corners)
    # The rest of the function remains the same...
    # (The following code is identical to your original script)
    vertical_pts = corners.copy()
    directions = vertical_pts - vp
    angles = np.arctan2(directions[:,1], directions[:,0])
    angle_thresh = np.deg2rad(5)
    used = np.zeros(len(vertical_pts), dtype=bool)

    clusters = []
    for i, angle_i in enumerate(angles):
        if used[i]:
            continue
        cluster_idx = [i]
        for j in range(i+1, len(angles)):
            if abs(angle_i - angles[j]) < angle_thresh:
                cluster_idx.append(j)
        pts_cluster = vertical_pts[cluster_idx]
        used[cluster_idx] = True
        clusters.append(pts_cluster)

    # Draw lines for each cluster toward VP
    for pts_cluster in clusters:
        mean_pt = pts_cluster.mean(axis=0)
        dir_vec = vp - mean_pt
        norm_dir = np.linalg.norm(dir_vec)
        if norm_dir < 1e-3:
            continue
        dir_vec /= norm_dir
        t = max(h, w) * 2
        p_start = mean_pt - t * dir_vec
        p_end = mean_pt + t * dir_vec
        cv2.line(
            img_out,
            tuple(np.round(p_start).astype(int)),
            tuple(np.round(p_end).astype(int)),
            color,
            thickness
        )

    return img_out



if __name__ == "__main__":
    image_path = "/home/kabs_d/mast3r/chess.png"
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # -----------------------
    # Step 0: Detect Shi–Tomasi corners
    # -----------------------
    corners2 = detect_corners_shitomasi(img)
    

    shi_pts = np.array(corners2, dtype=np.float32)
    
    shi_pts_filtered = filter_outlier_corners(shi_pts)
    img_corners = draw_corners(img, shi_pts_filtered, color=(0,0,255))
# Draw perspective grid using filtered points
    grid_img = draw_perspective_grid_slope(img_corners, shi_pts_filtered)
    cv2.imwrite("perspective_grid_filtered1.jpg", grid_img)
  
