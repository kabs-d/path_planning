#!/usr/bin/env python3
import cv2
import numpy as np

# --- Camera intrinsics ---
K = np.array([
    [875.4554, 0, 476.4002],
    [0, 522.7173, 265.4197],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros((5,1), dtype=np.float64)

# --- Image ---
image_path = "/home/kabs_d/mast3r/dust3r/croco/assets/skewed_floor.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(image_path)

height, width = image.shape[:2]

# --- World points ---
world_points_to_mark = np.array([
    [0.0, 0.0, 0.0], [54.0, 0.0, 0.0],
    [0.0, 56.5, 0.0], [54.0, 56.5, 0.0],
    [0.0, -56.5, 0.0], [54.0, -56.5, 0.0],
    [0.0, -113.0, 0.0], [54.0, -113.0, 0.0],
    [0.0, -169.5, 0.0], [54.0, -169.5, 0.0],
    [0.0, -226.0, 0.0], [54.0, -226.0, 0.0],
    [0.0, -282.5, 0.0], [54.0, -282.5, 0.0],
    [0.0, -339.0, 0.0], [54.0, -339.0, 0.0],
    [0.0, -395.5, 0.0], [54.0, -395.5, 0.0],
    [0.0, -452.0, 0.0], [54.0, -452.0, 0.0],
    [0.0, -508.5, 0.0], [54.0, -508.5, 0.0],
    [0.0, -565.0, 0.0], [54.0, -565.0, 0.0],
    [0.0, -621.5, 0.0], [54.0, -621.5, 0.0],
    [0.0, -678.0, 0.0], [54.0, -678.0, 0.0],
    [0.0, -734.5, 0.0], [54.0, -734.5, 0.0],
    [0.0, -791.0, 0.0], [54.0, -791.0, 0.0],
    [0.0, -847.5, 0.0], [54.0, -847.5, 0.0],
    [0.0, -904.0, 0.0], [54.0, -904.0, 0.0],
    [0.0, -960.5, 0.0], [54.0, -960.5, 0.0],
    [0.0, -1017.0, 0.0], [54.0, -1017, 0.0],
    [0.0, -1073.5, 0.0], [54.0, -1073.5, 0.0],
    [0.0, 113.5, 0.0], [54.0,113.5,0.0]
], dtype=np.float64)

# --- Define 8 points for PnP ---
world_points_for_pnp = world_points_to_mark[:8]

# --- Click routine with undo ---
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 8:
        clicked_points.append([x, y])
        print(f"Clicked point {len(clicked_points)}: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN and clicked_points:
        removed = clicked_points.pop()
        print(f"Removed last point: ({removed[0]}, {removed[1]})")

cv2.namedWindow("Click 8 points")
cv2.setMouseCallback("Click 8 points", mouse_callback)
print("Click 8 points in order: origin, X, Y, diagonal, upward steps. ESC when done. Right-click to undo.")

while len(clicked_points) < 8:
    disp = image.copy()
    for i, pt in enumerate(clicked_points):
        cv2.circle(disp, tuple(pt), 5, (0,0,255), -1)
        cv2.putText(disp, str(i+1), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.imshow("Click 8 points", disp)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()

if len(clicked_points) != 8:
    raise ValueError("Exactly 8 points needed for solvePnP")

image_points = np.array(clicked_points, dtype=np.float64)

# --- SolvePnP ---
success, rvec, tvec = cv2.solvePnP(
    world_points_for_pnp,
    image_points,
    K,
    dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)
if not success:
    raise RuntimeError("solvePnP failed")

print("Rotation vector (rvec):\n", rvec)
print("Translation vector (tvec):\n", tvec)

out = image.copy()

# --- Columns to use ---
all_columns = [0.0, 54.0, 112.0, -57.0, 168.0, -114.0]
column_colors = {
    0.0: (0,0,255), 54.0: (0,0,255),
    112.0: (0,165,255), -57.0: (255,165,0),
    168.0: (0,255,255), -114.0: (255,0,255)
}

# --- Project all points for all columns ---
proj_points_all_columns = {}
unique_ys = np.unique(world_points_to_mark[:, 1])
for x_val in all_columns:
    points = np.array([[x_val, y, 0.0] for y in unique_ys], dtype=np.float64)
    proj_pts, _ = cv2.projectPoints(points, rvec, tvec, K, dist_coeffs)
    proj_pts = proj_pts.reshape(-1,2)
    proj_points_all_columns[x_val] = proj_pts

    # Draw extra points
    for p in proj_pts:
        px, py = int(round(p[0])), int(round(p[1]))
        cv2.circle(out, (px, py), 3, column_colors[x_val], -1)
    
    # Draw vertical line using linear regression
    if len(proj_pts) >= 2:
        x_coords = proj_pts[:, 0]
        y_coords = proj_pts[:, 1]
        m, b = np.polyfit(x_coords, y_coords, 1)
        y_top = int(round(m*0 + b))
        y_bottom = int(round(m*(width-1) + b))
        cv2.line(out, (0, y_top), (width-1, y_bottom), column_colors[x_val], 1)

# --- Draw horizontal lines across all columns ---
for y_val in unique_ys:
    points_for_line = []
    for x_val in all_columns:
        proj_pts = proj_points_all_columns[x_val]
        idx = np.where(unique_ys == y_val)[0][0]
        points_for_line.append(proj_pts[idx])
    points_for_line = np.array(points_for_line)
    x_coords = points_for_line[:, 0]
    y_coords = points_for_line[:, 1]
    m, b = np.polyfit(x_coords, y_coords, 1)
    y_top = int(round(m*0 + b))
    y_bottom = int(round(m*(width-1) + b))
    cv2.line(out, (0, y_top), (width-1, y_bottom), (255,0,0), 1)

# --- Save and display ---
out_path = "/home/kabs_d/mast3r/floor2_with_fitted_grid_and_extra_columns_v5.jpg"
cv2.imwrite(out_path, out)
print(f"Saved marked image: {out_path}")

cv2.imshow("Result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
