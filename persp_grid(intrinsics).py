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

image_path = "/home/kabs_d/mast3r/dust3r/croco/assets/floor2.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(image_path)

height, width = image.shape[:2]

# --- World points (added 5 more points at the end) ---
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
], dtype=np.float64)

# --- Define 8 points for PnP ---
world_points_for_pnp = world_points_to_mark[:8]

# --- Click routine ---
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 8:
        clicked_points.append([x, y])
        print(f"Clicked point {len(clicked_points)}: ({x}, {y})")

cv2.namedWindow("Click 8 points")
cv2.setMouseCallback("Click 8 points", mouse_callback)
print("Click 8 points in order: origin, X, Y, diagonal, upward steps. ESC when done.")

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

# --- Project all world points ---
proj_pts, _ = cv2.projectPoints(world_points_to_mark, rvec, tvec, K, dist_coeffs)
proj_pts = proj_pts.reshape(-1,2)

out = image.copy()

# --- Draw projected points and print coordinates ---
for i, p in enumerate(proj_pts):
    px, py = int(round(p[0])), int(round(p[1]))
    cv2.circle(out, (px, py), 2, (0,255,0), -1)
    print(f"Projected point {i}: ({px}, {py})")

# --- Draw horizontal lines ---
for i in range(0, len(proj_pts)-1, 2):
    y1 = int(round(proj_pts[i][1]))
    y2 = int(round(proj_pts[i+1][1]))
    y_avg = (y1 + y2) // 2
    cv2.line(out, (0, y_avg), (width-1, y_avg), (255,0,0), 1)

# --- Vertical lines for main columns (0 and 54) ---
main_columns = [0.0, 54.0]
for x_val in main_columns:
    pts_column = np.array([proj_pts[i] for i, wp in enumerate(world_points_to_mark) if wp[0] == x_val])
    if len(pts_column) < 2:
        continue
    x_coords = pts_column[:, 0]
    y_coords = pts_column[:, 1]
    m, b = np.polyfit(x_coords, y_coords, 1)
    y_top = int(round(m*0 + b))
    y_bottom = int(round(m*(width-1) + b))
    cv2.line(out, (0, y_top), (width-1, y_bottom), (0,0,255), 1)

# --- Extra columns and points ---
unique_ys = np.unique(world_points_to_mark[:, 1])
extra_columns = {
    111.0: (0,165,255),
    -57.0: (255,165,0),
    168.0: (0,255,255),
    -114.0: (255,0,255)
}

for x_val, color in extra_columns.items():
    extra_points = np.array([[x_val, y, 0.0] for y in unique_ys], dtype=np.float64)
    proj_extra, _ = cv2.projectPoints(extra_points, rvec, tvec, K, dist_coeffs)
    proj_extra = proj_extra.reshape(-1,2)

    # Draw extra points
    for i, p in enumerate(proj_extra):
        px, py = int(round(p[0])), int(round(p[1]))
        cv2.circle(out, (px, py), 3, color, -1)
        print(f"Extra x={x_val} projected point {i}: ({px}, {py})")

    # Draw vertical line using linear regression
    if len(proj_extra) >= 2:
        x_coords = proj_extra[:, 0]
        y_coords = proj_extra[:, 1]
        m, b = np.polyfit(x_coords, y_coords, 1)
        y_top = int(round(m*0 + b))
        y_bottom = int(round(m*(width-1) + b))
        cv2.line(out, (0, y_top), (width-1, y_bottom), color, 1)

# --- Save and display ---
out_path = "/home/kabs_d/mast3r/floor2_with_fitted_grid_and_extra_columns_v7.jpg"
cv2.imwrite(out_path, out)
print(f"Saved marked image: {out_path}")

cv2.imshow("Result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
