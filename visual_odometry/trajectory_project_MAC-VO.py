import numpy as np
import cv2
import sys

# --- 1. CONFIGURATION (Paths) ---
GLOBAL_IMAGE_PATH = "/data/global_20_lr_gr.png" 
POSES_FILE = "Results/MACVO-Fast@rishita_sequence/11_12_173439/poses.npy"
RESULTS_FOLDER = "Results/MACVO-Fast@rishita_sequence/11_12_173439/"
OUTPUT_IMAGE_PATH = RESULTS_FOLDER + "projected_trajectory_NEW_LOGIC.png" # New output name

# --- 2. PROJECTION LOGIC (Camera Parameters) ---

# Using your specified MAC-VO intrinsics
fx = 409.2878112792969
fy = 409.2878112792969
cx = 424.6625061035156
cy = 242.31875610351562
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0,  1]
], dtype=np.float32)

# Using your specified zero distortion
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# -20 degree rotation around X-axis (pitch up)
angle_deg = -20.0 
a = np.radians(angle_deg)
R_up = np.array([
    [1, 0, 0],
    [0, np.cos(a), -np.sin(a)],
    [0, np.sin(a), np.cos(a)]
], dtype=np.float32)

# Camera is at (0,0,0) with no rotation
rvec = np.zeros((3, 1), dtype=np.float32)
tvec = np.zeros((3, 1), dtype=np.float32)

# --- 3. LOAD AND PROCESS POSES (NEW LOGIC) ---
print(f"Loading poses from {POSES_FILE}...")
try:
    # poses.npy format: [timestamp, tx, ty, tz, qx, qy, qz, qw]
    trajectory_data = np.load(POSES_FILE)
except FileNotFoundError:
    print(f"Error: Poses file not found at {POSES_FILE}")
    sys.exit(1)

print("Remapping MAC-VO poses to OpenCV coordinates (NEW LOGIC)...")
# Per your new instructions:
# OpenCV X (right) = 3rd column (index 2) -> ty
# OpenCV Z (fwd)   = 2nd column (index 1) -> tx
# OpenCV Y (down)  = 4th column (index 3) * -1 -> -tz
cv_x_coords = trajectory_data[:, 2] 
cv_z_coords = trajectory_data[:, 1] 
cv_y_coords = trajectory_data[:, 3] * -1.0 # <-- NEW LOGIC HERE

# Create the (N, 3) array of 3D points
# Shape: [X, Y, Z]
object_points_pose = np.stack([
    cv_x_coords, 
    cv_y_coords, # Use the new -tz coordinate
    cv_z_coords
], axis=1).astype(np.float32)

# --- 4. APPLY ROTATION, THEN SET Y (NEW LOGIC) ---

# 1. Apply the 20-degree rotation to ALL 3D points
print("Applying 20-degree rotation logic...")
rotated_points = (R_up @ object_points_pose.T).T

# 2. Get the Y-value of the FIRST frame *after* rotation
y0_rotated = rotated_points[0, 1]
print(f"Y-value of first rotated frame (y0) is: {y0_rotated}")

# 3. Set the Y-coordinate of ALL points to y0 + 0.75
print("Setting all Y-coordinates to y0 + 0.75...")
rotated_points[:, 1] = y0_rotated + 0.75

# The final points to project are the rotated and offset points
object_points_to_project = rotated_points

# --- 5. LOAD IMAGE AND PROJECT ---
print(f"Loading global image from {GLOBAL_IMAGE_PATH}...")
global_image = cv2.imread(GLOBAL_IMAGE_PATH)
if global_image is None:
    print(f"Error: Could not load image from {GLOBAL_IMAGE_PATH}")
    sys.exit(1)

# Check if image dimensions match your info
h, w = global_image.shape[:2]
if not (w == 848 and h == 480):
    print(f"Warning: Global image size is {w}x{h}, not 848x480 as specified.")

print(f"Projecting {len(object_points_to_project)} points onto {w}x{h} image...")

# Project all 3D points onto the 2D image
image_points, _ = cv2.projectPoints(object_points_to_project,
                                    rvec,
                                    tvec,
                                    camera_matrix,
                                    dist_coeffs)

# --- 6. DRAW AND SAVE ---
print("Drawing the path on the image...")
path_points = image_points.reshape(-1, 2).astype(np.int32)

# Filter out points that are outside the image boundaries
mask = (
    (path_points[:, 0] >= 0) & (path_points[:, 0] < w) &
    (path_points[:, 1] >= 0) & (path_points[:, 1] < h)
)
path_points = path_points[mask]

# Draw the path as a series of connected green lines
cv2.polylines(global_image,
              [path_points],
              isClosed=False,
              color=(0, 255, 0), # BGR Color (Green)
              thickness=1
)

# Draw start (red) and end (blue) points
if len(path_points) > 0:
    cv2.circle(global_image, tuple(path_points[0]), 5, (0, 0, 255), -1) # Red start
    cv2.circle(global_image, tuple(path_points[-1]), 5, (255, 0, 0), -1) # Blue end

# Save the final image
cv2.imwrite(OUTPUT_IMAGE_PATH, global_image)
print(f"\n✅ Success! Projected trajectory saved to:\n{OUTPUT_IMAGE_PATH}")
