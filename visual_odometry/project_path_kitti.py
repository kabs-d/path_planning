import cv2
import numpy as np
import glob
import os
import sys
import pandas as pd
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION FOR FRAME 0 ---
TARGET_FRAME_INDEX = 0 
IMAGE_FILENAME = "000000.png"

# --- 1. FIND THE NEW TRAJECTORY FILE (Source of Estimated Data) ---
RESULTS_DIR = "/home/kabs_d/MAC-VO/Results/"
TRAJ_PATH = None

try:
    # Find the newest folder containing 'K00_Corrected'
    list_of_folders = [f for f in glob.glob(os.path.join(RESULTS_DIR, '*/')) if 'K00_Corrected' in f]
    if not list_of_folders:
        raise FileNotFoundError("No 'K00_Corrected' run folder found in Results/")
    
    latest_folder = max(list_of_folders, key=os.path.getctime)
    
    # Find poses.npy inside the latest run's subdirectories
    npy_files = glob.glob(os.path.join(latest_folder, '**/poses.npy'), recursive=True)
    if not npy_files:
        raise FileNotFoundError(f"Can't find poses.npy in {latest_folder}")
    
    TRAJ_PATH = npy_files[0]
    print(f"Found trajectory file: {TRAJ_PATH}")

except Exception as e:
    print(f"Error finding trajectory file: {e}")
    sys.exit(1)

# --- 2. FILE PATHS ---
CALIB_PATH = "/home/kabs_d/MAC-VO/00/calib.txt"
IMAGE_PATH = "/home/kabs_d/MAC-VO/000000.png"
OUTPUT_PATH = "/home/kabs_d/MAC-VO/trajectory_on_image_DIRECT_PROJECTION_EST1.png" # Unique output name


# --- UTILITY FUNCTIONS ---
def get_projection_matrix(path):
    """
    Reads calib.txt and gets the P0 (left grayscale) projection matrix.
    """
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('P0:'):
                tokens = line.split(' ')
                values = [v for v in tokens if v.strip() and v != 'P0:'] 
                if len(values) != 12: raise ValueError(f"P0 line contains {len(values)} values, expected 12.")
                proj_matrix = np.array([float(v) for v in values]).reshape(3, 4)
                return proj_matrix
    raise ValueError("Could not find 'P0:' in calibration file.")


def main():
    # --- LOAD RAW ESTIMATED DATA ---
    try:
        # Load the raw N, 8 estimated pose array
        poses_8dof = np.load(TRAJ_PATH)
    except Exception as e:
        print(f"Error loading pose data from {TRAJ_PATH}: {e}")
        return

    # --- GET MATRICES & IMAGE ---
    P_rect_cam = get_projection_matrix(CALIB_PATH)
    
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        return

    img_h, img_w = image.shape[:2]
    points_2d = []
    NEAR_CLIP_PLANE = 0.5 
    TY_OFFSET = 1.65 # The required offset to subtract

    # --- FLAWED DIRECT PROJECTION LOOP ---
    # The loop iterates over the raw 8-DOF pose vectors
    for pose_8dof in poses_8dof:
        
        # Extract estimated Tx, Ty, Tz from the raw vector (indices 1, 2, 3)
        # Apply the -1.65m offset to the Ty component (World East axis)
        Tx = pose_8dof[1]
        Ty_adjusted = pose_8dof[2] + TY_OFFSET
        Tz = pose_8dof[3]
        
        # P_World_Column: The [Tx, Ty, Tz, 1] vector used as the direct input.
        P_World_Column = np.array([Tx, Ty_adjusted, Tz, 1.0])
        
        # ⚠️ FLAWED STEP: Project P_World_Column directly using P_rect_cam
        # This skips the necessary T_Cam0_World inverse extrinsic multiplication.
        y_image = P_rect_cam @ P_World_Column
        w = y_image[2] # Depth

        # Apply the near clipping plane check and perspective division
        if w > NEAR_CLIP_PLANE:
            
            # Calculate and round pixel coordinates
            u = y_image[0] / w
            v = y_image[1] / w
            
            pixel_x = np.round(u).astype(np.int32)
            pixel_y = np.round(v).astype(np.int32)
            
            if 0 <= pixel_x < img_w and 0 <= pixel_y < img_h:
                points_2d.append((pixel_x, pixel_y))

    # --- DRAW AND SAVE ---
    if points_2d:
        path_points = np.array(points_2d, dtype=np.int32).reshape((-1, 1, 2))
        
        # Draw a thick blue line (BGR: 255, 0, 0)
        cv2.polylines(image, [path_points], isClosed=False, color=(0, 255, 255), thickness=1) 
        
        cv2.imwrite(OUTPUT_PATH, image)
        print(f"Success! Projected path using estimated poses and Ty offset saved to {OUTPUT_PATH}")
    else:
        print("Warning: Trajectory points were entirely rejected. This confirms the major geometric distortion is present.")

if __name__ == "__main__":
    main()
