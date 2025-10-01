#Camera Calibration script by CHeckerbpard Corner Detection
import numpy as np
import cv2 as cv
import time

#####################################################################################
# ----------------------------- PARAMETERS ---------------------------------------- #
#####################################################################################

# Define the size of the chessboard pattern (inner corners)
# A standard board with 10x7 squares has (9, 6) inner corners.
CHESSBOARD_SIZE = (9, 6)

# The number of frames to capture for calibration
NUM_CALIBRATION_IMAGES = 5

# Physical size of one square in your chosen units (e.g., millimeters)
# This is only important if you need to perform real-world measurements.
SQUARE_SIZE_MM = 25

#####################################################################################
# ------------------------- SCRIPT LOGIC ------------------------------------------ #
#####################################################################################

def live_camera_calibration():
    """
    Performs a live camera calibration using a chessboard pattern.
    Returns the camera matrix and distortion coefficients.
    """
    # Termination criteria for corner refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    # Arrays to store points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Initialize video capture
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return None, None
        
    print("Starting live calibration...")
    print(f"Point the camera at a {CHESSBOARD_SIZE} checkerboard.")
    print(f"Press 'c' to capture a frame when corners are detected.")
    print("Press 'q' to quit.")

    frames_captured = 0
    
    while frames_captured < NUM_CALIBRATION_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret_corners, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        # Display instructions and status on the frame
        status_text = f"Captured: {frames_captured}/{NUM_CALIBRATION_IMAGES}"
        cv.putText(frame, status_text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        instruction_text = "Press 'c' to capture, 'q' to quit"
        cv.putText(frame, instruction_text, (50, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if ret_corners:
            # Draw the detected corners on the frame for visualization
            cv.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret_corners)
            
            # Check for user input
            key = cv.waitKey(1) & 0xFF
            if key == ord('c'):
                print(f"Capturing frame {frames_captured + 1}...")
                
                # Refine corner locations
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Add the object points and refined image points to our lists
                objpoints.append(objp)
                imgpoints.append(corners2)
                
                frames_captured += 1
                
                # Provide visual feedback that capture was successful
                cv.putText(frame, "Captured!", (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                           cv.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 255), 3)
                cv.imshow('Live Calibration', frame)
                cv.waitKey(500) # Pause for half a second

        else:
            key = cv.waitKey(1) & 0xFF

        cv.imshow('Live Calibration', frame)

        if key == ord('q'):
            print("Calibration cancelled by user.")
            cap.release()
            cv.destroyAllWindows()
            return None, None
            
    # Release camera and close windows
    cap.release()
    cv.destroyAllWindows()
    
    if len(objpoints) < 5:
        print("Calibration failed: Not enough frames were captured.")
        return None, None

    print("\nAll frames captured. Calibrating camera...")

    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if ret:
        print("Calibration successful!")
        return camera_matrix, dist_coeffs
    else:
        print("Calibration failed.")
        return None, None


if __name__ == '__main__':
    camera_matrix, dist_coeffs = live_camera_calibration()

    if camera_matrix is not None and dist_coeffs is not None:
        print("\n------------------ INTRINSICS ------------------")
        print("\nCamera Matrix (K):\n", camera_matrix)
        print("\nDistortion Coefficients (k1, k2, p1, p2, k3):\n", dist_coeffs)
        print("\n------------------------------------------------")
