This project is a collection of Python scripts, focusing on camera calibration, 3D pose estimation, and homography refinement. It demonstrates a powerful workflow that blends classical geometric vision algorithms from OpenCV with modern, gradient-based optimization in PyTorch and deep-learning-based feature matching with the MASt3R model.

Key Features
 
  Live Camera Calibration: Determine a camera's intrinsic matrix and distortion coefficients using a live video feed.

  Interactive Pose Estimation: Interactively select 2D-3D point correspondences to calculate a camera's 3D position and orientation using OpenCV's solvePnP.

  Deep Learning Feature Matching: Utilize the MASt3R model to find robust, dense correspondences between images for accurate homography estimation.

  Gradient-Based Optimization: Refine camera parameters and homography matrices by treating them as learnable tensors in PyTorch and minimizing reprojection error.

  Perspective Grid Visualization: Project 3D world grids onto 2D images to visually verify the accuracy of calculated camera poses and homographies.
  
  Mathematical Model Fitting: Analyze point data to find the best-fitting mathematical function for extrapolation and forecasting.

Camera_Calibration_Script.py : This script performs live camera calibration using a checkerboard pattern and OpenCV. It captures video from a webcam, detects the checkerboard corners in multiple frames, and allows the user to capture valid views. The script then calculates the camera's intrinsic matrix and distortion coefficients, printing the results to the console. These values are crucial for correcting lens distortion in subsequent computer vision applications.

Optimizing_homography.py : This script refines an initial homography matrix by treating its parameters as learnable tensors in PyTorch. It minimizes the reprojection error between a hard-coded set of corresponding points from two images. The script outputs the optimized homography matrix, saves a loss curve graph, and generates visualization images showing the warped image and the alignment accuracy.

api1.py: This script utilizes the gradio_client to interface with a remote Image Matching API hosted on Hugging Face Spaces. It uploads two local images to the service to find keypoint matches and compute a homography matrix between them. The script then uses this computed geometry to have the API generate a warped image, which is finally downloaded and displayed using OpenCV.

comparison.py: This script provides a visual analysis comparing predicted Y-coordinates against ground truth data. It uses Matplotlib to plot Y-values extrapolated from a rational decay function alongside several hard-coded lists of observed ground truth points. The resulting graph is used to evaluate the accuracy of the extrapolation model.

function_tester.py: This script is designed to find the best mathematical model for a given set of data points. It tests several candidate functions (e.g., linear, quadratic, exponential, and rational decay) by fitting them to the data. After identifying the best model based on the lowest Root Mean Square Error, it prints the derived formula, plots the results, and forecasts future values.

intrinsic_optimization.py: This script refines camera intrinsic and extrinsic parameters through gradient-based optimization. It first uses OpenCV's solvePnP on interactively-clicked 2D-3D point correspondences to get an initial pose estimate. It then treats the camera parameters as learnable tensors in PyTorch, to minimize the reprojection error and find more accurate values.

my_click.py: This is a simple, interactive utility script using OpenCV. It displays an image and captures the pixel coordinates of points selected by the user's mouse clicks. It then prints the final list of coordinates to the console, making it useful for quickly gathering ground truth data.

persp_grid(intrinsics).py: This script calculates the camera's 3D pose (its rotation and translation) relative to a defined world coordinate system. It first prompts the user to interactively click on 8 points in an image that correspond to known 3D world coordinates. It then computes the camera's extrinsics using OpenCV's solvePnP function. Finally, it uses this calculated pose to project an entire predefined 3D grid back onto the 2D image, visualizing the result with a perspective-correct grid overlay.

persp_grid_local1.py: This script overlays a perspective grid onto a local image using a pre-calculated homography matrix. It takes a hard-coded set of grid points from a "global" source image and uses cv2.perspectiveTransform to project them onto the local target image. The script then visualizes this mapping by drawing the grid:
  Vertical lines are drawn by fitting a line to the projected points of each column, correctly accounting for perspective.
  Horizontal lines are drawn using a special averaging technique on two reference columns to ensure the lines are perfectly parallel in the final view.
  The final output is an image with the complete, transformed grid drawn on it, demonstrating the geometric mapping between the two views.

perspective_grid(intrinsics)_skewed_left.py: This file serves the same purpose as the persp_grid(intrinsics).py file but for the left skewed image.

perspective_grid(intrinsics)_skewed_right.py: This file serves the same purpose as the persp_grid(intrinsics).py file but for the right skewed image.

warping_mast3r.py: This script performs image matching and alignment using the advanced MASt3R deep learning model. It loads the pretrained model to find dense and robust feature correspondences between two images. From these matches, it accurately computes a homography matrix using RANSAC and then uses this matrix with cv2.warpPerspective to warp one image, aligning its perspective with the other. The script handles the necessary scaling between the model's inference size and the original image resolution to ensure an accurate final transformation. The output includes the warped image and a visualization of the feature matches.
