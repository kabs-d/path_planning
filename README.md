This project is a collection of Python scripts, focusing on camera calibration, 3D pose estimation, and homography refinement. It demonstrates a powerful workflow that blends classical geometric vision algorithms from OpenCV with modern, gradient-based optimization in PyTorch and deep-learning-based feature matching with the MASt3R model.

Key Features
 
  1) Live Camera Calibration: Determine a camera's intrinsic matrix and distortion coefficients using a live video feed.
  2) Interactive Pose Estimation: Interactively select 2D-3D point correspondences to calculate a camera's 3D position and orientation using OpenCV's solvePnP.
  3) Deep Learning Feature Matching: Utilize the MASt3R model to find robust, dense correspondences between images for accurate homography estimation.
  4) Gradient-Based Optimization: Refine camera parameters and homography matrices by treating them as learnable tensors in PyTorch and minimizing reprojection error.
  5) Perspective Grid Visualization: Project 3D world grids onto 2D images to visually verify the accuracy of calculated camera poses and homographies.
  6) Mathematical Model Fitting: Analyze point data to find the best-fitting mathematical function for extrapolation and forecasting.
Installation
To get the project running on your local machine, please follow these steps. It is highly recommended to use a Python virtual environment to avoid conflicts with other projects.

1. Clone the Repository
First, clone this repository to your local machine using git
```bash
git clone https://github.com/kabs-d/path_planning
cd path_planning
```


3. Install Standard Dependencies
The project's standard library dependencies are listed in the requirements.txt file. You can install all of them with a single command:

```bash
pip install -r requirements.txt
```
This will install the following required libraries:

Plaintext

opencv-python
numpy
torch
matplotlib
scipy
scikit-learn
gradio_client
3. Special Installation for MASt3R
Important: The MASt3R library is a deep learning model and is not available on the standard Python Package Index (PyPI). Therefore, it cannot be installed with the command above.

You must install it manually by following the official instructions from its source repository.

Action Required: Visit the official MASt3R GitHub repository and follow their setup guide to install the library and download the model weights.

Once you have completed these steps, your environment will be fully configured to run all the scripts in this project.
FILE DESCRIPTIONS:

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

Getting Started: A Complete Workflow from 3D Scene to 2D Grid
Welcome! This guide will walk you through the repository in the correct, logical order. The primary goal is to project a known 3D world grid onto a target "local" image. The key insight is that this is a multi-step process where one script's output becomes the next script's input.

Step 1 (One-Time Setup): Calibrate Your Camera
Before any 3D work, you must understand your camera's unique properties.

Action: Run Camera_Calibration_Script.py using a checkerboard pattern.

Goal: To calculate your camera's Intrinsic Matrix (K) and distortion coefficients.

Result: You now have the K matrix, which you only need to calculate once for your camera.

Step 2: Create the "Global" Reference Image with a Perfect Grid
This is the foundational step. We create our "ground truth"—an image with a perfect, perspective-correct grid projected onto it from a known 3D pose.

Action: Run persp_grid(intrinsics).py (or its skewed variations).

Inputs:

A "source" photograph (e.g., floor2.jpg).

Your camera's Intrinsic Matrix (K) from Step 1.

A predefined list of 3D world coordinates.

Process: The script will ask you to click on points in the source photo that correspond to the known 3D coordinates. It then calculates the camera's 3D pose (extrinsics) and uses this to draw a perfect, extensive grid on the image.

Result: You now have a new image file (e.g., floor2_with_grid.jpg). This is your "Global Image." It contains the crucial visual reference grid.

Step 3: Find the Transformation (Homography) to a New View
Now, you have a new photo of the same scene from a different angle—your "Local Image". Your goal is to find the Homography Matrix (H) that maps the grid from the Global Image to this new Local Image.

You have two main ways to do this:

Method A (Recommended - AI-Powered & Automatic)
This is the fastest and most robust method.

Action: Run warping_mast3r.py.

Inputs:

Your Global Image (with the grid) from Step 2.

Your new Local Image.

Process: The MASt3R model finds hundreds of robust matching points between the two images and calculates a highly accurate Homography Matrix (H).

Result: You now have the crucial H matrix that connects the two views.

Method B (Manual Control & Optimization)
Choose this if you want to specify the exact points for the transformation.

Action: Use my_click.py to get coordinates of key grid intersections on your Global Image.

Action: Use my_click.py again to get coordinates of the exact same intersections on your Local Image.

Action: Input these corresponding points into Optimizing_homography.py to compute a refined Homography Matrix (H).

Result: A highly precise H matrix based on your hand-picked points.

Step 4: Visualize the Grid on the Local Image
You have the H matrix. Now you can apply it to see the final result.

Option 1 (Recommended - Grid Projection)
This method is precise, clean, and computationally efficient.

Action: Run persp_grid_local1.py.

Inputs:

Your Local Image (the original, clean version).

The H matrix from Step 3.

The original list of grid point coordinates from the Global Image.

Process: The script uses the H matrix to mathematically transform only the grid point coordinates. It then draws a new, clean grid directly on top of your original Local Image.

Result: A high-quality image with a perfectly aligned, non-degraded perspective grid.

Option 2 (Full Image Warping)
This method transforms the entire image, which is useful for direct visual comparison but can reduce image quality.

Action: The warping_mast3r.py script already produces this as its primary output.

Process: It uses the H matrix to warp the entire Global Image (pixels, grid, and all) to fit the perspective of the Local Image.

Result: A new image that looks like the Local Image but is composed of pixels from the Global Image. The original Local Image is completely covered.
