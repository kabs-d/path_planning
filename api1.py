#using api of image matching web ui for warping
from gradio_client import Client, handle_file
import cv2

# --- Paths to your images ---
IMG1_PATH = "/home/kabs_d/mast3r/floor2_with_fitted_grid_and_extra_columns_v4.jpg"
IMG0_PATH = "/home/kabs_d/mast3r/sec.jpg"

# --- Step 1: Connect to the API ---
client = Client("https://realcat-image-matching-webui.hf.space")

# --- Step 2: Run image matching ---
matches = client.predict(
    image0=handle_file(IMG0_PATH),
    image1=handle_file(IMG1_PATH),
    match_threshold=0.2,
    extract_max_keypoints=2000,
    keypoint_threshold=0.05,
    key="Mast3R",
    ransac_method="CV2_USAC_MAGSAC",
    ransac_reproj_threshold=8,
    ransac_confidence=0.9999,
    ransac_max_iter=10000,
    choice_geometry_type="Homography",
    force_resize=False,
    image_width=640,
    image_height=480,
    api_name="/run_matching"
)

# matches[5] contains "Reconstructed Geometry" used as matches_info
matches_info = matches[5]

if matches_info is None:
    raise RuntimeError("No reconstructed geometry returned. Matching failed.")

# --- Step 3: Generate warped image ---
warp_res = client.predict(
    input_image0=handle_file(IMG0_PATH),
    input_image1=handle_file(IMG1_PATH),
    matches_info=matches_info,
    choice="Homography",
    api_name="/generate_warp_images"
)

print("Raw warp_res:", warp_res, type(warp_res))

# Handle different return formats
if isinstance(warp_res, str):
    warped_image_path = warp_res
elif isinstance(warp_res, (list, tuple)):
    warped_image_path = warp_res[0]
else:
    raise RuntimeError(f"Unexpected warp_res format: {type(warp_res)}")

print("Warped image saved at:", warped_image_path)

# --- Step 4: Load and display the warped image ---
warped_img = cv2.imread(warped_image_path)
if warped_img is None:
    raise RuntimeError("Failed to read warped image from path.")

cv2.imshow("Warped Image", warped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
