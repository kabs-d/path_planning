import torch
import numpy as np
import cv2 as cv

# Minimal imports from the original libraries
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    # --- 1. DEFINE IMAGE PATHS ---
    image1_path = '/home/kabs_d/mast3r/floor2_with_fitted_grid_and_extra_columns_v4.jpg'
    image2_path = '/home/kabs_d/mast3r/sec.jpg'
    
    # --- 2. FIND MATCHES ---
    images = load_images([image1_path, image2_path], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # Filter out matches near the border
    H0_orig_tensor, W0_orig_tensor = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < W0_orig_tensor.item() - 3) & \
                        (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < H0_orig_tensor.item() - 3)
    
    H1_orig_tensor, W1_orig_tensor = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < W1_orig_tensor.item() - 3) & \
                        (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < H1_orig_tensor.item() - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0 = matches_im0[valid_matches]
    matches_im1 = matches_im1[valid_matches]
    
    print(f"Found {len(matches_im0)} valid matches.")

    # --- 3. COMPUTE HOMOGRAPHY AND WARP ---
    if len(matches_im0) >= 4:
        
        # Get original and resized image dimensions
        H0_orig, W0_orig = view1['true_shape'][0]
        H0_resized, W0_resized = view1['img'].shape[-2:]

        H1_orig, W1_orig = view2['true_shape'][0]
        H1_resized, W1_resized = view2['img'].shape[-2:]

        # Calculate scaling factors
        scale0_w = W0_orig.item() / W0_resized
        scale0_h = H0_orig.item() / H0_resized
        scale1_w = W1_orig.item() / W1_resized
        scale1_h = H1_orig.item() / H1_resized
        
        # Convert integer coordinate arrays to float arrays for scaling
        src_pts = matches_im0.astype(np.float32)
        dst_pts = matches_im1.astype(np.float32)

        # Scale the points to match the original image dimensions
        src_pts[:, 0] *= scale0_w
        src_pts[:, 1] *= scale0_h
        dst_pts[:, 0] *= scale1_w
        dst_pts[:, 1] *= scale1_h
    
        # Find the transformation matrix using the correctly scaled points
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        if H is not None:
            # Load the original, full-resolution images
            img_to_warp = cv.imread(image1_path)
            img_reference = cv.imread(image2_path)
            
            # Get the size of the reference image for the output
            h, w, _ = img_reference.shape
            
            # Warp the source image to align with the reference image
            warped_image = cv.warpPerspective(img_to_warp, H, (w, h))
            
            # --- 4. SAVE AND DISPLAY THE WARPED RESULT ---
            output_filename = 'warped_output_accurate.jpg'
            cv.imwrite(output_filename, warped_image)
            print(f"\n✅ Success! Warped image saved as '{output_filename}'")

            # --- 5. VISUALIZE ALL MATCHES WITH THICKER LINES (Updated Section) ---
            
            h1, w1 = img_to_warp.shape[:2]
            h2, w2 = img_reference.shape[:2]

            # Create a new canvas to draw the matches on
            match_canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
            match_canvas[0:h1, 0:w1] = img_to_warp
            match_canvas[0:h2, w1:w1 + w2] = img_reference

            # Draw lines for ALL matches
            for i in range(len(src_pts)):
                pt1 = (int(src_pts[i][0]), int(src_pts[i][1]))
                pt2 = (int(dst_pts[i][0] + w1), int(dst_pts[i][1])) # Offset x-coordinate for the second image
                
                # Draw a random color line with thickness 2
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv.line(match_canvas, pt1, pt2, color, thickness=2)

            # --- 6. DISPLAY ALL IMAGES ---
            print("Displaying images. Press any key to close all windows.")
            cv.imshow("Reference Image", img_reference)
            cv.imshow("Warped Output (Accurate)", warped_image)
            cv.imshow("MASt3R Matches (All)", match_canvas) # Display the new visualization
            
            cv.waitKey(0)
            cv.destroyAllWindows()
            
        else:
            print("\n❌ Error: Could not compute a valid homography matrix.")
    else:
        print("\n❌ Error: Not enough matches found to compute homography (need at least 4).")
