### Import OpenCV for homography and image manipulation
import cv2
import numpy as np
import torch
from matplotlib import pyplot as pl

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

if __name__ == '__main__':
    device = 'cuda'
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    
    # --- IMPORTANT: Change these to your image paths ---
    image_paths = ['/home/kabs_d/mast3r/floor2_with_fitted_grid_and_extra_columns_v4.jpg', 
                   '/home/kabs_d/mast3r/sec.jpg']
    images = load_images(image_paths, size=512)
    
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    # At this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # Find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # Ignore small border around the edge
    H0_orig, W0_orig = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0_orig) - 3) & \
                        (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0_orig) - 3)

    H1_orig, W1_orig = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1_orig) - 3) & \
                        (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1_orig) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
    
    print(f"Found {len(matches_im0)} valid matches.")

    # Prepare images for visualization and warping
    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    # Convert tensors to NumPy arrays in RGB format
    source_img_rgb = (view1['img'] * image_std + image_mean).squeeze(0).permute(1, 2, 0).cpu().numpy()
    target_img_rgb = (view2['img'] * image_std + image_mean).squeeze(0).permute(1, 2, 0).cpu().numpy()

    ### -------------------------------------------------------------------------- ###
    ### --- STEP 1: Display the Initial Matches Found by MASt3R --- ###
    ### -------------------------------------------------------------------------- ###
    
    num_matches = len(matches_im0)
    if num_matches > 0:
        n_viz = 95  # Number of matches to visualize
        
        # Select a subset of matches to display
        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, min(n_viz, num_matches))).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        # Get image dimensions
        H0, W0, _ = source_img_rgb.shape
        H1, W1, _ = target_img_rgb.shape
        
        # Create a new image by concatenating the two images side-by-side
        # Pad the shorter image to match the height of the taller one
        if H0 < H1:
            padding = np.zeros((H1 - H0, W0, 3), dtype=source_img_rgb.dtype)
            img0_padded = np.vstack((source_img_rgb, padding))
            img1_padded = target_img_rgb
        else:
            padding = np.zeros((H0 - H1, W1, 3), dtype=target_img_rgb.dtype)
            img0_padded = source_img_rgb
            img1_padded = np.vstack((target_img_rgb, padding))

        side_by_side_img = np.concatenate((img0_padded, img1_padded), axis=1)

        # Create the plot for matches
        pl.figure(figsize=(16, 8))
        pl.imshow(side_by_side_img)
        pl.title(f"MASt3R Matches (showing {len(viz_matches_im0)} of {num_matches})")
        pl.axis('off')

        # Draw lines connecting the matches
        cmap = pl.get_cmap('jet')
        for i in range(len(viz_matches_im0)):
            (x0, y0), (x1, y1) = viz_matches_im0[i], viz_matches_im1[i]
            # The x-coordinate for the second image needs to be offset by the width of the first image
            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (len(viz_matches_im0))), scalex=False, scaley=False)
        
        print("\nDisplaying initial matches. Close the plot window to continue to homography.")
        pl.show()  # This shows the first plot

    ### -------------------------------------------------------------------------- ###
    ### --- STEP 2: Compute Homography, Warp Image, and Display Result --- ###
    ### -------------------------------------------------------------------------- ###

    MIN_MATCH_COUNT = 4
    if num_matches > MIN_MATCH_COUNT:
        # Compute the homography matrix using RANSAC
        H, mask = cv2.findHomography(matches_im0, matches_im1, cv2.RANSAC, 5.0)
        print("\nHomography Matrix:\n", H)

        # Warp the source image to the target image's perspective
        h, w = target_img_rgb.shape[:2]
        warped_img = cv2.warpPerspective(source_img_rgb, H, (w, h))

        # Blend the warped image with the target image for a nice overlay effect
        blended_img = cv2.addWeighted(target_img_rgb, 0.7, warped_img, 0.3, 0)

        # Display the results
        fig, (ax1, ax2, ax3) = pl.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Homography Image Warping Result', fontsize=16)

        ax1.imshow(source_img_rgb); ax1.set_title('Source Image'); ax1.axis('off')
        ax2.imshow(target_img_rgb); ax2.set_title('Target Image'); ax2.axis('off')
        ax3.imshow(blended_img); ax3.set_title('Warped Source Overlaid on Target'); ax3.axis('off')

        pl.tight_layout()
        pl.show() # This shows the second plot

        # Save the output images
        warped_to_save = cv2.cvtColor((warped_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        blended_to_save = cv2.cvtColor((blended_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        cv2.imwrite('warped_output.jpg', warped_to_save)
        cv2.imwrite('blended_output.jpg', blended_to_save)
        print("\nSaved 'warped_output.jpg' and 'blended_output.jpg' successfully.")

    else:
        print(f"\nNot enough matches are found ({num_matches}/{MIN_MATCH_COUNT}) to compute a reliable homography.")
