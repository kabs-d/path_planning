#!/usr/bin/env python3
#overlaying perpsective grid on local image using corner point pixels of global image and using optimized homography matrix
import cv2
import numpy as np

if __name__ == '__main__':
    # Define the hard-coded homography matrix
    H = np.array([[ 4.82022100e-01 ,-1.80439926e+00 , 8.99994220e+02],
 [-8.72638433e-03, 6.42153559e-01, -3.25468825e+02],
 [-2.34114875e-05 ,-1.30960024e-03 , 1.00000000e+00]], dtype=np.float64)

    # Use your actual image file paths here
    local_image_path = "/home/kabs_d/mast3r/sec.jpg"

    # Load the original image
    img_local_orig = cv2.imread(local_image_path)
    if img_local_orig is None:
        raise FileNotFoundError("Local image not found. Check file path.")

    # Source points from the global image, structured by their intended vertical column
    source_columns = {
        0.0: [[343, 830], [289, 962], [372, 757], [391, 711], [404, 679], [414, 655], [421, 637], [427, 623], [432, 612], [435, 602], [439, 594], [441, 587], [444, 582], [446, 577], [448, 572], [449, 568], [451, 565], [452, 561], [453, 558], [454, 556], [455, 554]],
        54.0: [[606, 831], [658, 963], [577, 758], [558, 711], [546, 679], [536, 656], [529, 638], [524, 623], [519, 612], [515, 602], [512, 594], [509, 588], [507, 582], [505, 577], [503, 572], [502, 568], [500, 565], [499, 561], [498, 559], [497, 556], [496, 554]],
        111.0: [[539, 554], [542, 556], [545, 559], [549, 562], [553, 565], [557, 568], [562, 572], [568, 577], [574, 582], [581, 588], [590, 595], [600, 603], [612, 612], [626, 624], [644, 638], [666, 656], [695, 680], [735, 712], [793, 758], [884, 832], [1047, 964]],
        -57.0: [[412, 553], [409, 556], [406, 558], [402, 561], [398, 564], [394, 568], [389, 572], [383, 576], [377, 581], [369, 587], [361, 594], [351, 602], [339, 612], [325, 623], [307, 637], [284, 655], [255, 679], [215, 711], [157, 757], [65, 830], [-99, 961]],
        168.0: [[582, 554], [587, 556], [593, 559], [599, 562], [605, 565], [613, 568], [621, 572], [630, 577], [641, 582], [653, 588], [667, 595], [684, 603], [704, 612], [728, 624], [758, 638], [795, 656], [845, 680], [912, 712], [1009, 759], [1162, 832], [1437, 965]],
        -114.0: [[369, 553], [364, 556], [358, 558], [352, 561], [346, 564], [338, 568], [330, 572], [321, 576], [310, 581], [298, 587], [283, 594], [267, 602], [247, 611], [223, 623], [193, 637], [155, 655], [106, 678], [38, 710], [-59, 756], [-212, 829], [-488, 960]]
    }
    
    vis_image = img_local_orig.copy()
    height, width = vis_image.shape[:2]

    # --- Draw Vertical Lines ---
    print("Drawing vertical lines...")
    for group_id, source_points_list in source_columns.items():
        if len(source_points_list) < 2: continue
        
        source_np = np.float32(source_points_list).reshape(-1, 1, 2)
        projected_column_pts = cv2.perspectiveTransform(source_np, H).reshape(-1, 2)
        
        y_coords = projected_column_pts[:, 1]
        x_coords = projected_column_pts[:, 0]
        m, b = np.polyfit(y_coords, x_coords, 1)
        
        x_top = int(m * 0 + b)
        x_bottom = int(m * (height - 1) + b)
        cv2.line(vis_image, (x_top, 0), (x_bottom, height - 1), (255, 0, 0), 2) # Blue

    # --- Draw Horizontal Lines using the new averaging logic ---
    print("Drawing horizontal lines using 5th and 6th columns...")
    
    # To be deterministic, sort the keys numerically first
    sorted_keys = sorted(source_columns.keys())
    
    # --- MODIFICATION: Select the 5th (index 4) and 6th (index 5) keys ---
    key_5 = sorted_keys[4] # This will be 111.0
    key_6 = sorted_keys[5] # This will be 168.0
    
    print(f"Using columns with keys: {key_5} and {key_6}")
    
    col1_pts = source_columns[key_5]
    col2_pts = source_columns[key_6]
    num_points_per_column = min(len(col1_pts), len(col2_pts))

    for i in range(num_points_per_column):
        # 1. Get the i-th point from each of the two selected columns
        pt1_source = col1_pts[i]
        pt2_source = col2_pts[i]
        
        # 2. Project just this pair of points
        source_pair_np = np.float32([pt1_source, pt2_source]).reshape(-1, 1, 2)
        projected_pair = cv2.perspectiveTransform(source_pair_np, H).reshape(-1, 2)
        
        # 3. Calculate the average of their projected y-coordinates
        y1 = projected_pair[0][1]
        y2 = projected_pair[1][1]
        y_avg = int((y1 + y2) / 2)
        
        # 4. Draw a perfectly horizontal line (slope 0) across the image
        if 0 <= y_avg < height:
             cv2.line(vis_image, (0, y_avg), (width - 1, y_avg), (0, 255, 0), 2) # Green

    # --- Draw all projected points on top ---
    all_source_points = [pt for col in source_columns.values() for pt in col]
    all_source_np = np.float32(all_source_points).reshape(-1, 1, 2)
    all_projected_pts = cv2.perspectiveTransform(all_source_np, H).reshape(-1, 2)
    
    for pt in all_projected_pts:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(vis_image, (x, y), 4, (0, 0, 255), -1) # Red

    # --- Display the final result ---
    cv2.imshow('Local Image with Full Projected Grid', vis_image)
    cv2.imwrite('local_grid.jpg',vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
