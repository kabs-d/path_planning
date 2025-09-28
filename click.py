#Interactive Script to get pixel coordinates of clicked points
import cv2

# Load image
img_path = "/home/kabs_d/mast3r/floor2_with_fitted_grid_and_extra_columns_v4.jpg"
image = cv2.imread(img_path)

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 10:
        clicked_points.append([x, y])
        print(f"Clicked point {len(clicked_points)}: ({x}, {y})")

# Create window and set callback
cv2.namedWindow("Click 10 points")
cv2.setMouseCallback("Click 10 points", mouse_callback)

print("Click 10 points in order. Press ESC when done.")

while True:
    disp = image.copy()
    # Draw points + labels
    for i, pt in enumerate(clicked_points):
        cv2.circle(disp, tuple(pt), 5, (0, 0, 255), -1)
        cv2.putText(disp, str(i+1), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 255), 2)
    
    cv2.imshow("Click 10 points", disp)

    key = cv2.waitKey(20) & 0xFF
    # ESC to break
    if key == 27 or len(clicked_points) >= 10:
        break

cv2.destroyAllWindows()

print("Final clicked points:", clicked_points)
