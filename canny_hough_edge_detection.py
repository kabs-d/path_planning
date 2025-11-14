import cv2
import numpy as np
import math
from pathlib import Path
import csv

# =========================================
# CONFIG
# =========================================
IMG_PATH = "/home/kabs_d/mast3r/floor2.jpg"
OUT_DIR = Path("./hough_experiments_clean")
OUT_DIR.mkdir(exist_ok=True)

# =========================================
# Helper functions
# =========================================

def save_img(img, path):
    cv2.imwrite(str(path), img)

def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def draw_hough_lines(rgb, lines, color=(0, 0, 255), thickness=2):
    out = rgb.copy()
    if lines is None:
        return out
    for l in lines:
        x1, y1, x2, y2 = l[0]
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out

def compute_metrics(lines):
    if lines is None or len(lines) == 0:
        return {"n_lines": 0, "mean_len": 0.0, "pct_hv": 0.0}

    lengths = []
    hv_count = 0

    for l in lines:
        x1, y1, x2, y2 = l[0]
        L = math.hypot(x2 - x1, y2 - y1)
        lengths.append(L)

        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180
        if (angle <= 15) or (angle >= 75):
            hv_count += 1

    return {
        "n_lines": len(lengths),
        "mean_len": float(np.mean(lengths)),
        "pct_hv": float(100 * hv_count / len(lengths))
    }

def run_hough(edges, rho=1, theta=np.pi/180, thresh=100, minlen=50, maxgap=10):
    return cv2.HoughLinesP(edges, rho, theta, thresh,
                           minLineLength=minlen, maxLineGap=maxgap)

# =========================================
# Load image
# =========================================
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(IMG_PATH)

img_rgb = to_rgb(img_bgr)

# =========================================
# Experiment definitions (NO ROI, NO SOBEL)
# =========================================
EXPERIMENTS = [

    ("A_baseline", {
        "clahe": None,
        "blur": None,
        "canny": (100, 200),
        "hough": (100, 50, 10)
    }),

    ("B_clahe_mild", {
        "clahe": (1.5, (8, 8)),
        "blur": (3, 3, 1.0),
        "canny": (50, 150),
        "hough": (100, 50, 10)
    }),

    ("C_clahe_strong", {
        "clahe": (3.0, (8, 8)),
        "blur": (3, 3, 1.0),
        "canny": (50, 150),
        "hough": (100, 50, 10)
    }),

    ("D_gentle_blur_stronger_canny", {
        "clahe": (3.0, (8, 8)),
        "blur": (7, 7, 1.0),    # toned down
        "canny": (70, 160),     # toned down
        "hough": (80, 50, 10)
    }),
   
    ("E_strong_hough", {
        "clahe": (1.5, (8, 8)),
        "blur": (3, 3, 1.0),
        "canny": (50, 150),
        "hough": (120, 140, 15)  # strong Hough
    })

]

# =========================================
# Run experiments
# =========================================
CSV_PATH = OUT_DIR / "metrics.csv"

with open(CSV_PATH, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["exp", "n_lines", "mean_len", "pct_hv"])
    writer.writeheader()

    for exp_name, cfg in EXPERIMENTS:

        exp_dir = OUT_DIR / exp_name
        exp_dir.mkdir(exist_ok=True)

        # --- Step 1: grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # --- Step 2: CLAHE
        if cfg["clahe"] is not None:
            clip, grid = cfg["clahe"]
            clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
            gray = clahe_obj.apply(gray)

        # --- Step 3: blur
        if cfg["blur"] is not None:
            kx, ky, sigma = cfg["blur"]
            gray = cv2.GaussianBlur(gray, (kx, ky), sigma)

        # --- Step 4: Canny
        low, high = cfg["canny"]
        edges = cv2.Canny(gray, low, high)
        save_img(edges, exp_dir / "edges.png")

        # --- Step 5: Hough
        thr, minlen, gap = cfg["hough"]
        lines = run_hough(edges, rho=1, theta=np.pi/180,
                          thresh=thr, minlen=minlen, maxgap=gap)

        # --- Step 6: Overlay
        overlay = draw_hough_lines(img_rgb, lines)
        save_img(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), exp_dir / "hough_overlay.png")

        # --- Step 7: Metrics
        M = compute_metrics(lines)
        writer.writerow({
            "exp": exp_name,
            "n_lines": M["n_lines"],
            "mean_len": round(M["mean_len"], 2),
            "pct_hv": round(M["pct_hv"], 2)
        })

        print(f"Saved {exp_name} →", M)

print("Done. Output folder:", OUT_DIR)
