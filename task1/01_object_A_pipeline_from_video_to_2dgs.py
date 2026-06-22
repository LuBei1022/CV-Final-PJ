
"""
Object A final pipeline:
original video -> 36 frames -> automatic background cleanup
-> COLMAP -> undistort -> 2DGS training/render.

This is a cleaned Python version of the notebook workflow.
"""

from pathlib import Path
import subprocess
import shutil
import cv2
import numpy as np

PROJECT = Path("/root/autodl-tmp/CV_Final_Project/problem1_2dgs_aigc")
VIDEO = PROJECT / "data/object_A_original/raw_video/origin.mp4"
FRAME_DIR = PROJECT / "data/object_A_original_manual36/original_images_for_colmap"
CLEAN_DIR = PROJECT / "data/object_A_original_auto36_v2/images"
COLMAP_DIR = PROJECT / "data/object_A_original_auto36_v2_colmap"
UNDIST = PROJECT / "data/object_A_original_auto36_v2_undistorted"
GS = PROJECT / "code/2d-gaussian-splatting"
OUT = PROJECT / "outputs/object_A_original/object_A_original_auto36_v2_3000_retry2"

def run(cmd, cwd=None):
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)

def extract_36_frames():
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(int(total*0.05), int(total*0.95), 36).round().astype(int)
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(FRAME_DIR / f"origin_f{i:03d}.jpg"), frame)
    cap.release()

def auto_background_cleanup_v2():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(FRAME_DIR.glob("*.jpg")):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]

        # GrabCut foreground
        scale = 900 / max(h, w)
        small = cv2.resize(img, (int(w*scale), int(h*scale))) if scale < 1 else img.copy()
        sh, sw = small.shape[:2]
        rect = (int(sw*0.08), int(sh*0.08), int(sw*0.84), int(sh*0.84))
        mask = np.zeros((sh, sw), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
        if scale < 1:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(m, (11, 11), 0)
        alpha = m.astype(np.float32)[..., None] / 255.0
        out = (img.astype(np.float32) * alpha).astype(np.uint8)

        # Remove large white/gray leftover regions
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        R, G, B = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
        near_white = ((S < 65) & (V > 105) & (abs(R-G) < 45) & (abs(G-B) < 45) & (abs(R-B) < 45)).astype(np.uint8) * 255
        num, labels, stats, _ = cv2.connectedComponentsWithStats(near_white, connectivity=8)
        remove = np.zeros((h, w), np.uint8)
        for lab in range(1, num):
            x, y, ww, hh, area = stats[lab]
            touches = x <= 2 or y <= 2 or x + ww >= w - 3 or y + hh >= h - 3
            if touches or area > 1800:
                remove[labels == lab] = 255
        remove = cv2.dilate(remove, np.ones((5, 5), np.uint8), iterations=1)
        out[remove > 0] = 0
        cv2.imwrite(str(CLEAN_DIR / p.name), out)

def run_colmap():
    if COLMAP_DIR.exists():
        shutil.rmtree(COLMAP_DIR)
    (COLMAP_DIR / "images").mkdir(parents=True)
    (COLMAP_DIR / "sparse").mkdir(parents=True)
    for p in CLEAN_DIR.glob("*"):
        shutil.copy2(p, COLMAP_DIR / "images" / p.name)

    db = COLMAP_DIR / "database.db"
    run(["colmap", "feature_extractor", "--database_path", db, "--image_path", COLMAP_DIR/"images",
         "--ImageReader.single_camera", "1", "--SiftExtraction.use_gpu", "0",
         "--SiftExtraction.max_image_size", "1600", "--SiftExtraction.max_num_features", "8192",
         "--SiftExtraction.domain_size_pooling", "1"])
    run(["colmap", "exhaustive_matcher", "--database_path", db, "--SiftMatching.use_gpu", "0",
         "--SiftMatching.guided_matching", "1", "--SiftMatching.max_error", "8"])
    run(["colmap", "mapper", "--database_path", db, "--image_path", COLMAP_DIR/"images",
         "--output_path", COLMAP_DIR/"sparse", "--Mapper.multiple_models", "0",
         "--Mapper.min_num_matches", "4", "--Mapper.abs_pose_min_num_inliers", "8",
         "--Mapper.init_min_num_inliers", "20"])

def undistort():
    if UNDIST.exists():
        shutil.rmtree(UNDIST)
    run(["colmap", "image_undistorter", "--image_path", COLMAP_DIR/"images",
         "--input_path", COLMAP_DIR/"sparse/0", "--output_path", UNDIST,
         "--output_type", "COLMAP", "--max_image_size", "1600"])

def train_render():
    run(["python", "train.py", "-s", UNDIST, "-m", OUT, "-r", "1",
         "--iterations", "3000",
         "--test_iterations", "1000", "2000", "3000",
         "--save_iterations", "1000", "2000", "3000",
         "--checkpoint_iterations", "1000", "2000", "3000"], cwd=GS)
    run(["python", "render.py", "-s", UNDIST, "-m", OUT, "--iteration", "3000"], cwd=GS)

if __name__ == "__main__":
    print("Documentation script. Uncomment calls to rerun full pipeline.")
    # extract_36_frames()
    # auto_background_cleanup_v2()
    # run_colmap()
    # undistort()
    # train_render()
