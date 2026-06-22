
"""
Extract RGB object from threestudio triptych video frames:
left panel = RGB, middle panel = normal, right panel = mask.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image

PROJECT = Path("/root/autodl-tmp/CV_Final_Project/problem1_2dgs_aigc")
ASSET = PROJECT / "outputs/fusion_assets"
COLAB = ASSET / "colab_assets_unzipped/colab_assets"
OUT = ASSET / "selected_final_assets"
OUT.mkdir(parents=True, exist_ok=True)

def read_frame(path, idx=0):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(path)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def crop_to_alpha(rgba, pad=10):
    alpha = rgba[..., 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return rgba
    h, w = rgba.shape[:2]
    return rgba[max(0,ys.min()-pad):min(h,ys.max()+pad+1), max(0,xs.min()-pad):min(w,xs.max()+pad+1)]

def extract_triptych(rgb):
    h, w = rgb.shape[:2]
    pw = w // 3
    rgb_panel = rgb[:, :pw, :]
    mask_panel = rgb[:, 2*pw:3*pw, :]
    gray = cv2.cvtColor(mask_panel, cv2.COLOR_RGB2GRAY)
    alpha = np.where(gray > 30, 255, 0).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    return crop_to_alpha(np.dstack([rgb_panel, alpha]))

if __name__ == "__main__":
    b = extract_triptych(read_frame(COLAB / "object_B_plant_turntable.mp4", 0))
    c = extract_triptych(read_frame(COLAB / "object_C_mug_turntable.mp4", 0))
    Image.fromarray(b).save(OUT / "object_B_selected_frame_000.png")
    Image.fromarray(c).save(OUT / "object_C_selected_white_mug_only.png")
