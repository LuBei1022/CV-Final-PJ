
"""
Generate final fusion video:
A toy on table, B plant on floor-left, C mug on table with rotating view.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import math

PROJECT = Path("/root/autodl-tmp/CV_Final_Project/problem1_2dgs_aigc")
ASSET = PROJECT / "outputs/fusion_assets"
COLAB = ASSET / "colab_assets_unzipped/colab_assets"

BG = ASSET / "selected_final_assets/background_selected_000.png"
A_DIR = PROJECT / "outputs/object_A_original/object_A_original_auto36_v2_3000_retry2/train/ours_3000/renders"
B_VIDEO = COLAB / "object_B_plant_turntable.mp4"
C_VIDEO = COLAB / "object_C_mug_turntable.mp4"
OUT_DIR = ASSET / "final_fusion_video_FINAL"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = {
    "A": {"target_w_ratio": 0.105, "center_x_ratio": 0.375, "bottom_y_ratio": 0.740},
    "B": {"target_w_ratio": 0.230, "center_x_ratio": 0.190, "bottom_y_ratio": 0.970},
    "C": {"target_w_ratio": 0.108, "center_x_ratio": 0.740, "bottom_y_ratio": 0.715},
}

def read_rgb(p):
    im = cv2.imread(str(p))
    if im is None:
        raise RuntimeError(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def video_frames(p):
    cap = cv2.VideoCapture(str(p))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return out

def crop_alpha(rgba, pad=10):
    a = rgba[..., 3]
    ys, xs = np.where(a > 10)
    if len(xs) == 0:
        return rgba
    h, w = rgba.shape[:2]
    return rgba[max(0,ys.min()-pad):min(h,ys.max()+pad+1), max(0,xs.min()-pad):min(w,xs.max()+pad+1)]

def black_to_alpha(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    a = np.where(hsv[..., 2] > 18, 255, 0).astype(np.uint8)
    a = cv2.GaussianBlur(a, (5,5), 0)
    return crop_alpha(np.dstack([rgb, a]))

def triptych(rgb):
    h, w = rgb.shape[:2]
    pw = w // 3
    left = rgb[:, :pw, :]
    mask = rgb[:, 2*pw:3*pw, :]
    gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    a = np.where(gray > 30, 255, 0).astype(np.uint8)
    a = cv2.GaussianBlur(a, (5,5), 0)
    return crop_alpha(np.dstack([left, a]))

def resize_rgba(rgba, target_w):
    h, w = rgba.shape[:2]
    target_h = max(1, int(h * target_w / max(w, 1)))
    return cv2.resize(rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

def overlay(base, obj, cx, by):
    out = base.copy()
    H, W = out.shape[:2]
    h, w = obj.shape[:2]
    x1, y1 = int(cx - w/2), int(by - h)
    x2, y2 = x1 + w, y1 + h
    ox1, oy1 = max(0, -x1), max(0, -y1)
    ox2, oy2 = w - max(0, x2-W), h - max(0, y2-H)
    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = bx1 + (ox2-ox1), by1 + (oy2-oy1)
    if bx1 >= bx2 or by1 >= by2:
        return out
    crop = obj[oy1:oy2, ox1:ox2].astype(np.float32)
    rgb = crop[..., :3]
    a = crop[..., 3:4] / 255.0
    bg = out[by1:by2, bx1:bx2].astype(np.float32)
    out[by1:by2, bx1:bx2] = (rgb*a + bg*(1-a)).astype(np.uint8)
    return out

def main():
    bg = read_rgb(BG)
    H, W = bg.shape[:2]
    A = [black_to_alpha(read_rgb(p)) for p in sorted(A_DIR.glob("*.png"))]
    B_frames = video_frames(B_VIDEO)
    C_frames = video_frames(C_VIDEO)
    B = triptych(B_frames[0])
    C = [triptych(f) for f in C_frames]

    clean_path = OUT_DIR / "final_fusion_A_B_C_FINAL_clean.mp4"
    labeled_path = OUT_DIR / "final_fusion_A_B_C_FINAL_labeled.mp4"
    preview_path = OUT_DIR / "final_fusion_A_B_C_FINAL_preview_sheet.jpg"

    def make_frame(i, label=False):
        frame = bg.copy()
        a = resize_rgba(A[i % len(A)], int(W * PARAMS["A"]["target_w_ratio"]))
        b = resize_rgba(B, int(W * PARAMS["B"]["target_w_ratio"]))
        c = resize_rgba(C[i % len(C)], int(W * PARAMS["C"]["target_w_ratio"]))
        frame = overlay(frame, b, W*PARAMS["B"]["center_x_ratio"], H*PARAMS["B"]["bottom_y_ratio"])
        frame = overlay(frame, a, W*PARAMS["A"]["center_x_ratio"], H*PARAMS["A"]["bottom_y_ratio"])
        frame = overlay(frame, c, W*PARAMS["C"]["center_x_ratio"], H*PARAMS["C"]["bottom_y_ratio"])
        if label:
            pil = Image.fromarray(frame)
            d = ImageDraw.Draw(pil)
            d.rectangle([14,14,330,50], fill=(255,255,255))
            d.text((24,24), "Fusion: A toy + B plant + C mug", fill=(0,0,0))
            frame = np.array(pil)
        return frame

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for path, label in [(clean_path, False), (labeled_path, True)]:
        writer = cv2.VideoWriter(str(path), fourcc, 24, (W,H))
        for i in range(120):
            writer.write(cv2.cvtColor(make_frame(i, label), cv2.COLOR_RGB2BGR))
        writer.release()

    frames = [make_frame(i, True) for i in [0,20,40,60,80,100]]
    thumb_w = 360
    thumb_h = int(H * thumb_w / W)
    sheet = Image.new("RGB", (3*thumb_w, 2*(thumb_h+28)), "white")
    d = ImageDraw.Draw(sheet)
    for idx, fr in enumerate(frames):
        im = Image.fromarray(fr)
        im.thumbnail((thumb_w, thumb_h))
        x = (idx % 3) * thumb_w
        y = (idx // 3) * (thumb_h+28)
        d.text((x+5,y+5), f"preview frame {idx}", fill=(0,0,0))
        sheet.paste(im, (x, y+28))
    sheet.save(preview_path, quality=95)

if __name__ == "__main__":
    main()
