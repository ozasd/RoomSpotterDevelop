#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

# =========================
# Settings (no args)
# =========================
OUT_DIR = "stickers"

DICT_NAME = "DICT_4X4_50"
# your convention: [BL, BR, TR, TL]
CORNER_IDS = [0, 1, 2, 3]
NAME_MAP = {
    0: "ID0_BL",
    1: "ID1_BR",
    2: "ID2_TR",
    3: "ID3_TL",
}

# marker body size (px). Bigger = easier to paste/transform and still readable.
MARKER_PX = 1200

# quiet zone (white border) ratio relative to marker size
QUIET_RATIO = 0.25  # 0.2~0.3 recommended
QUIET_PX = int(round(MARKER_PX * QUIET_RATIO))

# for convenience pack image
PACK_COLS = 2
PACK_GAP = 80  # px gap between stickers in pack image

# =========================
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def make_marker_u8(dict_name: str, marker_id: int, marker_px: int) -> np.ndarray:
    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    img = np.zeros((marker_px, marker_px), dtype=np.uint8)
    aruco.generateImageMarker(aruco_dict, marker_id, marker_px, img, 1)
    return img

def add_quiet_zone(marker_u8: np.ndarray, quiet_px: int) -> np.ndarray:
    h, w = marker_u8.shape[:2]
    out = np.full((h + 2 * quiet_px, w + 2 * quiet_px), 255, dtype=np.uint8)
    out[quiet_px:quiet_px + h, quiet_px:quiet_px + w] = marker_u8
    return out

def save_png_gray(path: str, u8: np.ndarray):
    ok = cv2.imwrite(path, u8)
    if not ok:
        raise RuntimeError(f"Failed to write: {path}")

def build_pack(stickers_u8, cols=2, gap=80, bg=255):
    # stickers_u8: list of (name, u8)
    if not stickers_u8:
        return None

    rows = int(np.ceil(len(stickers_u8) / cols))
    h, w = stickers_u8[0][1].shape[:2]

    pack_h = rows * h + (rows + 1) * gap
    pack_w = cols * w + (cols + 1) * gap
    pack = np.full((pack_h, pack_w), bg, dtype=np.uint8)

    for idx, (name, img) in enumerate(stickers_u8):
        r = idx // cols
        c = idx % cols
        y1 = gap + r * (h + gap)
        x1 = gap + c * (w + gap)
        pack[y1:y1 + h, x1:x1 + w] = img

        # small label under sticker (optional)
        # keep it simple: write on pack, not on sticker
        cv2.putText(
            pack, name,
            (x1, y1 + h + int(gap * 0.6)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,), 2, cv2.LINE_AA
        )

    return pack

def main():
    ensure_dir(OUT_DIR)

    stickers = []
    for mid in CORNER_IDS:
        marker = make_marker_u8(DICT_NAME, mid, MARKER_PX)
        sticker = add_quiet_zone(marker, QUIET_PX)

        name = NAME_MAP.get(mid, f"ID{mid}")
        out_path = os.path.join(OUT_DIR, f"{name}.png")
        save_png_gray(out_path, sticker)

        stickers.append((name, sticker))
        print("Saved:", out_path)

    # pack image for quick drag & drop
    pack = build_pack(stickers, cols=PACK_COLS, gap=PACK_GAP, bg=255)
    if pack is not None:
        pack_path = os.path.join(OUT_DIR, "sticker_pack.png")
        save_png_gray(pack_path, pack)
        print("Saved:", pack_path)

    print("\nDone.")
    print(f"- DICT={DICT_NAME}")
    print(f"- marker_px={MARKER_PX}, quiet_px={QUIET_PX}")
    print(f"- output_dir={OUT_DIR}")

if __name__ == "__main__":
    main()
