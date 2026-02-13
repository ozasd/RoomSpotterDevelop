#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import numpy as np
from PIL import Image

import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# =========================
# 參數（你可自行調）
# =========================
OUT_DIR = "locator"          # 你要放的資料夾
BASE_NAME = "baseline"   # 檔名（不含副檔名）

OUT_PDF = os.path.join(OUT_DIR, f"{BASE_NAME}.pdf")
OUT_PNG = os.path.join(OUT_DIR, f"{BASE_NAME}.png")

DICT_NAME = "DICT_4X4_50"
CORNER_IDS = [0, 1, 2, 3]          # ID0=左下(原點), ID1=右下, ID2=右上, ID3=左上
MARKER_SIZE_MM = 40.0
MARGIN_MM = 15.0                  # 紙邊到 marker 外框距離
MARKER_RENDER_PX = 800            # marker 生成解析度（越大越銳利）
PNG_DPI = 300                     # PNG 預覽 DPI（列印建議用 PDF）

# =========================
# 建立輸出資料夾
# =========================
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 產生 ArUco marker 圖
# =========================
aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))

def marker_png_bytes(marker_id: int, size_px: int = 800) -> io.BytesIO:
    img = np.zeros((size_px, size_px), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px, img, 1)
    pil = Image.fromarray(img)
    bio = io.BytesIO()
    pil.save(bio, format="PNG")
    bio.seek(0)
    return bio

# =========================
# 1) 輸出 PDF（列印用）
# =========================
page_w_pt, page_h_pt = A4
marker_w_pt = MARKER_SIZE_MM * mm
margin_pt = MARGIN_MM * mm

# reportlab 座標原點在「左下」
# ID0=BL, ID1=BR, ID2=TR, ID3=TL
id0, id1, id2, id3 = CORNER_IDS
positions_pt = {
    id0: (margin_pt, margin_pt),  # BL
    id1: (page_w_pt - margin_pt - marker_w_pt, margin_pt),  # BR
    id2: (page_w_pt - margin_pt - marker_w_pt, page_h_pt - margin_pt - marker_w_pt),  # TR
    id3: (margin_pt, page_h_pt - margin_pt - marker_w_pt),  # TL
}

c = canvas.Canvas(OUT_PDF, pagesize=A4)
for mid in CORNER_IDS:
    bio = marker_png_bytes(mid, MARKER_RENDER_PX)
    x, y = positions_pt[mid]
    c.drawImage(ImageReader(bio), x, y, width=marker_w_pt, height=marker_w_pt, mask="auto")
c.showPage()
c.save()
print("Saved PDF:", OUT_PDF)

# =========================
# 2) 輸出 PNG（預覽用）
# =========================
# A4 = 210 x 297 mm
inch_per_mm = 1.0 / 25.4
page_w_px = int(round(210.0 * inch_per_mm * PNG_DPI))
page_h_px = int(round(297.0 * inch_per_mm * PNG_DPI))
marker_px = int(round(MARKER_SIZE_MM * inch_per_mm * PNG_DPI))
margin_px = int(round(MARGIN_MM * inch_per_mm * PNG_DPI))

canvas_img = Image.new("RGB", (page_w_px, page_h_px), "white")

# PIL 座標原點在「左上」，所以四角位置要換算
positions_px = {
    id3: (margin_px, margin_px),  # TL
    id2: (page_w_px - margin_px - marker_px, margin_px),  # TR
    id0: (margin_px, page_h_px - margin_px - marker_px),  # BL
    id1: (page_w_px - margin_px - marker_px, page_h_px - margin_px - marker_px),  # BR
}

for mid in CORNER_IDS:
    bio = marker_png_bytes(mid, MARKER_RENDER_PX)
    marker = Image.open(bio).convert("RGB").resize((marker_px, marker_px), Image.NEAREST)
    canvas_img.paste(marker, positions_px[mid])

canvas_img.save(OUT_PNG)
print("Saved PNG:", OUT_PNG)

print("\n提醒：列印請用 PDF 並選 100% 原尺寸（不要 fit-to-page）。")
