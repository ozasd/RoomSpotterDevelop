#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import recognition
# =========================
# I/O
# =========================
IN_DIR = "locator"
BASELINE_NAME = "baseline.png"
CURRENT_NAME  = "current.png"

OUT_DIR = os.path.join(IN_DIR, "out")
PATCH_DIR = os.path.join(OUT_DIR, "patches")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PATCH_DIR, exist_ok=True)

# =========================
# ArUco / Warp 設定
# =========================
DICT_NAME = "DICT_4X4_50"
CORNER_IDS = [0, 1, 2, 3]  # [ID0, ID1, ID2, ID3] = [BL, BR, TR, TL]

# A4 + 你生成底紙的物理規格（mm）
MARKER_SIZE_MM = 40.0
MARGIN_MM = 15.0
A4_W_MM = 210.0
A4_H_MM = 297.0

# 透視校正後輸出解析度：抓單根頭髮建議拉高
WARP_W = 2000
WARP_H = int(round(WARP_W * (A4_H_MM / A4_W_MM)))

# =========================
# Diff / Mask / ROI 參數
# =========================
USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

GAUSS_BLUR = 0          # 頭髮不要大模糊；0 or 3
MORPH_K = 3             # 小核 close（連起來）
MERGE_DILATE_K = 3      # Canny 細線膨脹，幫助連通

# ROI 篩選：大物件 or 細長物件（頭髮）
MIN_AREA_PX = 80
MIN_PERIM_PX = 160
MIN_LEN_PX = 45
MIN_ASPECT = 3.0
MAX_EXTENT = 0.45

# mask overlay 透明度
OVERLAY_ALPHA = 0.45

# =========================
# ArUco 偵測 / warp
# =========================
def detect_aruco_corners(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, DICT_NAME))
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    found = {}
    if ids is None:
        return found
    for c, i in zip(corners, ids.flatten()):
        found[int(i)] = c.reshape(4, 2).astype(np.float32)
    return found

def marker_center(corners_4x2):
    return corners_4x2.mean(axis=0)

def order_src_points(found_markers):
    pts = []
    for mid in CORNER_IDS:
        if mid not in found_markers:
            return None
        pts.append(marker_center(found_markers[mid]))
    return np.array(pts, dtype=np.float32)

def build_dst_points():
    # marker center 的理論位置（mm）
    xL = MARGIN_MM + MARKER_SIZE_MM / 2.0
    xR = A4_W_MM - MARGIN_MM - MARKER_SIZE_MM / 2.0
    yB = MARGIN_MM + MARKER_SIZE_MM / 2.0
    yT = A4_H_MM - MARGIN_MM - MARKER_SIZE_MM / 2.0

    # mm -> warp(px)
    def mm_to_px(x_mm, y_mm):
        x = (x_mm / A4_W_MM) * WARP_W
        y = (1.0 - (y_mm / A4_H_MM)) * WARP_H  # 影像 y 向下，做反轉
        return [x, y]

    # 順序對應 src：[BL, BR, TR, TL]
    dst = np.array([
        mm_to_px(xL, yB),  # ID0 BL
        mm_to_px(xR, yB),  # ID1 BR
        mm_to_px(xR, yT),  # ID2 TR
        mm_to_px(xL, yT),  # ID3 TL
    ], dtype=np.float32)
    return dst

def warp_to_canonical(img_bgr, tag="img"):
    found = detect_aruco_corners(img_bgr)

    # debug：畫出偵測結果
    dbg = img_bgr.copy()
    for mid, c in found.items():
        c_int = c.astype(int)
        cv2.polylines(dbg, [c_int], True, (0, 255, 0), 2)
        ctr = marker_center(c).astype(int)
        cv2.putText(dbg, f"ID{mid}", (ctr[0]+5, ctr[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imwrite(os.path.join(OUT_DIR, f"debug_detect_{tag}.png"), dbg)

    src = order_src_points(found)
    if src is None:
        missing = [mid for mid in CORNER_IDS if mid not in found]
        raise RuntimeError(f"[{tag}] 缺少 ArUco ID：{missing}，請確保四角都拍到且清晰。")

    dst = build_dst_points()
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, H, (WARP_W, WARP_H), flags=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(OUT_DIR, f"warped_{tag}.png"), warped)
    return warped

# =========================
# Diff engine
# =========================
def clahe_gray(g):
    cla = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return cla.apply(g)

def sobel_mag(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag

def auto_canny(img_u8, sigma=0.33):
    vals = img_u8[img_u8 > 0]
    v = float(np.median(vals)) if vals.size > 0 else float(np.median(img_u8))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img_u8, lower, upper)

def px_to_mm(x_px, y_px):
    """warp 平面上的像素 -> A4 mm 座標（原點在左下）"""
    x_mm = (x_px / WARP_W) * A4_W_MM
    y_mm = (1.0 - (y_px / WARP_H)) * A4_H_MM
    return x_mm, y_mm

def diff_and_rois(base_warp, cur_warp):
    base_g = cv2.cvtColor(base_warp, cv2.COLOR_BGR2GRAY)
    cur_g  = cv2.cvtColor(cur_warp,  cv2.COLOR_BGR2GRAY)

    if USE_CLAHE:
        base_g = clahe_gray(base_g)
        cur_g  = clahe_gray(cur_g)

    if GAUSS_BLUR and GAUSS_BLUR > 0:
        k = GAUSS_BLUR if GAUSS_BLUR % 2 == 1 else GAUSS_BLUR + 1
        base_g = cv2.GaussianBlur(base_g, (k, k), 0)
        cur_g  = cv2.GaussianBlur(cur_g,  (k, k), 0)

    # intensity diff
    diff_i = cv2.absdiff(base_g, cur_g)
    cv2.imwrite(os.path.join(OUT_DIR, "diff_intensity.png"), diff_i)

    # edge diff
    base_e = sobel_mag(base_g)
    cur_e  = sobel_mag(cur_g)
    diff_e = cv2.absdiff(base_e, cur_e)
    cv2.imwrite(os.path.join(OUT_DIR, "diff_edge.png"), diff_e)

    # --- A) intensity mask（抓大變化）---
    vals = diff_i[diff_i > 0]
    t_obj = int(max(8, np.percentile(vals, 92))) if vals.size > 50 else 8
    mask_obj = (diff_i >= t_obj).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUT_DIR, "mask_obj_raw.png"), mask_obj)

    # --- B) edge mask（抓細線：頭髮）---
    diff_e_boost = cv2.convertScaleAbs(diff_e, alpha=2.5, beta=0)  # 2.0~4.0 可調
    cv2.imwrite(os.path.join(OUT_DIR, "diff_edge_boost.png"), diff_e_boost)

    edge = auto_canny(diff_e_boost, sigma=0.33)
    cv2.imwrite(os.path.join(OUT_DIR, "mask_edge_canny.png"), edge)

    k1 = MERGE_DILATE_K if MERGE_DILATE_K % 2 == 1 else MERGE_DILATE_K + 1
    kernel_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    mask_edge = cv2.dilate(edge, kernel_merge, iterations=1)
    cv2.imwrite(os.path.join(OUT_DIR, "mask_edge_dilated.png"), mask_edge)

    # merged mask
    mask = cv2.bitwise_or(mask_obj, mask_edge)
    cv2.imwrite(os.path.join(OUT_DIR, "mask_merged_raw.png"), mask)

    # small close（連起來）不要 open
    k2 = MORPH_K if MORPH_K % 2 == 1 else MORPH_K + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(os.path.join(OUT_DIR, "diff_mask.png"), mask)

    # contours -> rois
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for c in cnts:
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)

        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect = long_side / short_side
        extent = area / float(max(1, w * h))

        keep_big = area >= MIN_AREA_PX
        keep_hair = (perim >= MIN_PERIM_PX and long_side >= MIN_LEN_PX and aspect >= MIN_ASPECT and extent <= MAX_EXTENT)

        if not (keep_big or keep_hair):
            continue

        # 中心點 + mm 座標
        cx = x + w / 2.0
        cy = y + h / 2.0
        cx_mm, cy_mm = px_to_mm(cx, cy)

        rois.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": float(area),
            "perim": float(perim),
            "aspect": float(aspect),
            "extent": float(extent),
            "cx_px": float(cx), "cy_px": float(cy),
            "cx_mm": float(cx_mm), "cy_mm": float(cy_mm),
        })

    rois.sort(key=lambda r: r["area"], reverse=True)
    return rois, mask

# =========================
# Visualization outputs
# =========================
def draw_rois(img_bgr, rois):
    out = img_bgr.copy()
    for i, r in enumerate(rois, 1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 2)

        yolo = r.get("yolo", None)
        if yolo:
            y_txt = f"{yolo['label']} {yolo['conf']:.2f}"
        else:
            y_txt = "no-yolo"

        label = (
            f"ROI{i} {y_txt} "
            f"A={int(r['area'])} AR={r['aspect']:.1f} "
            f"({r['cx_mm']:.1f},{r['cy_mm']:.1f}mm)"
        )
        cv2.putText(out, label, (x, max(0, y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
    return out

def overlay_mask(img_bgr, mask_u8):
    overlay = img_bgr.copy()
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = 255  # 紅色
    m = (mask_u8 > 0)[:, :, None]
    overlay = np.where(m, (overlay*(1-OVERLAY_ALPHA) + red*OVERLAY_ALPHA).astype(np.uint8), overlay)
    return overlay

def save_patches(img_bgr, rois):
    for i, r in enumerate(rois, 1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        patch = img_bgr[y:y+h, x:x+w].copy()
        cv2.imwrite(os.path.join(PATCH_DIR, f"roi_{i:03d}.png"), patch)

def main():
    baseline_path = os.path.join(IN_DIR, BASELINE_NAME)
    current_path  = os.path.join(IN_DIR, CURRENT_NAME)

    base = cv2.imread(baseline_path)
    cur  = cv2.imread(current_path)

    if base is None:
        raise FileNotFoundError(f"找不到或讀不到：{baseline_path}")
    if cur is None:
        raise FileNotFoundError(f"找不到或讀不到：{current_path}")

    base_warp = warp_to_canonical(base, tag="baseline")
    cur_warp  = warp_to_canonical(cur,  tag="current")

    rois, mask = diff_and_rois(base_warp, cur_warp)

    # 1) 畫框結果圖
    result_boxes = draw_rois(cur_warp, rois)
    cv2.imwrite(os.path.join(OUT_DIR, "result_rois_on_current.png"), result_boxes)

    # 2) mask 疊圖（超直覺）
    result_overlay = overlay_mask(cur_warp, mask)
    cv2.imwrite(os.path.join(OUT_DIR, "result_mask_overlay.png"), result_overlay)

    # 3) 裁切 patches（給 YOLO/SAHI 用）
    save_patches(cur_warp, rois)

      # 4) YOLO 粗辨識（COCO 預訓練，不用自己訓練）
    yolo_weights = "yolov8n.pt"  # 你專案根目錄已有這個檔
    recog = recognition.YoloPatchRecognizer(weights=yolo_weights, imgsz=320, conf=0.25, iou=0.45)

    rois, preds = recog.attach_to_rois(rois, PATCH_DIR)
    recog.save_json(preds, os.path.join(OUT_DIR, "roi_predictions_yolo.json"))
    recog.visualize_many(PATCH_DIR, preds, os.path.join(OUT_DIR, "yolo_test"))

    # 5) 重新輸出一張「含 YOLO label」的框選圖
    result_boxes_yolo = draw_rois(cur_warp, rois)
    cv2.imwrite(os.path.join(OUT_DIR, "result_rois_with_yolo.png"), result_boxes_yolo)

    # 4) ROI 文字輸出（方便後處理）
    txt_path = os.path.join(OUT_DIR, "rois.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rois, 1):
            f.write(
                f"ROI{i}\txywh=({r['x']},{r['y']},{r['w']},{r['h']})\t"
                f"area={r['area']:.1f}\taspect={r['aspect']:.2f}\t"
                f"center_mm=({r['cx_mm']:.1f},{r['cy_mm']:.1f})\n"
            )

    print("Done. outputs in:", OUT_DIR)
    print(" - result_rois_on_current.png")
    print(" - result_mask_overlay.png")
    print(" - patches/roi_XXX.png")
    print(" - rois.txt")

if __name__ == "__main__":
    main()
