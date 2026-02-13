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
# ArUco / Warp 設定（不再依賴 A4）
# =========================
DICT_NAME = "DICT_4X4_50"
CORNER_IDS = [0, 1, 2, 3]  # [BL, BR, TR, TL]

# canonical 輸出寬度（px）
WARP_W = 2000

# canonical 邊界（px）：代表你希望 marker center 不要貼到邊
MARGIN_PX = 140

# 如果你想固定輸出高度（不 auto aspect），把 AUTO_ASPECT=False，並指定 WARP_H_FIXED
AUTO_ASPECT = True
WARP_H_FIXED = 2800

# =========================
# Diff / Mask / ROI 參數
# =========================
USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

GAUSS_BLUR = 3          # 頭髮不要大模糊；0 or 3
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

def estimate_aspect_from_src(src_4x2):
    """
    src order: [BL, BR, TR, TL]
    用兩條寬、兩條高平均估算長寬比
    """
    BL, BR, TR, TL = src_4x2
    w1 = np.linalg.norm(BR - BL)
    w2 = np.linalg.norm(TR - TL)
    h1 = np.linalg.norm(TL - BL)
    h2 = np.linalg.norm(TR - BR)
    w = (w1 + w2) / 2.0
    h = (h1 + h2) / 2.0
    if w <= 1:
        return 1.0
    return float(h / w)

def build_dst_points(warp_w, warp_h):
    """
    canonical 平面上：四個 marker center 的目標位置（px）
    dst order 對應 src：[BL, BR, TR, TL]
    """
    xL = MARGIN_PX
    xR = warp_w - MARGIN_PX
    yB = warp_h - MARGIN_PX
    yT = MARGIN_PX

    dst = np.array([
        [xL, yB],  # BL
        [xR, yB],  # BR
        [xR, yT],  # TR
        [xL, yT],  # TL
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
        cv2.putText(dbg, f"ID{mid}", (ctr[0] + 5, ctr[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(OUT_DIR, f"debug_detect_{tag}.png"), dbg)

    src = order_src_points(found)
    if src is None:
        missing = [mid for mid in CORNER_IDS if mid not in found]
        raise RuntimeError(f"[{tag}] 缺少 ArUco ID：{missing}，請確保四角都拍到且清晰。")

    # 自動估 aspect ratio（不再需要 A4 mm）
    if AUTO_ASPECT:
        aspect = estimate_aspect_from_src(src)   # h/w
        warp_w = int(WARP_W)
        warp_h = int(round(WARP_W * aspect))
        warp_h = max(800, warp_h)
    else:
        warp_w = int(WARP_W)
        warp_h = int(WARP_H_FIXED)

    dst = build_dst_points(warp_w, warp_h)

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, H, (warp_w, warp_h), flags=cv2.INTER_LINEAR)

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

def px_to_norm(x_px, y_px, W, H):
    """canonical 平面像素 -> normalized 座標 (0~1)，原點左下"""
    nx = float(x_px) / float(max(1, W))
    ny = 1.0 - (float(y_px) / float(max(1, H)))
    return nx, ny

def diff_and_rois(base_warp, cur_warp):
    # 取得尺寸
    Hh, Ww = cur_warp.shape[:2]

    # 轉灰階
    base_g = cv2.cvtColor(base_warp, cv2.COLOR_BGR2GRAY)
    cur_g  = cv2.cvtColor(cur_warp,  cv2.COLOR_BGR2GRAY)

    # CLAHE 增強對比 (保留原本設定)
    if USE_CLAHE:
        base_g = clahe_gray(base_g)
        cur_g  = clahe_gray(cur_g)

    # 高斯模糊 (建議開啟，例如 GAUSS_BLUR=3，以減少噪點)
    if GAUSS_BLUR and GAUSS_BLUR > 0:
        k = GAUSS_BLUR if GAUSS_BLUR % 2 == 1 else GAUSS_BLUR + 1
        base_g = cv2.GaussianBlur(base_g, (k, k), 0)
        cur_g  = cv2.GaussianBlur(cur_g,  (k, k), 0)

    # 1. 計算差異圖 (Intensity Diff)
    diff_i = cv2.absdiff(base_g, cur_g)
    cv2.imwrite(os.path.join(OUT_DIR, "diff_intensity.png"), diff_i)

    # 2. 計算邊緣差異 (Edge Diff)
    base_e = sobel_mag(base_g)
    cur_e  = sobel_mag(cur_g)
    diff_e = cv2.absdiff(base_e, cur_e)
    cv2.imwrite(os.path.join(OUT_DIR, "diff_edge.png"), diff_e)

    # --- A) Intensity Mask (抓大塊顏色變化) ---
    vals = diff_i[diff_i > 0]
    # [修正] 提高最低門檻至 25 (原本 8 太敏感)，避免背景雜訊
    t_obj = int(max(25, np.percentile(vals, 92))) if vals.size > 50 else 25
    mask_obj = (diff_i >= t_obj).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUT_DIR, "mask_obj_raw.png"), mask_obj)

    # --- B) Edge Mask (抓輪廓變化) ---
    # [修正核心] 放棄 Canny，改用 Threshold 硬閥值
    # 邏輯：只有當邊緣差異強度 > 30 時，才視為有效變化
    # 如果雜訊還是多，可以試著把 30 調高到 40 或 50
    EDGE_THRESHOLD = 30
    _, mask_edge = cv2.threshold(diff_e, EDGE_THRESHOLD, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(OUT_DIR, "mask_edge_threshold.png"), mask_edge)

    # 稍微膨脹一點點，讓斷掉的線條連起來，但不要太大
    k_merge = 3
    kernel_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_merge, k_merge))
    mask_edge = cv2.dilate(mask_edge, kernel_merge, iterations=1)
    cv2.imwrite(os.path.join(OUT_DIR, "mask_edge_dilated.png"), mask_edge)

    # --- Merged Mask (合併) ---
    mask = cv2.bitwise_or(mask_obj, mask_edge)
    cv2.imwrite(os.path.join(OUT_DIR, "mask_merged_raw.png"), mask)

    # --- C) Morphology 清理 (防止擴散的關鍵步驟) ---
    
    # [關鍵步驟 1] Open (開運算)：先侵蝕再膨脹，用來「吃掉」孤立的白點 (噪點)
    # 這步能切斷雜訊之間的連結，防止它們在下一步被連成大方塊
    k_open = 3  # 3x3 或 5x5
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # [關鍵步驟 2] Close (閉運算)：先膨脹再侵蝕，用來「填滿」物體內部的空洞
    # 這裡可以用大一點的核，把真正的物體連起來
    k_close = 9 # 9x9 確保物體完整
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    cv2.imwrite(os.path.join(OUT_DIR, "diff_mask.png"), mask)

    # --- D) Contours & ROIs ---
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

        # 篩選條件 (保持原本邏輯)
        keep_big = area >= MIN_AREA_PX
        keep_hair = (perim >= MIN_PERIM_PX and long_side >= MIN_LEN_PX and 
                     aspect >= MIN_ASPECT and extent <= MAX_EXTENT)

        if not (keep_big or keep_hair):
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        cx_n, cy_n = px_to_norm(cx, cy, Ww, Hh)

        rois.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": float(area),
            "perim": float(perim),
            "aspect": float(aspect),
            "extent": float(extent),
            "cx_px": float(cx), "cy_px": float(cy),
            "cx_n": float(cx_n), "cy_n": float(cy_n),
        })

    rois.sort(key=lambda r: r["area"], reverse=True)
    return rois, mask    # ... (前面的 code: 轉灰階, CLAHE, GaussianBlur, diff_i, diff_e 都保持不變) ...

    # --- A) intensity mask（抓大變化）---
    vals = diff_i[diff_i > 0]
    # [修改] 提高最低門檻到 25，防止光線微變導致整張圖被選取
    t_obj = int(max(25, np.percentile(vals, 92))) if vals.size > 50 else 25
    mask_obj = (diff_i >= t_obj).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUT_DIR, "mask_obj_raw.png"), mask_obj)

    # --- B) edge mask（抓細線：頭髮）---
    # [修改核心] 放棄 Canny，改用 Threshold 硬閥值
    # 邏輯：只有當「邊緣差異」大於 30 (0~255) 時，才算作是物體
    # 這樣可以過濾掉背景那些微小的雜訊 (通常 < 10)
    
    EDGE_THRESHOLD = 30  # 如果還是太多雜訊，可以調高這個值 (例如 40 或 50)
    _, mask_edge = cv2.threshold(diff_e, EDGE_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(os.path.join(OUT_DIR, "mask_edge_threshold.png"), mask_edge)

    # [修改] 稍微膨脹一點點，讓斷掉的線連起來，但不要像之前那麼大
    # 使用 3x3 的核即可
    kernel_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_edge = cv2.dilate(mask_edge, kernel_merge, iterations=1)
    
    cv2.imwrite(os.path.join(OUT_DIR, "mask_edge_dilated.png"), mask_edge)

    # --- Merged mask ---
    mask = cv2.bitwise_or(mask_obj, mask_edge)
    cv2.imwrite(os.path.join(OUT_DIR, "mask_merged_raw.png"), mask)

    # === Step 1: 先 Open 去除剩下的孤立噪點 ===
    # 這裡非常重要，用 3x3 或 5x5 的核把小白點吃掉
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # === Step 2: 再 Close 把物體內部補滿 ===
    # 這裡可以用大一點的核，把物體連起來
    k2 = 9  # 稍微加大一點，確保物體完整
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    cv2.imwrite(os.path.join(OUT_DIR, "diff_mask.png"), mask)

    # ... (後面的 contours -> rois 保持不變) ...    Hh, Ww = cur_warp.shape[:2]

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
    t_obj = int(max(20, np.percentile(vals, 92))) if vals.size > 50 else 8
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

    # === [新增] Step 1: 先用 Open 把細碎雜訊吃掉 ===
    # 定義一個稍微小一點的核給 Open 用
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # === [原本] Step 2: 再用 Close 把物體內部補洞 ===
    # small close（連起來）不要 open -> 這裡原本的註解
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

        cx = x + w / 2.0
        cy = y + h / 2.0
        cx_n, cy_n = px_to_norm(cx, cy, Ww, Hh)

        rois.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": float(area),
            "perim": float(perim),
            "aspect": float(aspect),
            "extent": float(extent),
            "cx_px": float(cx), "cy_px": float(cy),
            "cx_n": float(cx_n), "cy_n": float(cy_n),   # normalized(0~1)
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
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)

        yolo = r.get("yolo", None)
        if yolo:
            y_txt = f"{yolo['label']} {yolo['conf']:.2f}"
        else:
            y_txt = "no-yolo"

        label = (
            f"ROI{i} {y_txt} "
            f"A={int(r['area'])} AR={r['aspect']:.1f} "
            f"(n={r['cx_n']:.3f},{r['cy_n']:.3f})"
        )
        cv2.putText(out, label, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return out

def overlay_mask(img_bgr, mask_u8):
    overlay = img_bgr.copy()
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = 255
    m = (mask_u8 > 0)[:, :, None]
    overlay = np.where(m, (overlay * (1 - OVERLAY_ALPHA) + red * OVERLAY_ALPHA).astype(np.uint8), overlay)
    return overlay

def save_patches(img_bgr, rois):
    for i, r in enumerate(rois, 1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        patch = img_bgr[y:y + h, x:x + w].copy()
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

    # 2) mask 疊圖
    result_overlay = overlay_mask(cur_warp, mask)
    cv2.imwrite(os.path.join(OUT_DIR, "result_mask_overlay.png"), result_overlay)

    # 3) 裁切 patches
    save_patches(cur_warp, rois)

    # 4) YOLO 粗辨識（COCO 預訓練）
    yolo_weights = "yolov8n.pt"
    recog = recognition.YoloPatchRecognizer(weights=yolo_weights, imgsz=320, conf=0.25, iou=0.45)

    rois, preds = recog.attach_to_rois(rois, PATCH_DIR)
    recog.save_json(preds, os.path.join(OUT_DIR, "roi_predictions_yolo.json"))
    recog.visualize_many(PATCH_DIR, preds, os.path.join(OUT_DIR, "yolo_test"))

    # 5) 重新輸出一張「含 YOLO label」的框選圖
    result_boxes_yolo = draw_rois(cur_warp, rois)
    cv2.imwrite(os.path.join(OUT_DIR, "result_rois_with_yolo.png"), result_boxes_yolo)

    # 6) ROI 文字輸出（改成 normalized 座標）
    txt_path = os.path.join(OUT_DIR, "rois.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rois, 1):
            f.write(
                f"ROI{i}\txywh=({r['x']},{r['y']},{r['w']},{r['h']})\t"
                f"area={r['area']:.1f}\taspect={r['aspect']:.2f}\t"
                f"center_norm=({r['cx_n']:.4f},{r['cy_n']:.4f})\n"
            )

    print("Done. outputs in:", OUT_DIR)
    print(" - debug_detect_baseline.png / debug_detect_current.png")
    print(" - warped_baseline.png / warped_current.png")
    print(" - result_rois_on_current.png")
    print(" - result_mask_overlay.png")
    print(" - patches/roi_XXX.png")
    print(" - rois.txt")
    print(" - roi_predictions_yolo.json")
    print(" - result_rois_with_yolo.png")

if __name__ == "__main__":
    main()
