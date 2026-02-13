# RoomSpotter: 基於混合架構與相對座標的房間髒污辨識系統

## 1. 專案簡介
本專案旨在解決環境中細微髒污（如頭髮、衛生紙團）難以辨識的問題。系統在場景中設置 **ArUco 定位標記**，使得拍攝角度、距離或影像尺寸不同時，仍能將影像對齊到同一個標準平面（俯視圖）。  
在完成對齊後，以「基準影像 baseline」與「當前影像 current」做差異檢測，鎖定 ROI（變動區域），並透過 **YOLO** 進行髒污/誤報分類。

---

## 2. 測試環境部署 (PoC Demo)
為了驗證系統邏輯，我們先用「A4 定位底紙」快速模擬房間場景：

1. **定位點設計**：生成一張 A4 白底定位圖，四角放置 ArUco 標籤（ID 0~3）。
2. **拍攝規則**：  
   - `baseline.png`：空白乾淨底紙（無物件）  
   - `current.png`：在底紙上放置頭髮、衛生紙團、飲料罐等物件後拍攝  
3. **功能**：即使 current 角度歪斜、距離不同、解析度不同，仍可透過四角 ArUco 完成透視校正，將 baseline/current 映射到同一張「標準俯視圖」後再比對。

> **註**：正式房間場景可把定位貼紙貼在牆面四角；PoC 階段用 A4 白紙是為了快速驗證整條流程與相對座標定位。

---

## 3. 系統核心流程 (Workflow)

### Step 1: 影像對齊與校正 (Alignment)
- **標籤偵測**：偵測 4 個 ArUco marker，取得各 marker 的像素角點位置。
- **透視變換**：以 marker 中心點為對應點，計算 Homography（`cv2.getPerspectiveTransform`），將傾斜照片轉成「標準俯視圖」。
- **歸一化**：將 baseline 與 current 皆轉成固定解析度（`WARP_W × WARP_H`），確保像素尺度一致後再做差異分析。

**輸出關鍵圖：**
- `debug_detect_*.png`：檢查四角 marker 辨識狀況
- `warped_*.png`：校正後的標準俯視圖（後續所有 diff 都以此為基準）

---

### Step 2: 差異檢測與 ROI 提取 (Diff Engine)
本版本採用「**Intensity Diff + Edge Gradient**」混合策略，並針對噪點與破碎物件進行優化。

1. **Intensity Diff（大變化）**：
   - 計算 `absdiff(gray_baseline, gray_current)`。
   - 用於捕捉：飲料罐、大面積陰影、實體物件。
   - 設定較高的門檻（Threshold=25）以過濾光線微變。

2. **Edge Diff（細線/輪廓）**：
   - 使用 **Sobel** 計算梯度強度，取差值。
   - **關鍵優化**：放棄 Canny 邊緣檢測，改用 **Hard Thresholding (硬閥值)**。避免背景噪點被 Canny 演算法過度放大與擴散。
   - 用於捕捉：單根頭髮、細碎屑、低對比輪廓。

3. **Mask 優化 (Morphology)**：
   - **Open (開運算)**：先侵蝕再膨脹，切斷並移除孤立的噪點（防止雜訊擴散）。
   - **Close (閉運算)**：先膨脹再侵蝕，填滿物體內部的空洞。

4. **ROI 提取與合併 (Merging)**：
   - **方框合併 (Box Merging)**：針對表面紋理複雜的物件（如衛生紙團），系統會檢測重疊的 ROI，並執行「大框吃小框」的合併邏輯，避免單一物件被切分成多個破碎框。

**輸出關鍵圖（Debug 用）：**
- `diff_intensity.png`：亮度差分圖
- `diff_edge.png`：梯度差分圖
- `mask_obj_raw.png`：Intensity 產生的物件遮罩
- `mask_edge_threshold.png`：Edge 產生的閾值遮罩（取代原本的 Canny）
- `mask_merged_raw.png`：合併後的原始遮罩
- `diff_mask.png`：經過形態學清理後的最終遮罩

---

### Step 3: YOLO 分類器判定 (AI Classification)
完成 ROI 裁切後，將 Patches 傳入 **YOLO**（COCO 預訓練）進行粗辨識：
- **常見物件**（bottle, cup, phone）：可被 YOLO 正確分類。
- **髒污類**（頭髮、灰塵）：通常顯示 `unknown` 或低置信度，但仍透過 Diff 演算法被穩定框出（後續可自建資料集訓練）。

---

### Step 4: 結果輸出與定位 (Output & Localization)
- **結果框選**：在 `warped_current` 上畫出 ROI 框。
- **mask 疊圖**：將 `diff_mask` 半透明疊加，直觀驗證抓取區域。
- **座標換算**：將像素座標換算回 Normalized 座標（未來可對應真實世界 mm）。
- **YOLO 標籤回寫**：將辨識結果 attach 回 ROI。

**輸出檔：**
- `result_rois_on_current.png`：ROI 框選結果圖
- `result_mask_overlay.png`：Diff mask 疊圖
- `result_rois_with_yolo.png`：含 YOLO label 的最終結果

---

## 4. 檔案結構 (File Structure)

```text
RoomSpotter/
├─ locator/
│  ├─ baseline.png                # 基準圖
│  ├─ current.png                 # 當前圖
│  └─ out/                        # 自動生成結果
│     ├─ diff_intensity.png
│     ├─ diff_edge.png
│     ├─ mask_obj_raw.png
│     ├─ mask_edge_threshold.png  # (新) 閾值邊緣遮罩
│     ├─ diff_mask.png            # 最終遮罩
│     ├─ result_rois_on_current.png
│     ├─ result_mask_overlay.png
│     ├─ result_rois_with_yolo.png
│     ├─ rois.txt
│     ├─ roi_predictions_yolo.json
│     └─ patches/                 # 裁切出的 ROI 圖像
├─ generate_locator.py            # 產生定位底紙
├─ compare_diff.py                # 主程式：對齊 + Diff + ROI Merge + YOLO
├─ recognition.py                 # YOLO 辨識模組
└─ yolov8n.pt                     # YOLO 權重            # COCO 預訓練權重（Ultralytics）



```

---

## 5. 技術架構 (Technology Stack)

| 類別 | 使用技術 | 關鍵演算法 |
| :--- | :--- | :--- |
| **程式語言** | Python 3.10+ | - |
| **影像對齊** | OpenCV | ArUco Markers, Perspective Transform (Homography) |
| **差異檢測** | OpenCV | AbsDiff, Sobel Gradient, **Thresholding**, Morphology (Open/Close) |
| **後處理** | Custom Logic | **Box Merging (NMS-like logic)**, Contour Analysis |
| **物件識別** | YOLO / Torch | YOLOv8 (Inference on Patches) |

---

## 6. 快速啟動 (Quick Start)

### 6.1 安裝依賴
建議建立一個乾淨的 Python 虛擬環境後執行：

```bash
pip install opencv-python-headless numpy ultralytics
```

### 6.2 生成 A4 定位底紙 或 貼紙

```bash
python generate_locator.py

python generate_stickers.py #貼紙

```

### 6.3 準備 baseline / current
請確保專案目錄下的 locator/ 資料夾中包含以下兩張圖片：

- `locator/baseline.png`
- `locator/current.png`

> 建議：拍攝時四角 ArUco 都要入鏡且清晰，current 可以歪、遠近不同、解析度不同都沒關係，因為會先做透視校正。

### 6.4 執行比對 + ROI + patches + YOLO
```bash
python compare_diff.py
```
> 程式執行完畢後，請至 locator/out/ 資料夾查看生成的標註結果圖與 Report。