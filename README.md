# RoomSpotter: 基於混合架構與相對座標的房間髒污辨識系統

## 1. 專案簡介
本專案旨在解決環境中細微髒污（如頭髮、灰塵）難以辨識的問題。系統在場景中設置 **ArUco 定位標記**，使得拍攝角度、距離或影像尺寸不同時，仍能將影像對齊到同一個標準平面（俯視圖）。  
在完成對齊後，以「基準影像 baseline」與「當前影像 current」做差異檢測，鎖定 ROI（變動區域），後續可將 ROI patches 丟入 **YOLO** 進行髒污/誤報分類。

---

## 2. 測試環境部署 (PoC Demo)
為了驗證系統邏輯，我們先用「A4 定位底紙」快速模擬房間場景：

1. **定位點設計**：生成一張 A4 白底定位圖，四角放置 ArUco 標籤（ID 0~3）。
2. **拍攝規則**：  
   - `baseline.png`：空白乾淨底紙（無物件）  
   - `current.png`：在底紙上放置頭髮、衛生紙團、飲料罐等物件後拍攝  
3. **功能**：即使 current 角度歪斜、距離不同、解析度不同，仍可透過四角 ArUco 完成透視校正，將 baseline/current 映射到同一張「標準俯視圖」後再比對。

> 註：正式房間場景可把定位貼紙貼在牆面四角；PoC 階段用 A4 白紙是為了快速驗證整條流程與相對座標定位。

---

## 3. 系統核心流程 (Workflow)

### Step 1: 影像對齊與校正 (Alignment)
- **標籤偵測**：偵測 4 個 ArUco marker，取得各 marker 的像素角點位置。
- **透視變換**：以 marker 中心點為對應點，計算 Homography（`cv2.getPerspectiveTransform`），將傾斜照片轉成「標準俯視圖」。
- **歸一化**：將 baseline 與 current 皆轉成固定解析度（`WARP_W × WARP_H`），確保像素尺度一致後再做差異分析。

輸出關鍵圖：
- `debug_detect_baseline.png` / `debug_detect_current.png`：檢查四角 marker 是否都被辨識到、ID 是否正確
- `warped_baseline.png` / `warped_current.png`：校正後的標準俯視圖（後續所有 diff 都以此為基準）

---

### Step 2: 差異檢測與 ROI 提取 (Diff Engine)
本版本採用「**intensity diff + edge diff**」混合策略，改善單根頭髮、白對白物件（如衛生紙團）難以檢出的問題。

- **Intensity Diff（大變化）**：`diff_intensity = absdiff(gray_baseline, gray_current)`  
  用來抓：飲料罐、物件整塊出現、明顯陰影邊界等。
- **Edge Diff（細線/輪廓）**：Sobel 梯度差 + 對比增強後用 Canny 擷取弱邊緣  
  用來抓：單根頭髮、細碎屑、低對比輪廓。
- **Mask 合併**：`mask = mask_obj OR mask_edge`  
  並只做 **小型 close** 連通，不做 open（避免把細線吃掉）。
- **ROI 擷取**：對 `diff_mask` 做連通域/輪廓分析，輸出 ROI bounding boxes；同時支援「大面積物件」與「細長物件（頭髮）」保留規則。

輸出關鍵圖（用來 debug 效果非常重要）：
- `diff_intensity.png`：亮度差分圖（大物件訊號）
- `diff_edge.png`：梯度差分圖（細線輪廓訊號）
- `diff_edge_boost.png`：梯度差分對比增強後結果
- `mask_edge_canny.png`：由 diff_edge_boost 產生的 Canny 邊緣遮罩
- `mask_edge_dilated.png`：邊緣遮罩膨脹後（讓細線更連通）
- `mask_obj_raw.png`：intensity 產生的物件遮罩
- `mask_merged_raw.png`：合併後的原始遮罩
- `diff_mask.png`：最終遮罩（拿來找 ROI）

---

### Step 3: YOLO 分類器判定 (AI Classification)
本次已完成「ROI patches → YOLO（COCO 預訓練）粗辨識」整合，用來示範：
- **常見物件**（例如 bottle / cup / cell phone / book / keyboard）可被 YOLO 正確分類
- **髒污類**（頭髮、灰塵、污漬）通常不在 COCO 類別中，會顯示 `unknown`，但仍可透過 diff/ROI 被穩定抓出（後續可再自建髒污資料訓練）

輸出：
- `out/patches/roi_XXX.png`：每個 ROI 的裁切 patch
- `out/roi_predictions_yolo.json`：每個 patch 的 top-1 label/conf
- `out/yolo_test/*.png`：patch 視覺化（含 top-1 框與 label）

---

### Step 4: 結果輸出與定位 (Output & Localization)
- **結果框選**：在 `warped_current` 上畫出 ROI 框並標註資訊（含 mm 座標）
- **mask 疊圖**：將 `diff_mask` 以半透明方式疊在 current，快速肉眼驗證抓取區域
- **相對座標（mm）**：把 ROI 中心點從 warp 像素座標換算回 A4 的 mm 座標（可延伸到 cm、或對應房間平面）
- **YOLO 標籤回寫**：將 YOLO 預測結果 attach 回 ROI，輸出「含 YOLO label」的最終框選圖

輸出檔：
- `result_rois_on_current.png`：ROI 框選結果圖（含中心點 mm）
- `result_mask_overlay.png`：diff_mask 疊圖（最直覺看抓到哪）
- `result_rois_with_yolo.png`：ROI 框選結果圖（含 YOLO label/conf）
- `patches/roi_XXX.png`：每個 ROI 的裁切 patch
- `roi_predictions_yolo.json`：每個 ROI patch 的 YOLO top-1 預測
- `rois.txt`：每個 ROI 的座標、面積、中心點 mm（後續可做統計或餵下一階段模型）

---

## 4. 檔案結構與描述 (Files)

建議專案最小結構如下：

```text
RoomSpotter/
├─ locator/
│  ├─ baseline.png                # 基準圖（乾淨底紙）
│  ├─ current.png                 # 當前圖（有髒污/物件）
│  └─ out/                        # 程式輸出（自動生成）
│     ├─ debug_detect_baseline.png
│     ├─ debug_detect_current.png
│     ├─ warped_baseline.png
│     ├─ warped_current.png
│     ├─ diff_intensity.png
│     ├─ diff_edge.png
│     ├─ diff_edge_boost.png
│     ├─ mask_obj_raw.png
│     ├─ mask_edge_canny.png
│     ├─ mask_edge_dilated.png
│     ├─ mask_merged_raw.png
│     ├─ diff_mask.png
│     ├─ result_rois_on_current.png
│     ├─ result_mask_overlay.png
│     ├─ result_rois_with_yolo.png
│     ├─ rois.txt
│     ├─ roi_predictions_yolo.json
│     ├─ yolo_test/
│     │  ├─ roi_001.png
│     │  └─ ...
│     └─ patches/
│        ├─ roi_001.png
│        └─ ...
├─ generate_locator_sheet.py      # 產生 A4 定位底紙（四角 ArUco）
├─ compare_diff.py                # baseline/current 對齊 + diff + ROI + patches +（可選）YOLO 整合
├─ recognition.py                        # YOLO patch 辨識模組（可被 compare_diff import）
└─ yolov8n.pt                     # COCO 預訓練權重（Ultralytics）



```

---

## 5. 技術架構 (Technology Stack)

| 類別 | 使用技術 |
| :--- | :--- |
| 程式語言 | Python 3.10+ |
| 影像處理 | OpenCV（ArUco、warpPerspective、absdiff、Sobel、Canny） |
| 深度學習 | YOLOv8 / YOLOv10（本 PoC 先用 COCO 預訓練做粗分類；後續可再訓練髒污專用模型） |
| 輔助工具 | NumPy、（可選）SAHI（切片推論） |

---

## 6. 快速啟動 (Quick Start)

### 6.1 安裝環境

> 建議用 requirements.txt（版本一致、最穩）

```bash
pip install -r requirements.txt
```

### 6.2 生成 A4 定位底紙

```bash
python generate_locator.py
```

### 6.3 準備 baseline / current
把你拍的兩張照片放進：
- `locator/baseline.png`
- `locator/current.png`

> 建議：拍攝時四角 ArUco 都要入鏡且清晰，current 可以歪、遠近不同、解析度不同都沒關係，因為會先做透視校正。

### 6.4 執行比對 + ROI + patches + YOLO
```bash
python compare_diff.py
```