#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import cv2
import numpy as np
from ultralytics import YOLO


def list_patches(patch_dir: str, exts=(".png", ".jpg", ".jpeg")):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(patch_dir, f"*{ext}")))
    return sorted(paths)


class YoloPatchRecognizer:
    """
    用 COCO 預訓練 YOLO(detect) 對 ROI patches 做粗辨識：
    - 有偵測：回傳 top-1 label/conf
    - 無偵測：label=unknown, conf=0
    """

    def __init__(self, weights: str, imgsz: int = 320, conf: float = 0.25, iou: float = 0.45):
        self.weights = weights
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.model = YOLO(self.weights)
        self.names = self.model.names

    def predict_one(self, patch_path: str):
        results = self.model.predict(
            source=patch_path,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )
        res = results[0]

        pred = {
            "patch": os.path.basename(patch_path),
            "label": "unknown",
            "conf": 0.0,
            "has_box": False,
            "box_xyxy": None
        }

        if res.boxes is not None and len(res.boxes) > 0:
            confs = res.boxes.conf.cpu().numpy()
            clss  = res.boxes.cls.cpu().numpy().astype(int)
            xyxy  = res.boxes.xyxy.cpu().numpy()

            best = int(np.argmax(confs))
            pred["conf"] = float(confs[best])
            cls_id = int(clss[best])
            pred["label"] = str(self.names.get(cls_id, f"class_{cls_id}"))
            pred["has_box"] = True
            pred["box_xyxy"] = xyxy[best].astype(int).tolist()

        return pred

    def attach_to_rois(self, rois, patch_dir: str, name_fmt: str = "roi_{i:03d}.png"):
        """
        對每個 ROI 對應的 patch 做 YOLO，結果掛回 roi["yolo"]。
        回傳 (rois, preds_list)
        """
        preds = []
        for i, r in enumerate(rois, 1):
            patch_name = name_fmt.format(i=i)
            patch_path = os.path.join(patch_dir, patch_name)

            if os.path.exists(patch_path):
                p = self.predict_one(patch_path)
            else:
                p = {"patch": patch_name, "label": "unknown", "conf": 0.0, "has_box": False, "box_xyxy": None}

            r["yolo"] = {"label": p["label"], "conf": p["conf"]}
            preds.append(p)

        return rois, preds

    @staticmethod
    def save_json(preds, out_json_path: str):
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)

    @staticmethod
    def visualize_many(patch_dir: str, preds, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        pred_map = {p["patch"]: p for p in preds}

        for pth in list_patches(patch_dir):
            bn = os.path.basename(pth)
            pred = pred_map.get(bn, {"label": "unknown", "conf": 0.0, "has_box": False, "box_xyxy": None})

            img = cv2.imread(pth)
            if img is None:
                continue

            if pred["has_box"] and pred["box_xyxy"] is not None:
                x1, y1, x2, y2 = pred["box_xyxy"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"{pred['label']} {pred['conf']:.2f}", (x1, max(0, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                cv2.putText(img, "unknown", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            cv2.imwrite(os.path.join(out_dir, bn), img)
