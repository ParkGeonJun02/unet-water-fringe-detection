import os
import sys
import torch
import numpy as np
import cv2
import rasterio
from tqdm import tqdm
import json
from shapely.geometry import shape
from rasterio.features import rasterize

sys.path.append(os.path.abspath('.'))
from src.model import UNet

def calculate_metrics(preds, targets, smooth=1e-6):
    preds = preds.byte()
    targets = targets.byte()

    tp = (preds & targets).sum().float().item()
    fp = (preds & ~targets).sum().float().item()
    fn = (~preds & targets).sum().float().item()
    tn = (~preds & ~targets).sum().float().item()

    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    total_pixels = tp + fp + fn + tn
    oa = (tp + tn) / total_pixels if total_pixels > 0 else 0.0

    return iou, precision, recall, oa

def compute_heuristic_water(image_rgb):
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean_img = cv2.boxFilter(gray, -1, (31, 31))
    sq_mean  = cv2.boxFilter(gray**2, -1, (31, 31))
    local_std = np.sqrt(np.clip(sq_mean - mean_img**2, 0, None))

    # 물 조건: 어둡고 + 파랑/초록 계열 + 질감 매끄러움
    color_water = (r < 85) & (g < 110) & (b < 110) & (g > r) & (local_std < 5.0)
    color_water = color_water.astype(np.uint8)

    # 모폴로지 노이즈 제거
    kernel = np.ones((9, 9), np.uint8)
    color_water = cv2.morphologyEx(color_water, cv2.MORPH_OPEN, kernel)
    return color_water

def load_pure_json_mask(label_path, height, width, geo_transform):
    water_mask = np.zeros((height, width), dtype=np.uint8)
    if os.path.exists(label_path):
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            features = meta.get("annotation", {}).get("features", [])
            water_shapes = []
            for feat in features:
                geo = feat.get("geometry", {})
                code = str(feat.get("properties", {}).get("CODE", "")).strip()
                # CODE=50: 수역 (물), CODE=20/40/511: 수변초목도 포함
                if code in ["50", "20", "40", "511"] and geo.get("coordinates"):
                    try:
                        poly_obj = shape(geo)
                        water_shapes.append((poly_obj, 1))
                    except Exception:
                        continue
            if water_shapes:
                water_mask = rasterize(
                    water_shapes,
                    out_shape=(height, width),
                    transform=geo_transform,
                    fill=0,
                    dtype=np.uint8
                )
        except Exception as e:
            pass
    return water_mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load UNet Model
    CHECKPOINT = "checkpoint/best_sam_unet_model.pth"
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    # We will evaluate on the training set because it contains ground truth JSON labels.
    # We will evaluate on the first 100 images that have JSON labels for speed.
    IMG_DIR = "data/processed/train/images"
    LABEL_DIR = "data/processed/train/labels"
    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.upper().endswith(('.TIF', '.TIFF'))])[:150]

    h_metrics = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "oa": 0.0}
    u_metrics = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "oa": 0.0}
    count = 0

    for fname in tqdm(img_files, desc="Comparing"):
        base_name = os.path.splitext(fname)[0]
        img_path = os.path.join(IMG_DIR, fname)
        label_path = os.path.join(LABEL_DIR, f"{base_name}.json")

        if not os.path.exists(label_path):
            continue

        try:
            with rasterio.open(img_path) as src:
                image = src.read([1, 2, 3])
                image = np.moveaxis(image, 0, -1).astype(np.uint8)
                geo_transform = src.transform
                height, width = src.height, src.width
        except Exception:
            continue

        # Load Pure JSON ground truth
        gt_mask = load_pure_json_mask(label_path, height, width, geo_transform)
        gt_512 = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        gt_tensor = torch.from_numpy(gt_512).byte()

        # Resized image for prediction
        img_512 = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

        # 1. Heuristic Prediction
        h_pred_mask = compute_heuristic_water(img_512)
        h_pred_tensor = torch.from_numpy(h_pred_mask).byte()

        # 2. U-Net Prediction
        # Normalize image
        img_norm = (img_512.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.sigmoid(outputs)[0, 0].cpu().numpy()
        u_pred_mask = (probs > 0.5).astype(np.uint8)
        u_pred_tensor = torch.from_numpy(u_pred_mask).byte()

        # Compute metrics
        h_iou, h_prec, h_rec, h_oa = calculate_metrics(h_pred_tensor, gt_tensor)
        u_iou, u_prec, u_rec, u_oa = calculate_metrics(u_pred_tensor, gt_tensor)

        h_metrics["iou"] += h_iou
        h_metrics["precision"] += h_prec
        h_metrics["recall"] += h_rec
        h_metrics["oa"] += h_oa

        u_metrics["iou"] += u_iou
        u_metrics["precision"] += u_prec
        u_metrics["recall"] += u_rec
        u_metrics["oa"] += u_oa

        count += 1

    for k in h_metrics:
        h_metrics[k] /= count
        u_metrics[k] /= count

    print("\n" + "="*50)
    print("         COMPARISON ON REAL JSON GT LABELS")
    print("="*50)
    print(f"Evaluated Images with JSON labels: {count}")
    print("\n[Approach A] Heuristic Color-Texture Filter:")
    print(f"- Mean IoU: {h_metrics['iou']*100:.2f}%")
    print(f"- Recall: {h_metrics['recall']*100:.2f}%")
    print(f"- Precision: {h_metrics['precision']*100:.2f}%")
    print(f"- Overall Accuracy (OA): {h_metrics['oa']*100:.2f}%")
    
    print("\n[Approach B] U-Net Model:")
    print(f"- Mean IoU: {u_metrics['iou']*100:.2f}%")
    print(f"- Recall: {u_metrics['recall']*100:.2f}%")
    print(f"- Precision: {u_metrics['precision']*100:.2f}%")
    print(f"- Overall Accuracy (OA): {u_metrics['oa']*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
