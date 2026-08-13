"""
boundary_error_analysis.py - F1-Score 기반 경계선 오차 분석 그래프 생성

분석 항목:
  1. F1-Score vs Threshold 곡선  (임계값 0.1~0.9 스캔)
  2. 경계선 허용 오차(픽셀 거리) vs Boundary-F1 곡선
  3. 이미지별 F1-Score 분포 히스토그램
"""

import os, sys, torch
import numpy as np
import cv2
from scipy import ndimage
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.abspath('.'))
from src.dataset import WaterFringeDataset
from src.model import UNet

# ─────────────────────────────────────────────
CHECKPOINT  = "checkpoint/best_sam_unet_model.pth"
IMG_DIR     = "data/processed/val/images"
LABEL_DIR   = "data/processed/val/labels"
ARTIFACT    = r"C:\Users\kimse\.gemini\antigravity\brain\831c6100-a054-4b10-9032-e60e39028194"
SAVE_PATHS  = [
    r"C:\Users\kimse\water_fringe_detection - 복사본\result_images\boundary_f1_analysis.png",
    r"C:\Users\kimse\water_fringe_detection\result_images\boundary_f1_analysis.png",
    rf"{ARTIFACT}\images\boundary_f1_analysis.png",
    rf"{ARTIFACT}\boundary_f1_analysis.png",
]
# ─────────────────────────────────────────────

def dice_f1(pred_mask, gt_mask, smooth=1e-6):
    """바이너리 마스크 간 Dice F1 계산"""
    tp = float((pred_mask & gt_mask).sum())
    fp = float((pred_mask & ~gt_mask).sum())
    fn = float((~pred_mask & gt_mask).sum())
    return (2*tp + smooth) / (2*tp + fp + fn + smooth)

def boundary_mask(mask, radius):
    """마스크 경계선 픽셀을 반지름 radius 픽셀만큼 팽창"""
    dilated  = ndimage.binary_dilation(mask, iterations=radius)
    eroded   = ndimage.binary_erosion (mask, iterations=radius)
    return dilated ^ eroded   # XOR → 경계 띠 (boundary band)

def boundary_f1(pred, gt, radius):
    """
    경계선 허용 오차 Boundary-F1:
    - 예측 경계선 안에 GT 경계선이 얼마나 들어오는가 (Precision)
    - GT 경계선 안에 예측 경계선이 얼마나 들어오는가 (Recall)
    """
    pred_b = boundary_mask(pred, radius)
    gt_b   = boundary_mask(gt,   radius)
    # dilated boundary overlap
    pred_dil = ndimage.binary_dilation(pred_b, iterations=radius)
    gt_dil   = ndimage.binary_dilation(gt_b,   iterations=radius)
    prec = (pred_b & gt_dil ).sum() / (pred_b.sum() + 1e-6)
    rec  = (gt_b   & pred_dil).sum() / (gt_b.sum()  + 1e-6)
    return 2*prec*rec/(prec+rec+1e-6)

# ─── 데이터 로드 ──────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])
val_ds = WaterFringeDataset(IMG_DIR, LABEL_DIR, transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
print(f"Validation images: {len(val_ds)}")

model = UNet(in_channels=3, num_classes=1).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

# ─── 이미지별 확률맵 & 마스크 수집 ──────────────
print("Collecting predictions...")
all_probs  = []
all_labels = []

with torch.no_grad():
    for images, masks in tqdm(val_loader, desc="Inference"):
        images = images.to(device)
        with torch.cuda.amp.autocast():
            probs = torch.sigmoid(model(images))
        for p, m in zip(probs.cpu().numpy(), masks.cpu().numpy()):
            all_probs.append(p[0])    # (H, W)
            all_labels.append(m[0].astype(bool))  # (H, W)

all_probs  = np.array(all_probs,  dtype=np.float32)  # (N, H, W)
all_labels = np.array(all_labels, dtype=bool)         # (N, H, W)
N = len(all_probs)
print(f"Collected {N} samples.")

# ─── 분석 1: F1 vs Threshold 곡선 ─────────────
print("Analysis 1: F1 vs Threshold ...")
thresholds = np.linspace(0.05, 0.95, 37)
f1_means, f1_stds = [], []

for thr in thresholds:
    preds = all_probs > thr          # (N, H, W) bool
    f1s = []
    for p, g in zip(preds, all_labels):
        f1s.append(dice_f1(p, g))
    f1_means.append(np.mean(f1s))
    f1_stds.append(np.std(f1s))

f1_means = np.array(f1_means)
f1_stds  = np.array(f1_stds)
best_thr_idx = np.argmax(f1_means)
best_thr     = thresholds[best_thr_idx]
best_f1      = f1_means[best_thr_idx]

# ─── 분석 2: Boundary-F1 vs Tolerance 곡선 ────
print("Analysis 2: Boundary-F1 vs Tolerance ...")
thr_fixed = 0.5          # 고정 임계값
radii = [1, 2, 3, 4, 5, 7, 9, 12, 15]
bf1_means = []

for r in radii:
    bf1s = []
    for prob, gt in zip(all_probs, all_labels):
        pred = prob > thr_fixed
        if gt.sum() < 10:    # GT 경계 없으면 스킵
            continue
        bf1s.append(boundary_f1(pred, gt, r))
    bf1_means.append(np.mean(bf1s))
    print(f"  radius={r:2d}px → BF1={bf1_means[-1]*100:.2f}%")

# ─── 분석 3: 이미지별 F1 분포 ─────────────────
print("Analysis 3: Per-image F1 distribution ...")
per_f1 = []
preds_fixed = all_probs > thr_fixed
for p, g in zip(preds_fixed, all_labels):
    per_f1.append(dice_f1(p, g))
per_f1 = np.array(per_f1)

# ─── 시각화 ───────────────────────────────────
print("Plotting ...")
plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0f0f1a')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# ─ 패널 [0,0]: F1 vs Threshold ─
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor('#0f0f1a')
ax0.plot(thresholds, f1_means*100, color='#5cd6ff', lw=2.5, label='Mean F1-Score')
ax0.fill_between(thresholds,
                 (f1_means - f1_stds)*100,
                 (f1_means + f1_stds)*100,
                 alpha=0.18, color='#5cd6ff', label='±1 Std Dev')
ax0.axvline(best_thr, color='#ffdd57', lw=1.8, ls='--',
            label=f'Best Threshold = {best_thr:.2f}')
ax0.axhline(best_f1*100, color='#ff6b6b', lw=1.2, ls=':',
            label=f'Peak F1 = {best_f1*100:.2f}%')
ax0.axvline(0.5, color='#aaaaaa', lw=1.0, ls=':', alpha=0.6, label='Current Thr = 0.50')
ax0.scatter([best_thr], [best_f1*100], color='#ffdd57', s=90, zorder=5)
ax0.set_xlabel('Threshold', fontsize=11, color='#a0a0c0')
ax0.set_ylabel('F1-Score (%)', fontsize=11, color='#a0a0c0')
ax0.set_title('F1-Score vs Threshold Curve', fontsize=13, fontweight='bold', color='white', pad=12)
ax0.tick_params(colors='#a0a0c0', labelsize=9)
ax0.legend(fontsize=8.5, facecolor='#0f0f1a', edgecolor='#555', labelcolor='white')
ax0.grid(ls='--', alpha=0.18, color='#a0a0c0')
for spine in ax0.spines.values():
    spine.set_visible(False)

# ─ 패널 [0,1]: Boundary-F1 vs Tolerance ─
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_facecolor('#0f0f1a')
pixel_mm = [r for r in radii]   # 1 pixel ≈ 실제 해상도에 비례
ax1.plot(radii, np.array(bf1_means)*100, 'o-', color='#ff7f0e',
         lw=2.5, ms=7, label='Boundary F1-Score')
for r, v in zip(radii, bf1_means):
    ax1.annotate(f'{v*100:.1f}%', (r, v*100),
                 textcoords="offset points", xytext=(0, 8),
                 ha='center', fontsize=8.5, color='#ff7f0e', fontweight='bold')
ax1.set_xlabel('Boundary Tolerance (pixels)', fontsize=11, color='#a0a0c0')
ax1.set_ylabel('Boundary F1-Score (%)', fontsize=11, color='#a0a0c0')
ax1.set_title('Boundary Error Tolerance Analysis\n(경계선 허용 오차 vs Boundary-F1)', fontsize=12, fontweight='bold', color='white', pad=12)
ax1.tick_params(colors='#a0a0c0', labelsize=9)
ax1.legend(fontsize=9, facecolor='#0f0f1a', edgecolor='#555', labelcolor='white')
ax1.grid(ls='--', alpha=0.18, color='#a0a0c0')
for spine in ax1.spines.values():
    spine.set_visible(False)

# ─ 패널 [1,0]: 이미지별 F1 히스토그램 ─
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#0f0f1a')
counts, bins, patches = ax2.hist(per_f1*100, bins=22,
                                  color='#2ca02c', edgecolor='none', alpha=0.82)
# 구간별 색상 그라디언트
for patch, left in zip(patches, bins):
    patch.set_facecolor(plt.cm.RdYlGn(left/100))
ax2.axvline(np.mean(per_f1)*100, color='#ffdd57', lw=2, ls='--',
            label=f'Mean F1 = {np.mean(per_f1)*100:.2f}%')
ax2.axvline(np.median(per_f1)*100, color='#5cd6ff', lw=1.5, ls=':',
            label=f'Median F1 = {np.median(per_f1)*100:.2f}%')
ax2.set_xlabel('F1-Score (%)', fontsize=11, color='#a0a0c0')
ax2.set_ylabel('Image Count', fontsize=11, color='#a0a0c0')
ax2.set_title('Per-Image F1-Score Distribution\n(이미지별 F1 분포, threshold=0.50)', fontsize=12, fontweight='bold', color='white', pad=12)
ax2.tick_params(colors='#a0a0c0', labelsize=9)
ax2.legend(fontsize=9, facecolor='#0f0f1a', edgecolor='#555', labelcolor='white')
ax2.grid(axis='y', ls='--', alpha=0.18, color='#a0a0c0')
for spine in ax2.spines.values():
    spine.set_visible(False)

# ─ 패널 [1,1]: F1 누적 분포 (CDF) ─
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#0f0f1a')
sorted_f1 = np.sort(per_f1)*100
cdf = np.arange(1, N+1) / N * 100
ax3.plot(sorted_f1, cdf, color='#9467bd', lw=2.5, label='CDF')
ax3.fill_between(sorted_f1, cdf, alpha=0.12, color='#9467bd')
# 주요 수치 표시
for pct, clr in [(50, '#ffdd57'), (80, '#5cd6ff'), (90, '#ff6b6b')]:
    idx = np.searchsorted(cdf, pct)
    if idx < len(sorted_f1):
        ax3.axhline(pct, color=clr, lw=1.2, ls=':', alpha=0.7)
        ax3.axvline(sorted_f1[idx], color=clr, lw=1.2, ls=':', alpha=0.7,
                    label=f'{pct}th pct → F1≥{sorted_f1[idx]:.1f}%')
ax3.set_xlabel('F1-Score (%)', fontsize=11, color='#a0a0c0')
ax3.set_ylabel('Cumulative % of Images', fontsize=11, color='#a0a0c0')
ax3.set_title('Cumulative F1-Score Distribution (CDF)\n(이미지별 F1 누적 분포)', fontsize=12, fontweight='bold', color='white', pad=12)
ax3.tick_params(colors='#a0a0c0', labelsize=9)
ax3.legend(fontsize=8.5, facecolor='#0f0f1a', edgecolor='#555', labelcolor='white')
ax3.grid(ls='--', alpha=0.18, color='#a0a0c0')
for spine in ax3.spines.values():
    spine.set_visible(False)

fig.suptitle(
    'U-Net Binary Water Segmentation — F1-Score Based Boundary Error Analysis',
    fontsize=16, fontweight='bold', color='white', y=1.01
)

for p in SAVE_PATHS:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {p}")
plt.close()
print("Done!")
