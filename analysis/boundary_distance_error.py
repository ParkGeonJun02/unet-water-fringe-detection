"""
boundary_distance_error.py

Boundary accuracy comparison between:
1. JSON polygon Ground Truth
2. Color + Texture heuristic baseline
3. U-Net water segmentation

Metrics:
- Mean Boundary Distance (MBD)
- Boundary F1-Score at multiple pixel tolerances

Note:
This analysis uses images with available JSON annotations
from the training dataset for baseline comparison.
"""

import os, sys, json, torch
import numpy as np
import cv2
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('.'))
from src.model import UNet

# ─── 설정 ────────────────────────────────────────────
CHECKPOINT  = "checkpoint/best_unet_model.pth"
TRAIN_IMG   = "data/processed/train/images"
TRAIN_LABEL = "data/processed/train/labels"

SAVE_MBD = "results/boundary_mbd.png"
SAVE_F1  = "results/boundary_f1_bar.png"

RADII = [1, 5, 10, 15]
THR   = 0.5
# ──────────────────────────────────────────────────────

# ─── 헬퍼 함수 ───────────────────────────────────────
def get_boundary(mask_bool, w=1):
    d = ndimage.binary_dilation(mask_bool, iterations=w)
    e = ndimage.binary_erosion (mask_bool, iterations=w)
    return d ^ e

def sym_mbd(pred_bool, gt_bool):
    """대칭 평균 경계선 거리 (픽셀), None if invalid"""
    pb = get_boundary(pred_bool)
    gb = get_boundary(gt_bool)
    if pb.sum() < 5 or gb.sum() < 5:
        return None
    d2pred = ndimage.distance_transform_edt(~pb)
    d2gt   = ndimage.distance_transform_edt(~gb)
    return float((d2pred[gb].mean() + d2gt[pb].mean()) / 2)

def boundary_f1(pred_bool, gt_bool, r):
    pb = get_boundary(pred_bool)
    gb = get_boundary(gt_bool)
    if pb.sum() < 5 or gb.sum() < 5:
        return None
    pb_dil = ndimage.binary_dilation(pb, iterations=r)
    gb_dil = ndimage.binary_dilation(gb, iterations=r)
    prec = (pb & gb_dil).sum() / (pb.sum() + 1e-8)
    rec  = (gb & pb_dil).sum() / (gb.sum() + 1e-8)
    return float(2*prec*rec/(prec+rec+1e-8)) if prec+rec > 1e-8 else None

def color_heuristic(img_uint8):
    r  = img_uint8[:,:,0].astype(np.float32)
    g  = img_uint8[:,:,1].astype(np.float32)
    b  = img_uint8[:,:,2].astype(np.float32)
    gr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mn = cv2.boxFilter(gr,-1,(31,31))
    sq = cv2.boxFilter(gr**2,-1,(31,31))
    sd = np.sqrt(np.clip(sq-mn**2,0,None))
    cw = ((r<85)&(g<110)&(b<110)&(g>r)&(sd<5.0)).astype(np.uint8)
    cw = cv2.morphologyEx(cw, cv2.MORPH_OPEN, np.ones((9,9),np.uint8))
    return cw.astype(bool)

def load_json_gt(label_path, H, W, geo_transform):
    gt = np.zeros((H,W), dtype=np.uint8)
    if not os.path.exists(label_path): return gt
    try:
        with open(label_path,"r",encoding="utf-8") as f: meta=json.load(f)
        shapes = []
        for feat in meta.get("annotation",{}).get("features",[]):
            code = str(feat.get("properties",{}).get("CODE","")).strip()
            geo  = feat.get("geometry",{})
            if code in ["50","20","40","511"] and geo.get("coordinates"):
                try: shapes.append((shape(geo),1))
                except: pass
        if shapes:
            gt = rasterize(shapes, out_shape=(H,W),
                           transform=geo_transform, fill=0, dtype=np.uint8)
    except: pass
    return gt

# ─── 간단한 Dataset (학습셋 이미지, 정규화 포함) ─────
class SimpleImgDataset(Dataset):
    def __init__(self, img_dir, img_files):
        self.img_dir   = img_dir
        self.img_files = img_files
        self.transform = A.Compose([
            A.Resize(512,512),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])
    def __len__(self): return len(self.img_files)
    def __getitem__(self, idx):
        name = self.img_files[idx]
        try:
            with rasterio.open(os.path.join(self.img_dir, name)) as src:
                raw = np.moveaxis(src.read([1,2,3]),0,-1).astype(np.uint8)
        except:
            return torch.zeros(3,512,512), idx
        t = self.transform(image=raw)["image"]
        return t, idx

# ─── 1. 학습셋 이미지 목록 ─────────────────────────
img_files = sorted([f for f in os.listdir(TRAIN_IMG)
                    if f.upper().endswith((".TIF",".TIFF"))])
print(f"Train images: {len(img_files)}")

# ─── 2. U-Net 추론 ──────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = UNet(in_channels=3, num_classes=1).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

ds_img = SimpleImgDataset(TRAIN_IMG, img_files)
loader = DataLoader(ds_img, batch_size=4, shuffle=False, num_workers=0)

unet_probs = [None]*len(img_files)
print("U-Net inference on train set ...")
with torch.no_grad():
    for imgs, idxs in tqdm(loader, desc="Inference"):
        imgs = imgs.to(device)
        with torch.amp.autocast('cuda'):
            p = torch.sigmoid(model(imgs))
        for prob, idx in zip(p.cpu().numpy(), idxs.numpy()):
            unet_probs[idx] = prob[0].astype(np.float32)

# ─── 3. GT(JSON) + Heuristic 생성 ────────────────────
print("Loading GT & Heuristic ...")
gt_masks   = []
heur_masks = []
raw_imgs   = []   # for heuristic (원본 픽셀)

for name in tqdm(img_files, desc="Load"):
    base = os.path.splitext(name)[0]
    try:
        with rasterio.open(os.path.join(TRAIN_IMG, name)) as src:
            raw = np.moveaxis(src.read([1,2,3]),0,-1).astype(np.uint8)
            geo = src.transform
            H,W = src.height, src.width
    except:
        gt_masks.append(None); heur_masks.append(None); continue

    gt_raw  = load_json_gt(os.path.join(TRAIN_LABEL,f"{base}.json"),H,W,geo)
    heur_raw = color_heuristic(raw)

    gt_r   = cv2.resize(gt_raw.astype(np.uint8),  (512,512),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    heur_r = cv2.resize(heur_raw.astype(np.uint8),(512,512),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    gt_masks.append(gt_r)
    heur_masks.append(heur_r)

# ─── 4. 지표 계산 (매칭 페어) ─────────────────────────
print("\nComputing boundary metrics ...")
mbd_u_list, mbd_h_list = [], []
bf1_u = {r:[] for r in RADII}
bf1_h = {r:[] for r in RADII}
n_valid = 0

for i in range(len(img_files)):
    gt = gt_masks[i]
    if gt is None or gt.sum() < 100:  # GT 물 영역 충분한 이미지
        continue
    unet = unet_probs[i] > THR if unet_probs[i] is not None else None
    if unet is None: continue
    heur = heur_masks[i]
    n_valid += 1

    # MBD
    mu = sym_mbd(unet, gt)
    mh = sym_mbd(heur, gt)
    if mu is not None and mh is not None:
        mbd_u_list.append(mu)
        mbd_h_list.append(mh)

    # Boundary F1
    for r in RADII:
        fu = boundary_f1(unet, gt, r)
        fh = boundary_f1(heur, gt, r)
        if fu is not None: bf1_u[r].append(fu)
        if fh is not None: bf1_h[r].append(fh)

mbd_u = np.array(mbd_u_list)
mbd_h = np.array(mbd_h_list)
n_mbd = len(mbd_u)

print(f"\nValid images (GT>=100px): {n_valid}, MBD pairs: {n_mbd}")
print(f"Mean Boundary Distance:")
print(f"  Heuristic : {np.mean(mbd_h):.2f} px  (std={np.std(mbd_h):.2f})")
print(f"  U-Net     : {np.mean(mbd_u):.2f} px  (std={np.std(mbd_u):.2f})")
reduce_px = np.mean(mbd_h) - np.mean(mbd_u)
print(f"  U-Net 감소: {reduce_px:+.2f} px")

bf1_u_m = [np.mean(bf1_u[r])*100 if bf1_u[r] else 0 for r in RADII]
bf1_h_m = [np.mean(bf1_h[r])*100 if bf1_h[r] else 0 for r in RADII]
bf1_u_s = [np.std (bf1_u[r])*100 if bf1_u[r] else 0 for r in RADII]
bf1_h_s = [np.std (bf1_h[r])*100 if bf1_h[r] else 0 for r in RADII]
print("\nBoundary F1:")
for r,u,h in zip(RADII,bf1_u_m,bf1_h_m):
    print(f"  {r:2d}px  H={h:.2f}%  U={u:.2f}%  Δ={u-h:+.2f}%")

# ─── 5. 시각화 ────────────────────────────────────────
COLOR_H = '#ff7043'
COLOR_U = '#29b6f6'

plt.style.use('dark_background')

# =====================================================
# Figure 1: Mean Boundary Distance (MBD)
# =====================================================
fig1, ax0 = plt.subplots(figsize=(9, 8))
fig1.patch.set_facecolor('#0f0f1a')
ax0.set_facecolor('#0f0f1a')

mu_h = np.mean(mbd_h)
mu_u = np.mean(mbd_u)
reduce_px = mu_h - mu_u

clip_val = np.percentile(
    np.concatenate([mbd_h, mbd_u]),
    90
)

vp = ax0.violinplot(
    [
        np.clip(mbd_h, 0, clip_val),
        np.clip(mbd_u, 0, clip_val)
    ],
    positions=[1, 2],
    showmedians=True,
    showextrema=True,
    widths=0.6
)

for body, color in zip(
    vp['bodies'],
    [COLOR_H, COLOR_U]
):
    body.set_facecolor(color)
    body.set_alpha(0.3)

vp['cmedians'].set_color('white')
vp['cmedians'].set_linewidth(2.5)

for part in ['cmaxes', 'cmins', 'cbars']:
    vp[part].set_color('#777788')
    vp[part].set_linewidth(1.2)

ax0.scatter(
    [1],
    [mu_h],
    s=180,
    color=COLOR_H,
    zorder=6,
    edgecolors='white',
    linewidths=2,
    label=f'Heuristic mean = {mu_h:.1f} px'
)

ax0.scatter(
    [2],
    [mu_u],
    s=180,
    color=COLOR_U,
    zorder=6,
    edgecolors='white',
    linewidths=2,
    label=f'U-Net mean = {mu_u:.1f} px'
)

ax0.annotate(
    '',
    xy=(2.32, mu_u),
    xytext=(2.32, mu_h),
    arrowprops=dict(
        arrowstyle='<->',
        color='#7effc4',
        lw=3
    )
)

ax0.text(
    2.52,
    (mu_h + mu_u) / 2,
    f'U-Net\n▼ {reduce_px:.1f} px\nreduced',
    color='#7effc4',
    fontsize=13,
    fontweight='bold',
    va='center'
)

ax0.set_xticks([1, 2])
ax0.set_xticklabels(
    [
        'Heuristic\n(Color+Texture)',
        'Proposed\nU-Net'
    ],
    fontsize=13,
    color='#d0d0e8'
)

ax0.set_ylabel(
    'Mean Boundary Distance (pixels)',
    fontsize=12,
    color='#a0a0c0'
)

ax0.set_title(
    f'Boundary Location Error (MBD)\n'
    f'(Training-set JSON annotations, {n_mbd} matched pairs)',
    fontsize=14,
    fontweight='bold',
    color='white',
    pad=14
)

ax0.tick_params(
    colors='#a0a0c0',
    labelsize=11
)

ax0.set_ylim(0, clip_val * 1.32)

ax0.legend(
    fontsize=11,
    facecolor='#0f0f1a',
    edgecolor='#444455',
    labelcolor='white',
    loc='upper right'
)

ax0.grid(
    axis='y',
    linestyle='--',
    alpha=0.18,
    color='#a0a0c0'
)

for spine in ax0.spines.values():
    spine.set_visible(False)

fig1.suptitle(
    f'U-Net reduced boundary error by {reduce_px:.1f} px '
    f'({mu_h:.1f} px → {mu_u:.1f} px)',
    fontsize=13,
    fontweight='bold',
    color='#7effc4',
    y=1.02
)

plt.tight_layout()

os.makedirs(
    os.path.dirname(SAVE_MBD),
    exist_ok=True
)

fig1.savefig(
    SAVE_MBD,
    dpi=180,
    bbox_inches='tight',
    facecolor=fig1.get_facecolor()
)

print(f"Saved: {SAVE_MBD}")
plt.close(fig1)


# =====================================================
# Figure 2: Boundary F1
# =====================================================
fig2, ax1 = plt.subplots(figsize=(13, 9))
fig2.patch.set_facecolor('#0f0f1a')
ax1.set_facecolor('#0f0f1a')

x = np.arange(len(RADII))
w = 0.36

ax1.bar(
    x - w/2,
    bf1_h_m,
    w,
    label='Heuristic (Color+Texture)',
    color=COLOR_H,
    alpha=0.88,
    zorder=3,
    yerr=bf1_h_s,
    capsize=3
)

ax1.bar(
    x + w/2,
    bf1_u_m,
    w,
    label='Proposed U-Net',
    color=COLOR_U,
    alpha=0.88,
    zorder=3,
    yerr=bf1_u_s,
    capsize=3
)

y_top = max(
    max(bf1_u_m),
    max(bf1_h_m)
)

for i, (u, h) in enumerate(
    zip(bf1_u_m, bf1_h_m)
):
    delta = u - h

    ax1.text(
        i,
        max(u, h) + 5,
        f'Δ {delta:+.1f}%',
        ha='center',
        fontsize=10,
        color='#7effc4',
        fontweight='bold'
    )

ax1.set_xticks(x)

ax1.set_xticklabels(
    [f'{r}px' for r in RADII],
    fontsize=12,
    color='#d0d0e8'
)

ax1.set_xlabel(
    'Boundary Tolerance (pixels)',
    fontsize=12,
    color='#a0a0c0'
)

ax1.set_ylabel(
    'Boundary F1-Score (%)',
    fontsize=12,
    color='#a0a0c0'
)

ax1.set_title(
    'Boundary F1 @ Tolerance\n'
    '(Training-set JSON annotation comparison)',
    fontsize=14,
    fontweight='bold',
    color='white',
    pad=14
)

ax1.tick_params(
    colors='#a0a0c0',
    labelsize=11
)

ax1.set_ylim(
    0,
    y_top + 16
)

ax1.legend(
    fontsize=11,
    facecolor='#0f0f1a',
    edgecolor='#444455',
    labelcolor='white'
)

ax1.grid(
    axis='y',
    linestyle='--',
    alpha=0.18,
    color='#a0a0c0',
    zorder=0
)

for spine in ax1.spines.values():
    spine.set_visible(False)

best_delta = max(
    bf1_u_m[i] - bf1_h_m[i]
    for i in range(len(RADII))
)

fig2.suptitle(
    f'U-Net Boundary F1 improvement '
    f'(maximum: +{best_delta:.1f} percentage points)',
    fontsize=13,
    fontweight='bold',
    color='#7effc4',
    y=1.01
)

plt.tight_layout()

os.makedirs(
    os.path.dirname(SAVE_F1),
    exist_ok=True
)

fig2.savefig(
    SAVE_F1,
    dpi=180,
    bbox_inches='tight',
    facecolor=fig2.get_facecolor()
)

print(f"Saved: {SAVE_F1}")
plt.close(fig2)

print("All done!")
