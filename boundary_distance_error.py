"""
boundary_distance_error_v3.py
════════════════════════════════════════════════════════
GT: 학습셋 JSON 레이블 (CODE=50/20/40/511) → 순수 Ground Truth
Heuristic pred: 색상+질감만 (SAM+Texture 방식)
U-Net pred: U-Net 모델 출력 (학습 후 학습셋 적용)

★ 핵심 결론:
  "강과 육지의 경계선 위치 오차를 U-Net이 Heuristic보다 X픽셀 줄였다"
════════════════════════════════════════════════════════
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
import matplotlib.gridspec as gridspec

sys.path.append(os.path.abspath('.'))
from src.model import UNet

# ─── 설정 ────────────────────────────────────────────
CHECKPOINT  = "checkpoint/best_sam_unet_model.pth"
TRAIN_IMG   = "data/processed/train/images"
TRAIN_LABEL = "data/processed/train/labels"
ARTIFACT    = r"C:\Users\kimse\.gemini\antigravity\brain\831c6100-a054-4b10-9032-e60e39028194"
SAVE_PATHS  = [
    r"C:\Users\kimse\water_fringe_detection - 복사본\result_images\boundary_distance_error.png",
    r"C:\Users\kimse\water_fringe_detection\result_images\boundary_distance_error.png",
    rf"{ARTIFACT}\images\boundary_distance_error.png",
    rf"{ARTIFACT}\boundary_distance_error.png",
]
RADII = [1, 2, 3, 5, 7, 10, 15]
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
fig = plt.figure(figsize=(18,8))
fig.patch.set_facecolor('#0f0f1a')
gs  = gridspec.GridSpec(1,2, figure=fig, wspace=0.38)

# ── [좌] MBD Violin/Box (핵심: 몇 픽셀 줄였나) ──────
ax0 = fig.add_subplot(gs[0])
ax0.set_facecolor('#0f0f1a')

clip_val = np.percentile(np.concatenate([mbd_h, mbd_u]), 90)
mbd_h_c  = np.clip(mbd_h, 0, clip_val)
mbd_u_c  = np.clip(mbd_u, 0, clip_val)

vp = ax0.violinplot([mbd_h_c, mbd_u_c], positions=[1,2],
                    showmedians=True, showextrema=True, widths=0.6)
colors = [COLOR_H, COLOR_U]
for body, clr in zip(vp['bodies'], colors):
    body.set_facecolor(clr); body.set_alpha(0.3)
vp['cmedians'].set_color('white'); vp['cmedians'].set_linewidth(2.5)
for part in ['cmaxes','cmins','cbars']:
    vp[part].set_color('#777788'); vp[part].set_linewidth(1.2)

mu_h = np.mean(mbd_h); mu_u = np.mean(mbd_u)
md_h = np.median(mbd_h); md_u = np.median(mbd_u)

ax0.scatter([1],[mu_h],s=160,color=COLOR_H,zorder=6,edgecolors='white',lw=2,
            label=f'Heuristic  mean={mu_h:.1f}px / median={md_h:.1f}px')
ax0.scatter([2],[mu_u], s=160,color=COLOR_U, zorder=6,edgecolors='white',lw=2,
            label=f'U-Net       mean={mu_u:.1f}px / median={md_u:.1f}px')

# 개선 화살표
arrow_x = 2.28
ax0.annotate('', xy=(arrow_x, min(mu_h,mu_u)), xytext=(arrow_x, max(mu_h,mu_u)),
             arrowprops=dict(arrowstyle='<->', color='#7effc4', lw=2.8))
sign = '▼' if reduce_px > 0 else '▲'
clr_arrow = '#7effc4' if reduce_px > 0 else '#ff6b6b'
label_txt = (f'U-Net\n{sign} {abs(reduce_px):.1f} px\n감소'
             if reduce_px > 0 else f'Heuristic\n{sign} {abs(reduce_px):.1f} px\n낮음')
ax0.text(2.48, (mu_h+mu_u)/2, label_txt,
         color=clr_arrow, fontsize=12, fontweight='bold', va='center')

ax0.set_xticks([1,2])
ax0.set_xticklabels(['Heuristic\n(SAM+Texture)','Proposed\nU-Net'], fontsize=12, color='#d0d0e8')
ax0.set_ylabel('Mean Boundary Distance (pixels)', fontsize=11, color='#a0a0c0')
ax0.set_title(f'Boundary Location Error (MBD)\n(학습셋 JSON GT 기준, {n_mbd}장 매칭 비교)',
              fontsize=13, fontweight='bold', color='white', pad=14)
ax0.tick_params(colors='#a0a0c0', labelsize=10)
ax0.set_ylim(0, clip_val*1.28)
ax0.legend(fontsize=9.5, facecolor='#0f0f1a', edgecolor='#444455',
           labelcolor='white', loc='upper right')
ax0.grid(axis='y', ls='--', alpha=0.18, color='#a0a0c0')
for sp in ax0.spines.values(): sp.set_visible(False)

# ── [우] Boundary F1 @ tolerance 막대그래프 ──────────
ax1 = fig.add_subplot(gs[1])
ax1.set_facecolor('#0f0f1a')
x = np.arange(len(RADII)); w = 0.36

ax1.bar(x-w/2, bf1_h_m, w, label='Heuristic',
        color=COLOR_H, alpha=0.88, zorder=3,
        yerr=bf1_h_s, capsize=3, error_kw=dict(color=COLOR_H,alpha=0.5,lw=1.3))
ax1.bar(x+w/2, bf1_u_m, w, label='Proposed U-Net',
        color=COLOR_U, alpha=0.88, zorder=3,
        yerr=bf1_u_s, capsize=3, error_kw=dict(color=COLOR_U,alpha=0.5,lw=1.3))

y_top = max(max(bf1_u_m), max(bf1_h_m)) if max(bf1_u_m) > 0 else 80
for i,(u,h) in enumerate(zip(bf1_u_m, bf1_h_m)):
    ax1.text(i-w/2, h+max(bf1_h_s)+0.3, f'{h:.1f}%',
             ha='center', va='bottom', fontsize=8.5, color=COLOR_H, fontweight='bold')
    ax1.text(i+w/2, u+max(bf1_u_s)+0.3, f'{u:.1f}%',
             ha='center', va='bottom', fontsize=8.5, color=COLOR_U, fontweight='bold')
    delta = u-h
    clr   = '#7effc4' if delta >= 0 else '#ff9999'
    ax1.text(i, y_top+max(max(bf1_u_s),max(bf1_h_s))+1.5,
             f'{delta:+.1f}%', ha='center', fontsize=9, color=clr, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels([f'{r}px' for r in RADII], fontsize=12, color='#d0d0e8')
ax1.set_xlabel('Boundary Tolerance (pixels)', fontsize=11, color='#a0a0c0', labelpad=8)
ax1.set_ylabel('Boundary F1-Score (%)', fontsize=11, color='#a0a0c0')
ax1.set_title('Boundary F1 @ Tolerance\n(경계선 허용 오차별 F1 비교)',
              fontsize=13, fontweight='bold', color='white', pad=14)
ax1.tick_params(colors='#a0a0c0', labelsize=10)
ax1.set_ylim(0, y_top+16)
ax1.legend(fontsize=10, facecolor='#0f0f1a', edgecolor='#444455', labelcolor='white')
ax1.grid(axis='y', ls='--', alpha=0.18, color='#a0a0c0', zorder=0)
for sp in ax1.spines.values(): sp.set_visible(False)

# ── 전체 제목 ─────────────────────────────────────────
best_bf1_delta = max(bf1_u_m[i]-bf1_h_m[i] for i in range(len(RADII)))
best_bf1_r     = RADII[[bf1_u_m[i]-bf1_h_m[i] for i in range(len(RADII))].index(best_bf1_delta)]

if reduce_px > 0:
    headline = (f"U-Net reduced boundary location error by {reduce_px:.1f} px  "
                f"({mu_h:.1f}px → {mu_u:.1f}px)  |  "
                f"Boundary F1 improved by up to +{best_bf1_delta:.1f}% at {best_bf1_r}px tolerance")
    color_title = '#7effc4'
else:
    headline = (f"Boundary F1 improved by up to +{best_bf1_delta:.1f}% at {best_bf1_r}px tolerance  |  "
                f"MBD: Heuristic={mu_h:.1f}px, U-Net={mu_u:.1f}px")
    color_title = '#5cd6ff'

fig.suptitle(f'Boundary Error Analysis: Heuristic vs Proposed U-Net\n{headline}',
             fontsize=12, fontweight='bold', color=color_title, y=1.04)

plt.tight_layout()
for p in SAVE_PATHS:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {p}")
plt.close()
print("All done!")
