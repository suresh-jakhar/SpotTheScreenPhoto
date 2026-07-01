import sys
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TORCH_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_model.pt")

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Logistic Regression weights
_FEAT_MEAN = np.array([
    0.471953, 0.156936, 0.039762, 0.057894, 977.260199, 42.487581, 45.188113,
    0.404669, 20.824939, 16.852001, 1.077532, 1.020260, 0.473109, 1.770352,
    0.297208, 0.374090, 0.176587, 0.046537, 1.918966, 22.601349, 12.912323
])
_FEAT_SCALE = np.array([
    0.480031, 0.072693, 0.040399, 0.084061, 763.911188, 19.273182, 18.643994,
    0.375783, 19.854671, 19.010620, 0.282192, 0.221774, 0.135561, 0.252893,
    0.175835, 0.224321, 0.256253, 0.132650, 0.564843, 18.230482, 15.714429
])
_FEAT_COEF = np.array([
    5.199552, -0.086854, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, -0.322890, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000
])
_FEAT_BIAS = 0.163181

_model_cache = None


def _load_model():
    global _model_cache
    if _model_cache is None:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            _model_cache = torch.jit.trace(model, dummy)
    return _model_cache


def _extract_features(path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    sz = min(w, h, 512)
    cx, cy = w // 2, h // 2
    
    crop = np.asarray(img.crop((cx - sz//2, cy - sz//2, cx + sz//2, cy + sz//2))).astype(np.float32)
    gray = crop.mean(axis=2)
    H, W = gray.shape
    feats = []

    # 1. 2D FFT features
    f = np.abs(np.fft.fft2(gray))
    f[0, 0] = 0
    fs = np.fft.fftshift(f)
    
    cy2, cx2 = H // 2, W // 2
    y, x = np.ogrid[-cy2:H - cy2, -cx2:W - cx2]
    r2 = y * y + x * x
    
    e_low = float(fs[r2 < 400].mean())
    e_mid = float(fs[(r2 >= 400) & (r2 < 3600)].mean())
    e_high = float(fs[r2 >= 3600].mean())
    tot = e_low + e_mid + e_high + 1e-6
    
    feats.extend([
        e_mid / tot,
        e_high / tot,
        e_high / (e_low + 1e-6),
        float(f.max() / (f.mean() + 1e-6))
    ])

    # 2. 1D Projections
    for proj in [gray.mean(axis=0), gray.mean(axis=1)]:
        proj_zero_mean = proj - proj.mean()
        ps = np.abs(np.fft.rfft(proj_zero_mean))[1:]
        feats.append(float(ps.max() / (ps.mean() + 1e-6)))

    # 3. High-frequency noise residuals
    mid_row = gray[H // 2, :]
    smooth_row = np.convolve(mid_row, np.ones(5) / 5, mode='same')
    res = mid_row - smooth_row
    
    acf = np.correlate(res, res, mode='full')[len(res) - 1:]
    acf /= (acf[0] + 1e-6)
    feats.extend([float(acf[2:25].max()), float(np.std(res)), float(np.mean(np.abs(res)))])

    # 4. DCT Blockiness
    b_r = [abs(float(gray[r].mean() - gray[r - 1].mean())) for r in range(8, H - 1, 8)]
    i_r = [abs(float(gray[r].mean() - gray[r - 1].mean())) for r in range(9, H - 1) if r % 8]
    feats.append(np.mean(b_r) / (np.mean(i_r) + 1e-6) if b_r and i_r else 1.0)
    
    b_c = [abs(float(gray[:, c].mean() - gray[:, c - 1].mean())) for c in range(8, W - 1, 8)]
    i_c = [abs(float(gray[:, c].mean() - gray[:, c - 1].mean())) for c in range(9, W - 1) if c % 8]
    feats.append(np.mean(b_c) / (np.mean(i_c) + 1e-6) if b_c and i_c else 1.0)

    # 5. Gradient orientations
    gy = (gray[1:, :] - gray[:-1, :])[:, :W - 1]
    gx = (gray[:, 1:] - gray[:, :W - 1])[:H - 1, :]
    angle = np.degrees(np.arctan2(gy, gx)) % 180
    
    hv_edges = float(((angle < 12) | (angle > 168) | ((angle > 78) & (angle < 102))).mean())
    hist, _ = np.histogram(angle.ravel(), bins=8, range=(0, 180), density=True)
    hist /= (hist.sum() + 1e-6)
    feats.extend([hv_edges, float(-np.sum(hist * np.log(hist + 1e-9)))])

    # 6. Color Saturation
    hsv = np.asarray(Image.fromarray(crop.astype(np.uint8)).convert("HSV")).astype(np.float32)
    sat = hsv[:, :, 1] / 255
    feats.extend([float(sat.mean()), float(np.percentile(sat, 75)), float((sat > 0.5).mean())])

    # 7. Brightness Bimodality
    lum = gray / 255
    hist_b, _ = np.histogram(lum.ravel(), bins=16, range=(0, 1), density=True)
    hist_b /= (hist_b.sum() + 1e-6)
    feats.extend([float((lum < 0.10).mean()), float(-np.sum(hist_b * np.log(hist_b + 1e-9)))])

    # 8. Local Texture Variance
    stds = np.array([
        float(gray[r:r + 15, c:c + 15].std())
        for r in range(0, H - 15, 15)
        for c in range(0, W - 15, 15)
    ])
    feats.extend([float(stds.mean()), float(np.percentile(stds, 10))])

    return np.array(feats, dtype=np.float64)


def predict(image_path: str) -> float:
    model = _load_model()
    img = Image.open(image_path)
    img.draft("RGB", (IMG_SIZE, IMG_SIZE))
    
    x = _TRANSFORM(img.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        cnn_score = float(torch.softmax(_model_cache(x), dim=1)[0, 1].item())
    
    feats = np.array([cnn_score] + list(_extract_features(image_path)), dtype=np.float64)
    x_scaled = (feats - _FEAT_MEAN) / (_FEAT_SCALE + 1e-9)
    logit = float(np.dot(x_scaled, _FEAT_COEF) + _FEAT_BIAS)
    
    return 1.0 / (1.0 + np.exp(-logit))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)
    
    image_path = " ".join(sys.argv[1:])
    
    if not os.path.exists(image_path):
        print(f"Error: File not found at '{image_path}'", file=sys.stderr)
        sys.exit(1)
        
    res = predict(image_path)
    label = "SCREEN" if res > 0.5 else "REAL"
    print(f"Prediction: {label} (Screen confidence: {res:.2f})")
