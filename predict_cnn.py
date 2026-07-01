import sys, os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

IMG_SIZE   = 224
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_model.pt")

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_FEAT_MEAN  = np.array([0.479354, 0.156936, 0.039762, 0.057894, 977.260199, 42.487581, 45.188113, 0.404669, 20.824939, 16.852001, 1.077532, 1.020260, 0.473109, 1.770352, 0.297208, 0.374090, 0.176587, 0.046537, 1.918966, 22.601349, 12.912323])
_FEAT_SCALE = np.array([0.442274, 0.072693, 0.040399, 0.084061, 763.911188, 19.273182, 18.643994, 0.375783, 19.854671, 19.010620, 0.282192, 0.221774, 0.135561, 0.252893, 0.175835, 0.224321, 0.256253, 0.132650, 0.564843, 18.230482, 15.714429])
_FEAT_COEF  = np.array([3.460854, -0.585190, -0.050778, 0.051411, 0.114216, -0.053902, -0.370575, 1.882412, -0.091033, 0.171035, -0.092371, -0.496738, 0.045128, -0.023932, -0.139912, 0.396726, -0.228919, -0.047700, -0.120029, 0.114035, 0.251046])
_FEAT_BIAS  = 0.915841

_model_cache = None


def _load_model():
    global _model_cache
    if _model_cache is None:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        example = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            _model_cache = torch.jit.trace(model, example)
    return _model_cache


def _extract_features(path) -> np.ndarray:
    img  = Image.open(path).convert("RGB")
    w, h = img.size
    sz   = min(w, h, 512)
    cx, cy = w // 2, h // 2
    crop = np.asarray(img.crop((cx-sz//2, cy-sz//2, cx+sz//2, cy+sz//2))).astype(np.float32)
    gray = crop.mean(axis=2)
    H, W = gray.shape
    feats = []

    f  = np.abs(np.fft.fft2(gray));  f[0, 0] = 0
    fs = np.fft.fftshift(f);  cy2, cx2 = H//2, W//2
    y_, x_ = np.ogrid[-cy2:H-cy2, -cx2:W-cx2];  r2 = y_*y_ + x_*x_
    e_low  = float(fs[r2 <  400].mean())
    e_mid  = float(fs[(r2 >= 400) & (r2 < 3600)].mean())
    e_high = float(fs[r2 >= 3600].mean())
    tot    = e_low + e_mid + e_high + 1e-6
    feats += [e_mid/tot, e_high/tot, e_high/(e_low+1e-6),
              float(f.max()/(f.mean()+1e-6))]

    for proj in [gray.mean(axis=0), gray.mean(axis=1)]:
        proj = proj - proj.mean()
        ps   = np.abs(np.fft.rfft(proj))[1:]
        feats.append(float(ps.max()/(ps.mean()+1e-6)))

    mid      = gray[H//2, :]
    residual = mid - np.convolve(mid, np.ones(5)/5, mode='same')
    acf      = np.correlate(residual, residual, mode='full')[len(residual)-1:]
    acf      = acf / (acf[0] + 1e-6)
    feats   += [float(acf[2:25].max()), float(np.std(residual)), float(np.mean(np.abs(residual)))]

    b_r = [abs(float(gray[r].mean()-gray[r-1].mean())) for r in range(8, H-1, 8)]
    i_r = [abs(float(gray[r].mean()-gray[r-1].mean())) for r in range(9, H-1) if r%8]
    feats.append(np.mean(b_r)/(np.mean(i_r)+1e-6) if b_r and i_r else 1.0)
    b_c = [abs(float(gray[:,c].mean()-gray[:,c-1].mean())) for c in range(8, W-1, 8)]
    i_c = [abs(float(gray[:,c].mean()-gray[:,c-1].mean())) for c in range(9, W-1) if c%8]
    feats.append(np.mean(b_c)/(np.mean(i_c)+1e-6) if b_c and i_c else 1.0)

    gy    = (gray[1:,:]-gray[:-1,:])[:,:W-1]
    gx    = (gray[:,1:]-gray[:,:W-1])[:H-1,:]
    angle = np.degrees(np.arctan2(gy, gx)) % 180
    hv    = float(((angle<12)|(angle>168)|((angle>78)&(angle<102))).mean())
    hist, _ = np.histogram(angle.ravel(), bins=8, range=(0,180), density=True)
    hist    = hist/(hist.sum()+1e-6)
    feats  += [hv, float(-np.sum(hist*np.log(hist+1e-9)))]

    hsv = np.asarray(Image.fromarray(crop.astype(np.uint8)).convert("HSV")).astype(np.float32)
    sat = hsv[:,:,1]/255
    feats += [float(sat.mean()), float(np.percentile(sat,75)), float((sat>0.5).mean())]

    lum = gray/255
    hist_b, _ = np.histogram(lum.ravel(), bins=16, range=(0,1), density=True)
    hist_b    = hist_b/(hist_b.sum()+1e-6)
    feats    += [float((lum<0.10).mean()),
                 float(-np.sum(hist_b*np.log(hist_b+1e-9)))]

    ps2  = 15
    stds = [float(gray[r:r+ps2, c:c+ps2].std())
            for r in range(0, H-ps2, ps2) for c in range(0, W-ps2, ps2)]
    stds = np.array(stds)
    feats += [float(stds.mean()), float(np.percentile(stds, 10))]

    return np.array(feats, dtype=np.float64)


def predict(image_path: str) -> float:
    model = _load_model()
    img   = Image.open(image_path)
    img.draft("RGB", (IMG_SIZE, IMG_SIZE))
    img   = img.convert("RGB")
    x     = _TRANSFORM(img).unsqueeze(0)
    with torch.inference_mode():
        cnn_score = float(torch.softmax(_model_cache(x), dim=1)[0, 1].item())
    feats = np.array([cnn_score] + list(_extract_features(image_path)), dtype=np.float64)
    x_sc  = (feats - _FEAT_MEAN) / (_FEAT_SCALE + 1e-9)
    logit = float(np.dot(x_sc, _FEAT_COEF) + _FEAT_BIAS)
    return 1.0 / (1.0 + np.exp(-logit))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_cnn.py <image_path>", file=sys.stderr)
        sys.exit(1)
    print(round(predict(sys.argv[1]), 4))