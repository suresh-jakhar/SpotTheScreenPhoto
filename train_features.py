import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from predict_cnn import _extract_features, predict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

def list_images(folder):
    if not os.path.isdir(folder): return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in IMG_EXTS]

def collect(dirs):
    paths, labels = [], []
    for d, label in dirs:
        for p in list_images(d):
            paths.append(p); labels.append(label)
    return paths, labels

if __name__ == "__main__":
    dirs = [
        ("dataset/real_pics",    0),
        ("dataset/screen_pics",  1),
    ]
    paths, labels = collect(dirs)
    print(f"Extracting features from {len(paths)} images...")

    X, y = [], []
    for i, (path, label) in enumerate(zip(paths, labels)):
        try:
            cnn_score = predict(path)
            feats     = _extract_features(path)
            X.append([cnn_score] + list(feats))
            y.append(label)
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{len(paths)}")
        except Exception as e:
            print(f"  Skipped {path}: {e}")

    X = np.array(X, dtype=np.float64)
    y = np.array(y)
    print(f"\nFeature matrix: {X.shape}  (images x features)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
    print(f"5-fold CV accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    train_acc = clf.score(X_scaled, y)
    print(f"Training accuracy:  {train_acc*100:.1f}%")

    def fmt(arr):
        return "np.array([" + ", ".join(f"{v:.6f}" for v in arr) + "])"

    print("\n" + "="*70)
    print("="*70)
    print(f"_FEAT_MEAN  = {fmt(scaler.mean_)}")
    print(f"_FEAT_SCALE = {fmt(scaler.scale_)}")
    print(f"_FEAT_COEF  = {fmt(clf.coef_[0])}")
    print(f"_FEAT_BIAS  = {clf.intercept_[0]:.6f}")
    print("="*70)
