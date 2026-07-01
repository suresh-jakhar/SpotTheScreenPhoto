import os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from predict import _extract_features, predict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

def get_images(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in IMG_EXTS] if os.path.isdir(folder) else []

if __name__ == "__main__":
    dirs = [("dataset/real_pics", 0), ("dataset/screen_pics", 1)]
    paths, labels = [], []
    for d, lbl in dirs:
        for p in get_images(d):
            paths.append(p); labels.append(lbl)

    print(f"Extracting features from {len(paths)} images...")
    X, y = [], []
    for i, (path, label) in enumerate(zip(paths, labels)):
        try:
            X.append([predict(path)] + list(_extract_features(path)))
            y.append(label)
        except Exception as e:
            print(f"Skipped {path}: {e}")

    X, y = np.array(X, dtype=np.float64), np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42)
    clf.fit(X_scaled, y)

    cv = cross_val_score(clf, X_scaled, y, cv=5)
    print(f"5-fold CV: {cv.mean()*100:.1f}% | Train Acc: {clf.score(X_scaled, y)*100:.1f}%")

    fmt = lambda arr: "np.array([" + ", ".join(f"{v:.6f}" for v in arr) + "])"
    print(f"\n_FEAT_MEAN  = {fmt(scaler.mean_)}")
    print(f"_FEAT_SCALE = {fmt(scaler.scale_)}")
    print(f"_FEAT_COEF  = {fmt(clf.coef_[0])}")
    print(f"_FEAT_BIAS  = {clf.intercept_[0]:.6f}")
