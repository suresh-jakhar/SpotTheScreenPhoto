import argparse, os, copy, time, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE, NORM = 224, ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def list_images(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".heic", ".webp"}] if os.path.isdir(folder) else []

class PhotoDataset(Dataset):
    def __init__(self, paths, labels, augment=True):
        self.paths, self.labels = paths, labels
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(0.25, 0.25, 0.15, 0.05),
            transforms.RandomRotation(12),
            transforms.ToTensor(),
            transforms.Normalize(*NORM),
        ] if augment else [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(*NORM),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB")), self.labels[idx]

def build_model():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.features.parameters(): p.requires_grad = False
    m.classifier[1] = nn.Linear(m.last_channel, 2)
    return m

def evaluate(model, ldr, device):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for imgs, lbls in ldr:
            correct += (model(imgs.to(device)).argmax(1) == lbls.to(device)).sum().item()
            n += len(imgs)
    return correct / n

def fit(model, tr_ldr, val_ldr, epochs, unfreeze_at, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=unfreeze_at)
    best_acc, best_state = 0.0, None
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(epochs):
        t0 = time.perf_counter()
        if ep == unfreeze_at:
            for p in model.features.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - unfreeze_at)

        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for imgs, lbls in tr_ldr:
            imgs, lbls = imgs.to(device), lbls.to(device).long()
            opt.zero_grad()
            out = model(imgs); loss = criterion(out, lbls)
            loss.backward(); opt.step()
            loss_sum += loss.item() * len(imgs); correct += (out.argmax(1) == lbls).sum().item(); n += len(imgs)
        sch.step()

        train_loss, train_acc = loss_sum / n, correct / n
        val_acc = evaluate(model, val_ldr, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  ep {ep+1:2d}/{epochs}  loss={train_loss:.4f}  train={train_acc*100:5.1f}%  val={val_acc*100:5.1f}%  {int(time.perf_counter()-t0)}s" + (" *" if val_acc > best_acc else ""))
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())

    if best_state: model.load_state_dict(best_state)
    return model, best_acc, history

def save_plots(history):
    try:
        import matplotlib.pyplot as plt
        eps = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1); plt.plot(eps, history["train_loss"], 'r-'); plt.title('Loss'); plt.grid(True)
        plt.subplot(1, 2, 2); plt.plot(eps, [x*100 for x in history["train_acc"]], 'b-', label='Train')
        plt.plot(eps, [x*100 for x in history["val_acc"]], 'g-', label='Val')
        plt.title('Accuracy (%)'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig('training_curves.png')
        print("Saved training_curves.png")
    except ImportError: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--unfreeze_at", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_p, scr_p = list_images("dataset/real_pics"), list_images("dataset/screen_pics")
    paths, labels = np.array(real_p + scr_p), np.array([0]*len(real_p) + [1]*len(scr_p))
    print(f"Device: {device} | {len(paths)} images ({len(real_p)} real / {len(scr_p)} screen)")

    tr_p, val_p, tr_l, val_l = train_test_split(paths, labels, test_size=0.15, stratify=labels, random_state=args.seed)
    
    loader = lambda p, l, aug, shuf: DataLoader(PhotoDataset(p, l, aug), batch_size=args.batch_size, shuffle=shuf, num_workers=0)
    model, best_val, history = fit(build_model().to(device), loader(tr_p, tr_l, True, True), loader(val_p, val_l, False, False), args.epochs, args.unfreeze_at, device)
    
    print(f"Best val accuracy: {best_val*100:.1f}%")
    save_plots(history)
    torch.save(model.state_dict(), "cnn_model.pt")
    print("Saved best checkpoint to cnn_model.pt")

if __name__ == "__main__":
    main()