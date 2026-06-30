import argparse, os, copy, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

IMG_SIZE = 224
IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
NORM     = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def fmt_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s" if m else f"{s}s"

def list_images(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in IMG_EXTS]


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

def make_loader(paths, labels, augment, shuffle):
    return DataLoader(PhotoDataset(paths, labels, augment), batch_size=32, shuffle=shuffle, num_workers=0)

def evaluate(model, ldr, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, lbls in ldr:
            preds.extend(model(imgs.to(device)).argmax(1).cpu())
            trues.extend(lbls)
    return accuracy_score(trues, preds)

def fit(model, tr_ldr, val_ldr, epochs, unfreeze_at, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=unfreeze_at)
    best_acc, best_state = 0.0, None

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
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(imgs)
            correct  += (out.argmax(1) == lbls).sum().item()
            n        += len(imgs)
        sch.step()

        val_acc = evaluate(model, val_ldr, device) if val_ldr else None
        mark    = " *" if val_acc and val_acc > best_acc else ""
        val_str = f"  val={val_acc*100:5.1f}%" if val_acc else ""
        print(f"  ep {ep+1:2d}/{epochs}  loss={loss_sum/n:.4f}  train={correct/n*100:5.1f}%{val_str}  {fmt_time(time.perf_counter()-t0)}{mark}")

        if val_acc and val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())

    if best_state: model.load_state_dict(best_state)
    return model, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir",    default="dataset/real_pics")
    ap.add_argument("--screen_dir",  default="dataset/screen_pics")
    ap.add_argument("--epochs",      type=int, default=40)
    ap.add_argument("--unfreeze_at", type=int, default=3)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_p = list_images(args.real_dir)
    scr_p  = list_images(args.screen_dir)
    paths  = np.array(real_p + scr_p)
    labels = np.array([0]*len(real_p) + [1]*len(scr_p))
    print(f"Device: {device}  |  {len(paths)} images ({len(real_p)} real / {len(scr_p)} screen)")

    tr_p, val_p, tr_l, val_l = train_test_split(paths, labels, test_size=0.15, stratify=labels, random_state=args.seed)

    print(f"\nMonitor ({len(tr_p)} train / {len(val_p)} val):")
    _, best_val = fit(build_model().to(device),
                      make_loader(tr_p, tr_l, True, True), make_loader(val_p, val_l, False, False),
                      args.epochs, args.unfreeze_at, device)
    print(f"Best val: {best_val*100:.1f}%")

    print(f"\nFinal — all {len(paths)} images:")
    model, _ = fit(build_model().to(device), make_loader(paths, labels, True, True), None,
                   args.epochs, args.unfreeze_at, device)

    torch.save(model.state_dict(), "cnn_model.pt")
    print(f"Saved cnn_model.pt ({os.path.getsize('cnn_model.pt')/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()