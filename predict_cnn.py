import sys
import os
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


def predict(image_path: str) -> float:
    model = _load_model()
    img   = Image.open(image_path)
    img.draft("RGB", (IMG_SIZE, IMG_SIZE))
    img   = img.convert("RGB")
    x     = _TRANSFORM(img).unsqueeze(0)
    with torch.inference_mode():
        probs = torch.softmax(_model_cache(x), dim=1)
    return float(probs[0, 1].item())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_cnn.py <image_path>", file=sys.stderr)
        sys.exit(1)
    print(round(predict(sys.argv[1]), 4))