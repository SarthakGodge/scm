"""Dataset, transforms, model builder and checkpoint loader for the ResNet-18 classifier."""
import torch, torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms as T
from PIL import Image
from .config import IMG_SIZE

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def train_tf(size=IMG_SIZE, strong=False):
    if not strong:
        return T.Compose([T.RandomResizedCrop(size, scale=(0.78, 1.0)), T.RandomHorizontalFlip(),
                          T.ColorJitter(0.15, 0.15, 0.10), T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([T.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.75, 1.33)), T.RandomHorizontalFlip(),
                      T.TrivialAugmentWide(), T.ToTensor(), T.Normalize(MEAN, STD),
                      T.RandomErasing(p=0.25, scale=(0.02, 0.2))])


def eval_tf(size=IMG_SIZE, mode="crop"):
    """Deterministic (no random augmentation). mode='crop': Resize(size*8/7) + CenterCrop(size) (256->224);
    mode='squash': resize straight to size x size, ignoring aspect ratio."""
    if mode == "squash":
        return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([T.Resize(int(round(size * 256 / 224))), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


class GroceryDS(Dataset):
    def __init__(self, records, tf):
        self.records, self.tf = records, tf
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        p, y = self.records[i]
        return self.tf(Image.open(p).convert("RGB")), y


def build_model(num_classes, pretrained=True, arch="resnet18", dropout=0.0):
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif arch == "resnet50":   # optional comparison only; the project's primary model is ResNet-18
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        raise ValueError(f"unsupported arch {arch}")
    fc = nn.Linear(m.fc.in_features, num_classes)
    m.fc = nn.Sequential(nn.Dropout(dropout), fc) if dropout > 0 else fc
    return m


def load_checkpoint(path, device="cpu"):
    """Returns (model, class_names). Reads arch/dropout/img_size stored in the checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("state_dict", ck.get("model_state_dict"))      # also accepts the standalone train.py format
    if sd is None or "class_names" not in ck:
        raise ValueError("Checkpoint missing weights/'class_names'. Retrain with scripts/03_train_classifier.py")
    arch = ck.get("arch", ck.get("architecture", "resnet18"))
    m = build_model(len(ck["class_names"]), pretrained=False, arch=arch, dropout=ck.get("dropout", 0.0))
    m.load_state_dict(sd); m.to(device).eval()
    m.img_size = ck.get("img_size", ck.get("input_size", IMG_SIZE))
    m.eval_mode = ck.get("eval_mode", "crop")                  # preprocessing the model was validated with
    return m, ck["class_names"]
