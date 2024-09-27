from albumentations import Compose, Normalize
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(use_norm=False):
    ts = []
    ts.append(ToTensorV2())
    if use_norm:
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ts.append(norm)
    return Compose(ts)
