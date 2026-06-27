import glob, os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def make_transform(size, train=True):
    ops = [transforms.Resize(size), transforms.CenterCrop(size)]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    return transforms.Compose(ops)


class Images(Dataset):
    def __init__(self, root, size=256, train=True):
        self.ds = ImageFolder(root, transform=make_transform(size, train))
        self.num_classes = len(self.ds.classes)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]


class Latents(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not self.files:
            raise FileNotFoundError(root)
        self.index = []
        for fi, f in enumerate(self.files):
            with np.load(f, mmap_mode="r") as d:
                self.index += [(fi, r) for r in range(d["z"].shape[0])]
        self.cache = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, r = self.index[i]
        if fi not in self.cache:
            self.cache = {fi: np.load(self.files[fi])}
        d = self.cache[fi]
        z = torch.from_numpy(np.asarray(d["z"][r])).float()
        y = int(d["y"][r]) if "y" in d else 0
        return z, y
