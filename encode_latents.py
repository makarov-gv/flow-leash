import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import parse_args
from data import Images
from models import VAE
from utils import get_device, set_seed, load_ckpt


def main():
    cfg = parse_args()
    set_seed(cfg.get("seed", 0))
    device = get_device(cfg.get("device", "auto"))
    out_dir = cfg.io.out_dir
    os.makedirs(out_dir, exist_ok=True)

    vae = VAE(cfg.vae.pretrained, cfg.vae.get("scaling_factor")).to(device).eval()
    ckpt = cfg.vae.get("checkpoint")
    if ckpt and os.path.exists(ckpt):
        vae.load_state_dict(load_ckpt(ckpt)["vae"], strict=False)
        print("loaded", ckpt)

    ds = Images(cfg.data.root, cfg.data.image_size, train=False)
    dl = DataLoader(ds, batch_size=cfg.io.get("batch_size", 32), shuffle=False,
                    num_workers=cfg.get("num_workers", 8))

    shard_size = cfg.io.get("shard_size", 5000)
    zs, ys, shard, total = [], [], 0, 0
    for imgs, labels in dl:
        with torch.no_grad():
            z = vae.encode_scaled(imgs.to(device), sample=False)
        zs.append(z.half().cpu().numpy())
        ys.append(np.asarray(labels))
        if sum(a.shape[0] for a in zs) >= shard_size:
            _save(out_dir, shard, zs, ys)
            total += sum(a.shape[0] for a in zs)
            shard += 1
            zs, ys = [], []
            print("shard", shard, "total", total)
    if zs:
        _save(out_dir, shard, zs, ys)
        total += sum(a.shape[0] for a in zs)
    print("done", total)


def _save(out_dir, idx, zs, ys):
    np.savez(os.path.join(out_dir, f"shard_{idx:05d}.npz"),
             z=np.concatenate(zs, 0), y=np.concatenate(ys, 0))


if __name__ == "__main__":
    main()
