import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import parse_args
from data import Images
from losses import LPIPS
from metrics import Inception, fid, latent_stats
from models import VAE
from utils import get_device, load_ckpt, set_seed


def psnr(a, b):
    mse = torch.mean((a.clamp(-1, 1) - b.clamp(-1, 1)) ** 2, dim=[1, 2, 3])
    return (10 * torch.log10(4.0 / (mse + 1e-12))).mean().item()


@torch.no_grad()
def main():
    cfg = parse_args()
    set_seed(cfg.get("seed", 0))
    device = get_device(cfg.get("device", "auto"))

    vae = VAE(cfg.vae.pretrained, cfg.vae.get("scaling_factor")).to(device).eval()
    if cfg.eval.get("vae_ckpt"):
        st = load_ckpt(cfg.eval.vae_ckpt)
        vae.load_state_dict(st["vae"], strict=False)
        vae.scaling_factor = st.get("scaling_factor", vae.scaling_factor)

    lpips = LPIPS(cfg.loss.get("lpips_net", "vgg")).to(device)
    incep = Inception().to(device).eval()
    ds = Images(cfg.data.root, cfg.data.image_size, train=False)
    dl = DataLoader(ds, batch_size=cfg.eval.get("batch_size", 32), shuffle=False,
                    num_workers=cfg.get("num_workers", 8))

    n = cfg.eval.get("n_recon", 50000)
    real_f, recon_f, psnrs, lps, stats, seen = [], [], [], [], [], 0
    for imgs, _ in dl:
        imgs = imgs.to(device)
        z = vae.encode(imgs).mean
        recon = vae.decode(z).clamp(-1, 1)
        psnrs.append(psnr(recon, imgs))
        lps.append(lpips(recon, imgs).item())
        real_f.append(incep(imgs).cpu().numpy())
        recon_f.append(incep(recon).cpu().numpy())
        stats.append(latent_stats(z * vae.scaling_factor))
        seen += imgs.shape[0]
        if seen >= n:
            break

    res = {"psnr": float(np.mean(psnrs)), "lpips": float(np.mean(lps)),
           "rfid": fid(np.concatenate(real_f), np.concatenate(recon_f)),
           "var_min": float(np.mean([s["var_min"] for s in stats])),
           "eff_rank": float(np.mean([s["eff_rank"] for s in stats])), "n": seen}
    print(json.dumps(res, indent=2))
    out = cfg.eval.get("recon_json", os.path.join(os.path.dirname(cfg.eval.vae_ckpt or "."), "recon.json"))
    json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
