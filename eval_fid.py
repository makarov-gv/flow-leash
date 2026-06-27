import json
import os
import torch
from torch.utils.data import DataLoader

from config import parse_args
from data import Images
from flow import heun_sample
from metrics import Inception, fid
from models import VAE
from sit import SiT
from utils import get_device, load_ckpt, set_seed


def load_gen(path, device):
    ck = load_ckpt(path)
    m = ck["meta"]
    model = SiT(m["preset"], in_channels=m["in_channels"], input_size=m["input_size"],
                patch_size=m["patch_size"], num_classes=m["num_classes"])
    model.load_state_dict(ck.get("ema", ck["model"]))
    return model.to(device).eval(), m


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
    gen, meta = load_gen(cfg.eval.gen_ckpt, device)
    in_ch, hw, num_classes = meta["in_channels"], meta["input_size"], meta["num_classes"]

    incep = Inception().to(device).eval()
    n = cfg.eval.get("n_samples", 50000)
    bs = cfg.eval.get("batch_size", 64)
    steps = cfg.eval.get("steps", 50)
    cfg_scale = cfg.eval.get("cfg_scale", 1.0)

    real_feats = []
    ds = Images(cfg.data.root, cfg.data.image_size, train=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=cfg.get("num_workers", 8))
    seen = 0
    for imgs, _ in dl:
        real_feats.append(incep(imgs.to(device)).cpu().numpy())
        seen += imgs.shape[0]
        if seen >= n:
            break

    fake_feats = []
    done = 0
    use_cfg = cfg_scale > 1.0 and num_classes > 0
    while done < n:
        b = min(bs, n - done)
        if use_cfg:
            y = torch.randint(0, num_classes, (b,), device=device)
            yn = torch.full((b,), num_classes, device=device, dtype=torch.long)
            z = heun_sample(gen, (2 * b, in_ch, hw, hw), device, steps,
                            y=torch.cat([y, yn]), cfg_scale=cfg_scale)[:b]
        else:
            y = torch.randint(0, num_classes, (b,), device=device) if num_classes > 0 else None
            z = heun_sample(gen, (b, in_ch, hw, hw), device, steps, y=y)
        imgs = vae.decode_scaled(z).clamp(-1, 1)
        fake_feats.append(incep(imgs).cpu().numpy())
        done += b

    import numpy as np
    score = fid(np.concatenate(real_feats), np.concatenate(fake_feats))
    print("gFID", round(score, 3))
    out = cfg.eval.get("out_json", os.path.join(os.path.dirname(cfg.eval.gen_ckpt), "fid.json"))
    json.dump({"gfid": score, "n": done, "steps": steps, "cfg_scale": cfg_scale}, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
