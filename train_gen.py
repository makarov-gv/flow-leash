import os
import torch
from torch.utils.data import DataLoader

from config import parse_args
from data import Latents
from flow import fm_loss
from sit import SiT
from utils import EMA, Logger, amp_dtype, get_device, save_ckpt, set_seed


def main():
    cfg = parse_args()
    set_seed(cfg.get("seed", 0))
    device = get_device(cfg.get("device", "auto"))
    os.makedirs(cfg.io.out_dir, exist_ok=True)
    log = Logger(os.path.join(cfg.io.out_dir, "log.jsonl"))

    ds = Latents(cfg.data.latent_dir)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                    num_workers=cfg.get("num_workers", 8), drop_last=True, pin_memory=True)
    z0, _ = ds[0]
    in_ch, hw = z0.shape[0], z0.shape[-1]

    model = SiT(cfg.sit.preset, in_channels=in_ch, input_size=hw,
                patch_size=cfg.sit.get("patch_size", 2),
                num_classes=cfg.sit.get("num_classes", 1000),
                class_dropout=cfg.sit.get("class_dropout", 0.1)).to(device)
    meta = dict(preset=cfg.sit.preset, patch_size=cfg.sit.get("patch_size", 2),
                num_classes=cfg.sit.get("num_classes", 1000), in_channels=in_ch, input_size=hw)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, betas=(0.9, 0.95),
                            weight_decay=cfg.train.get("weight_decay", 0.0))
    ema = EMA(model, cfg.train.get("ema_decay", 0.9999)) if cfg.train.get("use_ema", True) else None
    adt = amp_dtype(device) if cfg.train.get("amp", True) else None
    scaler = torch.cuda.amp.GradScaler(enabled=adt == torch.float16)
    use_label = cfg.sit.get("num_classes", 1000) > 0
    t_dist = cfg.train.get("t_dist", "uniform")

    step, max_steps = 0, cfg.train.max_steps
    model.train()
    while step < max_steps:
        for z, y in dl:
            z = z.to(device)
            y = y.to(device) if use_label else None
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=adt, enabled=adt is not None):
                loss = fm_loss(model, z, y, t_dist)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            if ema:
                ema.update(model)
            if step % cfg.train.get("log_every", 100) == 0:
                log.log(step, loss=loss.item())
            step += 1
            if step >= max_steps:
                break
    save_ckpt(os.path.join(cfg.io.out_dir, "model.pt"),
              model=model, ema=ema.model if ema else model, meta=meta, step=step)
    print("done")


if __name__ == "__main__":
    main()
