import os
import torch
from torch.utils.data import DataLoader

from config import parse_args
from data import Images
from losses import Adapter, DiffusionCritic, LPIPS, recon_loss, vf_loss
from metrics import latent_stats
from models import DINO, VAE
from sit import SiT
from utils import Logger, amp_dtype, get_device, load_ckpt, save_ckpt, set_seed


def base_variance(vae, dl, device, n_batches):
    vals = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(dl):
            z = vae.encode_scaled(imgs.to(device), sample=False)
            vals.append(z.transpose(0, 1).reshape(z.shape[1], -1).var(dim=1, unbiased=False))
            if i + 1 >= n_batches:
                break
    return torch.stack(vals).mean(0)


def load_critic(path, device):
    ck = load_ckpt(path)
    m = ck["meta"]
    critic = SiT(m["preset"], in_channels=m["in_channels"], input_size=m["input_size"],
                 patch_size=m["patch_size"], num_classes=m["num_classes"])
    critic.load_state_dict(ck["ema"])
    return critic.to(device).eval()


def main():
    cfg = parse_args()
    set_seed(cfg.get("seed", 0))
    device = get_device(cfg.get("device", "auto"))
    os.makedirs(cfg.io.out_dir, exist_ok=True)
    log = Logger(os.path.join(cfg.io.out_dir, "log.jsonl"))

    vfm_w = cfg.loss.get("vfm_weight", 0.0)
    crit_w = cfg.loss.get("critic_weight", 0.0)
    print(f"vfm_weight={vfm_w} critic_weight={crit_w}")

    ds = Images(cfg.data.root, cfg.data.image_size, train=True)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                    num_workers=cfg.get("num_workers", 8), drop_last=True, pin_memory=True)

    vae = VAE(cfg.vae.pretrained, cfg.vae.get("scaling_factor")).to(device)
    vae.set_trainable(encoder=True, decoder=cfg.vae.get("train_decoder", False))
    if cfg.vae.get("grad_checkpoint", False):
        vae.enable_gradient_checkpointing()
    sf = vae.scaling_factor
    params = list(vae.trainable_parameters())

    lpips = LPIPS(cfg.loss.get("lpips_net", "vgg")).to(device) if cfg.loss.get("lpips_weight", 1.0) > 0 else None

    dino = adapter = None
    if vfm_w > 0:
        dino = DINO(cfg.vfm.get("name", "dinov2_vitb14"), cfg.vfm.get("img_size", 224)).to(device)
        adapter = Adapter(vae.latent_channels, dino.embed_dim).to(device)
        params += list(adapter.parameters())

    critic = None
    if crit_w > 0:
        bvar = base_variance(vae, dl, device, cfg.critic.get("var_batches", 8))
        target_var = cfg.critic.get("var_floor_ratio", 0.5) * bvar.min().item()
        critic = DiffusionCritic(
            load_critic(cfg.critic.checkpoint, device),
            t_lo=cfg.critic.get("t_lo", 0.1), t_hi=cfg.critic.get("t_hi", 0.6),
            weight_scheme=cfg.critic.get("weight_scheme", "snr"),
            target_var=target_var, var_anchor_weight=cfg.critic.get("var_anchor_weight", 1.0),
            grad_clip=cfg.critic.get("grad_clip_value", 50.0)).to(device)
        print(f"critic loaded, var_floor={target_var:.3f}")

    opt = torch.optim.AdamW(params, lr=cfg.train.lr, betas=(0.9, 0.95),
                            weight_decay=cfg.train.get("weight_decay", 0.0))
    adt = amp_dtype(device) if cfg.train.get("amp", True) else None
    scaler = torch.cuda.amp.GradScaler(enabled=adt == torch.float16)
    kl_w = cfg.loss.get("kl_weight", 1e-6)
    warmup = cfg.critic.get("warmup_steps", 2000) if crit_w > 0 else 0

    step, max_steps = 0, cfg.train.max_steps
    vae.train()
    while step < max_steps:
        for imgs, _ in dl:
            imgs = imgs.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=adt, enabled=adt is not None):
                dist = vae.encode(imgs)
                zu = dist.sample()
                recon = vae.decode(zu)
                z = zu * sf
                loss = recon_loss(recon, imgs, lpips, cfg.loss.get("l1_weight", 1.0),
                                  cfg.loss.get("lpips_weight", 1.0)) + kl_w * dist.kl()
                rec = {"recon": loss.item()}
                if vfm_w > 0:
                    loss = loss + vfm_w * vf_loss(z, dino(imgs), adapter,
                                                  cfg.vfm.get("w_cos", 1.0), cfg.vfm.get("w_dist", 1.0))
                if crit_w > 0:
                    cw = crit_w * min(1.0, (step + 1) / max(1, warmup))
                    cl, cval, anchor = critic(z)
                    loss = loss + cw * cl
                    rec["critic"] = cval.item()
                    rec["anchor"] = anchor.item()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.train.get("grad_clip", 1.0))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.train.get("grad_clip", 1.0))
                opt.step()
            if step % cfg.train.get("log_every", 100) == 0:
                st = latent_stats(z.detach())
                log.log(step, total=loss.item(), var_min=st["var_min"], eff_rank=st["eff_rank"], **rec)
            step += 1
            if step >= max_steps:
                break
    save_ckpt(os.path.join(cfg.io.out_dir, "vae.pt"),
              vae=vae, adapter=adapter if adapter is not None else {}, scaling_factor=sf, step=step)
    print("done")


if __name__ == "__main__":
    main()
