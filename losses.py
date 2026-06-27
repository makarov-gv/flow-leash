import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPS(nn.Module):
    def __init__(self, net="vgg"):
        super().__init__()
        import lpips
        self.model = lpips.LPIPS(net=net)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def forward(self, x, y):
        return self.model(x.clamp(-1, 1), y.clamp(-1, 1)).mean()


def recon_loss(recon, target, lpips=None, l1_w=1.0, lpips_w=1.0):
    loss = l1_w * F.l1_loss(recon, target)
    if lpips is not None and lpips_w > 0:
        loss = loss + lpips_w * lpips(recon, target)
    return loss


class Adapter(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_dim, 1)

    def forward(self, z):
        return self.conv(z)


def vf_loss(z, feats, adapter, w_cos=1.0, w_dist=1.0):
    zp = adapter(z)
    h, w = zp.shape[-2:]
    f = F.interpolate(feats, size=(h, w), mode="bilinear", align_corners=False)
    zt = F.normalize(zp.flatten(2).transpose(1, 2), dim=-1)
    ft = F.normalize(f.flatten(2).transpose(1, 2), dim=-1)
    cos = (1.0 - (zt * ft).sum(-1)).mean()
    gz = torch.bmm(zt, zt.transpose(1, 2))
    gf = torch.bmm(ft, ft.transpose(1, 2))
    dist = (gz - gf).pow(2).mean()
    return w_cos * cos + w_dist * dist


def tau_weight(t, scheme):
    if scheme == "uniform":
        return torch.ones_like(t)
    if scheme == "snr":
        return t * (1 - t) / 0.25
    raise ValueError(scheme)


class DiffusionCritic(nn.Module):
    def __init__(self, critic, t_lo=0.1, t_hi=0.6, weight_scheme="snr",
                 target_var=None, var_anchor_weight=1.0, grad_clip=50.0):
        super().__init__()
        self.critic = critic
        self.t_lo, self.t_hi = t_lo, t_hi
        self.weight_scheme = weight_scheme
        self.target_var = target_var
        self.var_anchor_weight = var_anchor_weight
        self.grad_clip = grad_clip
        for p in self.critic.parameters():
            p.requires_grad_(False)
        self.critic.eval()

    def variance_penalty(self, z):
        if self.target_var is None:
            return z.new_zeros(())
        var = z.transpose(0, 1).reshape(z.shape[1], -1).var(dim=1, unbiased=False)
        return torch.relu(self.target_var - var).mean()

    def forward(self, z):
        t = self.t_lo + (self.t_hi - self.t_lo) * torch.rand(z.shape[0], device=z.device)
        tb = t.view(-1, *([1] * (z.dim() - 1)))
        noise = torch.randn_like(z)
        x_t = tb * z + (1.0 - tb) * noise
        target = (z - noise).clamp(-self.grad_clip, self.grad_clip)
        v = self.critic(x_t, t, None)
        w = tau_weight(t, self.weight_scheme).view(-1, *([1] * (z.dim() - 1)))
        crit = (w * (v - target) ** 2).mean()
        anchor = self.variance_penalty(z)
        return crit + self.var_anchor_weight * anchor, crit.detach(), anchor.detach()
