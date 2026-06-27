import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = nn.Identity()
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        x = (x.clamp(-1, 1) + 1) * 0.5
        x = F.interpolate(x, size=299, mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        return self.net(x)


def _sqrtm(A, iters=80):
    A = A.astype(np.float64)
    norm = np.linalg.norm(A)
    Y = A / norm
    I = np.eye(A.shape[0])
    Z = np.eye(A.shape[0])
    for _ in range(iters):
        T = 0.5 * (3 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Y * np.sqrt(norm)


def fid(feats_a, feats_b):
    mu1, mu2 = feats_a.mean(0), feats_b.mean(0)
    s1, s2 = np.cov(feats_a, rowvar=False), np.cov(feats_b, rowvar=False)
    try:
        from scipy import linalg
        cov, _ = linalg.sqrtm(s1 @ s2, disp=False)
        cov = cov.real if np.iscomplexobj(cov) else cov
    except Exception:
        cov = _sqrtm(s1 @ s2)
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(cov))


@torch.no_grad()
def latent_stats(z):
    C = z.shape[1]
    flat = z.transpose(0, 1).reshape(C, -1)
    var = flat.var(dim=1, unbiased=False)
    tv = (z[..., 1:, :] - z[..., :-1, :]).abs().mean() + (z[..., :, 1:] - z[..., :, :-1]).abs().mean()
    eig = torch.linalg.eigvalsh(torch.cov(flat)).clamp(min=0)
    eff_rank = (eig.sum() ** 2) / (eig.pow(2).sum() + 1e-12)
    return {"var_mean": var.mean().item(), "var_min": var.min().item(),
            "tv": tv.item(), "eff_rank": eff_rank.item()}
