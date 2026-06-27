import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL


class LatentDist:
    def __init__(self, mean, logvar):
        self.mean = mean
        self.logvar = logvar.clamp(-30.0, 20.0)

    def sample(self):
        std = torch.exp(0.5 * self.logvar)
        return self.mean + std * torch.randn_like(self.mean)

    def kl(self):
        var = torch.exp(self.logvar)
        kl = 0.5 * (self.mean.pow(2) + var - 1.0 - self.logvar)
        return kl.flatten(1).sum(1).mean()


class VAE(nn.Module):
    def __init__(self, pretrained="stabilityai/sd-vae-ft-mse", scaling_factor=None):
        super().__init__()
        self.net = AutoencoderKL.from_pretrained(pretrained)
        self.scaling_factor = scaling_factor or float(self.net.config.scaling_factor)
        self.latent_channels = self.net.config.latent_channels

    def encode(self, x):
        post = self.net.encode(x).latent_dist
        return LatentDist(post.mean, post.logvar)

    def decode(self, z):
        return self.net.decode(z).sample

    def encode_scaled(self, x, sample=True):
        d = self.encode(x)
        z = d.sample() if sample else d.mean
        return z * self.scaling_factor

    def decode_scaled(self, z):
        return self.decode(z / self.scaling_factor)

    def set_trainable(self, encoder=True, decoder=False):
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.net.encoder.parameters():
            p.requires_grad_(encoder)
        for p in self.net.quant_conv.parameters():
            p.requires_grad_(encoder)
        for p in self.net.decoder.parameters():
            p.requires_grad_(decoder)
        for p in self.net.post_quant_conv.parameters():
            p.requires_grad_(decoder)
        return self

    def enable_gradient_checkpointing(self):
        self.net.enable_gradient_checkpointing()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


DINO_DIM = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}


class DINO(nn.Module):
    def __init__(self, name="dinov2_vitb14", img_size=224):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", name)
        self.embed_dim = DINO_DIM.get(name, 768)
        self.img_size = img_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        x = (x.clamp(-1, 1) + 1) * 0.5
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bicubic", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        feats = self.model.forward_features(x)["x_norm_patchtokens"]
        B, N, D = feats.shape
        g = int(N ** 0.5)
        return feats.transpose(1, 2).reshape(B, D, g, g)
