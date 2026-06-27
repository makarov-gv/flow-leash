import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbed(nn.Module):
    def __init__(self, dim, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        return self.mlp(timestep_embedding(t * 1000.0, self.freq_dim))


class LabelEmbed(nn.Module):
    def __init__(self, num_classes, dim, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.table = nn.Embedding(num_classes + 1, dim)

    def forward(self, y, train, force_drop=None):
        if (train and self.dropout > 0) or force_drop is not None:
            drop = force_drop if force_drop is not None else \
                (torch.rand(y.shape[0], device=y.device) < self.dropout)
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        return self.table(y)


def pos_embed_2d(dim, grid, device=None):
    omega = torch.arange(dim // 4, dtype=torch.float32, device=device) / (dim // 4)
    omega = 1.0 / (10000.0 ** omega)
    coords = torch.arange(grid, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    out = []
    for g in (gx.reshape(-1), gy.reshape(-1)):
        a = g[:, None] * omega[None]
        out += [torch.sin(a), torch.cos(a)]
    return torch.cat(out, dim=1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        sa1, ss1, g1, sa2, ss2, g2 = self.ada(c).chunk(6, dim=1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), sa1, ss1))
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sa2, ss2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch * patch * out_ch)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


PRESETS = {
    "Ti": dict(depth=6, dim=192, heads=3),
    "S": dict(depth=12, dim=384, heads=6),
    "B": dict(depth=12, dim=768, heads=12),
    "L": dict(depth=24, dim=1024, heads=16),
}


class SiT(nn.Module):
    def __init__(self, preset="S", in_channels=4, input_size=32, patch_size=2,
                 num_classes=1000, class_dropout=0.1):
        super().__init__()
        p = PRESETS[preset]
        dim, depth, heads = p["dim"], p["depth"], p["heads"]
        self.in_channels = in_channels
        self.patch = patch_size
        self.grid = input_size // patch_size
        self.num_classes = num_classes
        self.use_label = num_classes > 0

        self.x_embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        self.t_embed = TimeEmbed(dim)
        if self.use_label:
            self.y_embed = LabelEmbed(num_classes, dim, class_dropout)
        self.register_buffer("pos", pos_embed_2d(dim, self.grid).unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.final = FinalLayer(dim, patch_size, in_channels)
        self._init()

    def _init(self):
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(basic)
        nn.init.normal_(self.x_embed.weight, std=0.02)
        nn.init.zeros_(self.x_embed.bias)
        for b in self.blocks:
            nn.init.zeros_(b.ada[-1].weight)
            nn.init.zeros_(b.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def unpatchify(self, x):
        B = x.shape[0]
        x = x.reshape(B, self.grid, self.grid, self.patch, self.patch, self.in_channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(B, self.in_channels, self.grid * self.patch, self.grid * self.patch)

    def forward(self, x, t, y=None, force_drop=None):
        h = self.x_embed(x).flatten(2).transpose(1, 2) + self.pos
        c = self.t_embed(t)
        if self.use_label:
            if y is None:
                y = torch.full((x.shape[0],), self.num_classes, device=x.device, dtype=torch.long)
            c = c + self.y_embed(y, self.training, force_drop)
        for b in self.blocks:
            h = b(h, c)
        return self.unpatchify(self.final(h, c))

    def forward_cfg(self, x, t, y, scale):
        half = x[: len(x) // 2]
        v = self.forward(torch.cat([half, half], 0), t, y)
        cond, uncond = v.chunk(2, 0)
        guided = uncond + scale * (cond - uncond)
        return torch.cat([guided, guided], 0)
