import torch


def sample_t(n, device, dist="uniform"):
    if dist == "uniform":
        return torch.rand(n, device=device)
    if dist == "logitnormal":
        return torch.sigmoid(torch.randn(n, device=device))
    raise ValueError(dist)


def interpolate(x1, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x1)
    tb = t.view(-1, *([1] * (x1.dim() - 1)))
    x_t = tb * x1 + (1.0 - tb) * noise
    return x_t, x1 - noise, noise


def fm_loss(model, x1, y=None, t_dist="uniform"):
    t = sample_t(x1.shape[0], x1.device, t_dist)
    x_t, target, _ = interpolate(x1, t)
    return torch.mean((model(x_t, t, y) - target) ** 2)


@torch.no_grad()
def heun_sample(model, shape, device, steps=50, y=None, cfg_scale=1.0, dtype=torch.float32):
    x = torch.randn(shape, device=device, dtype=dtype)
    ts = torch.linspace(0, 1, steps + 1, device=device)
    use_cfg = cfg_scale > 1.0 and y is not None

    def vel(xc, tc):
        tt = tc.expand(xc.shape[0])
        return model.forward_cfg(xc, tt, y, cfg_scale) if use_cfg else model(xc, tt, y)

    for i in range(steps):
        dt = ts[i + 1] - ts[i]
        v1 = vel(x, ts[i])
        xp = x + dt * v1
        if i == steps - 1:
            x = xp
        else:
            x = x + dt * 0.5 * (v1 + vel(xp, ts[i + 1]))
    return x
