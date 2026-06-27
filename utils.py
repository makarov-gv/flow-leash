import json, os, random, time
import numpy as np
import torch


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_dtype(device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


class EMA:
    def __init__(self, model, decay=0.9999):
        import copy
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.model.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, p in zip(self.model.buffers(), model.buffers()):
            s.copy_(p)


def save_ckpt(path, **obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = {k: (v.state_dict() if hasattr(v, "state_dict") else v) for k, v in obj.items()}
    torch.save(out, path)


def load_ckpt(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


class Logger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.t0 = time.time()

    def log(self, step, **m):
        rec = {"step": step, "t": round(time.time() - self.t0, 1)}
        rec.update({k: float(v) for k, v in m.items()})
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in rec.items()))
