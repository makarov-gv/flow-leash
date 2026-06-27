import argparse, copy
from pathlib import Path
import yaml


class Config(dict):
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _cast(s):
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for f in (int, float):
        try:
            return f(s)
        except ValueError:
            pass
    return s


def load_config(path, overrides=None):
    path = Path(path)
    raw = yaml.safe_load(open(path)) or {}
    base = raw.pop("base", None)
    if base:
        raw = _merge(load_config(path.parent / base), raw)
    cfg = Config(raw)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, Config())
        node[parts[-1]] = _cast(val)
    return cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--set", "-s", nargs="*", default=[])
    a = p.parse_args()
    return load_config(a.config, a.set)
