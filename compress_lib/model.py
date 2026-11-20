"""Checkpoint-level compress / restore for Qwen3.5-4B (text tower).

Compressed format (a normal HF-style directory, tensors in safetensors, size = sum numel*itemsize outside `visual`):
  for each quantized body matrix  <name>.weight :
      <name>.weight.rans   uint8   rANS stream
      <name>.weight.cnt    int32   symbol counts (the entropy model)
      <name>.weight.scale  float16 per-row step (rotated domain)
      <name>.weight.meta   int32   [rows, cols, minsym]
  embedding (tied lm_head), three tiers:
      embed.tier   uint8  2-bit packed tier id per row (0=T1 ecsq hi-rate, 1=T2 ecsq lo-rate, 2=T3 centroid)
      embed.t{1,2}.{rans,cnt,scale,meta}; embed.t3.centroids bf16 [K, d]; embed.t3.idx int16 [n3]
  every other non-visual, non-mtp tensor is stored unchanged. `visual.*` tensors are stored unchanged (free, zeroed by TAs).
  mtp.* tensors are dropped; comp_manifest.json (not a tensor) records their shapes so restore writes zeros.
Restore needs ONLY the compressed directory (no model_name, no downloads).
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .core import (alpha_for_rate, entropy_bits, gptq_uniform, rans_decode, rans_encode, rotate_hessian, rotate_in,
                   unrotate_in)

LM = "model.language_model."
BODY_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.(mlp\.(gate_proj|up_proj|down_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj))\.weight$")
EMBED = "model.language_model.embed_tokens.weight"


def hessian_key(name: str) -> str:
    m = BODY_RE.match(name)
    L = int(m.group(1))
    leaf = name.split(".")[-2]
    if leaf in ("gate_proj", "up_proj"):
        return f"L{L}.mlp_in"
    if leaf == "down_proj":
        return f"L{L}.down_in"
    if leaf in ("out_proj", "o_proj"):
        return f"L{L}.o_in"
    return f"L{L}.attn_in"  # q,k,v,in_proj_qkv,in_proj_z


def group_of(name: str) -> str:
    """Component group used by the bit plan: attn / dnet / ffn_gu / ffn_down x layer-band."""
    m = BODY_RE.match(name)
    L = int(m.group(1))
    band = "a" if L < 8 else ("b" if L < 24 else "c")
    leaf = name.split(".")[-2]
    if leaf in ("q_proj", "k_proj", "v_proj", "o_proj"):
        kind = "attn"
    elif leaf in ("in_proj_qkv", "in_proj_z", "out_proj"):
        kind = "dnet"
    elif leaf == "down_proj":
        kind = "ffn_down"
    else:
        kind = "ffn_gu"
    return f"{kind}.{band}"


def read_all(model_dir: Path):
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                yield k, sf.get_tensor(k)


def _quant_matrix(W: torch.Tensor, name: str, bits: float, H: torch.Tensor | None, dev: str, refine: bool = True):
    """Rotate, choose step for `bits`, GPTQ, rANS. Returns dict of tensors + stats."""
    Wp = rotate_in(W.to(dev).float(), name)
    Hp = rotate_hessian(H.to(dev).float(), name) if H is not None else None
    std = Wp.std(dim=1).clamp_min(1e-6)
    alpha = alpha_for_rate(Wp, bits)
    for it in range(2 if (refine and Hp is not None) else 1):
        delta = (std * alpha).clamp_min(1e-4).half().float()  # fp16-exact step, so GPTQ sees what restore will use
        Q = gptq_uniform(Wp, Hp, delta)
        eb = entropy_bits(Q)
        if abs(eb - bits) < 0.02 or Hp is None:
            break
        alpha *= 2 ** (eb - bits)  # entropy ~ -log2(step)
    delta16 = delta.half()
    assert torch.equal(delta16.float(), delta)
    Wq = Q.float() * delta16.float().unsqueeze(1)
    Wr = unrotate_in(Wq, name)
    rel = ((Wr - W.to(dev).float()).norm() / W.to(dev).float().norm()).item()
    stream, cnt, mn = rans_encode(Q.cpu().numpy())
    out = {
        ".rans": torch.from_numpy(stream),
        ".cnt": torch.from_numpy(cnt),
        ".scale": delta16.cpu(),
        ".meta": torch.tensor([W.shape[0], W.shape[1], mn], dtype=torch.int32),
    }
    return out, dict(bits=stream.size * 8 / W.numel(), entropy=eb, rel_err=rel)


def _dequant_matrix(t: dict, name: str, dev: str) -> torch.Tensor:
    rows, cols, mn = [int(x) for x in t[".meta"]]
    Q = rans_decode(t[".rans"].numpy(), t[".cnt"].numpy(), mn, rows * cols).reshape(rows, cols)
    Wq = torch.from_numpy(Q).to(dev).float() * t[".scale"].to(dev).float().unsqueeze(1)
    return unrotate_in(Wq, name)


# ------------------------------------------------------------------- embedding ---------------------------------------


def _pack2(x: np.ndarray) -> np.ndarray:
    pad = (-len(x)) % 4
    x = np.concatenate([x, np.zeros(pad, dtype=np.uint8)]).reshape(-1, 4)
    return (x[:, 0] | (x[:, 1] << 2) | (x[:, 2] << 4) | (x[:, 3] << 6)).astype(np.uint8)


def _unpack2(p: np.ndarray, n: int) -> np.ndarray:
    return np.stack([(p >> s) & 3 for s in (0, 2, 4, 6)], 1).reshape(-1)[:n]


def _kmeans(X: torch.Tensor, K: int, iters: int = 15, seed: int = 0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:K]].clone()
    for _ in range(iters):
        idx = torch.cat([torch.cdist(c, C).argmin(1) for c in X.split(8192)])
        S = torch.zeros_like(C).index_add_(0, idx, X)
        n = torch.bincount(idx, minlength=K).clamp_min(1).unsqueeze(1).float()
        newC = S / n
        empty = torch.bincount(idx, minlength=K) == 0
        newC[empty] = C[empty]
        C = newC
    idx = torch.cat([torch.cdist(c, C).argmin(1) for c in X.split(8192)])
    return C, idx


def compress_embedding(E: torch.Tensor, counts: np.ndarray, n1: int, n2: int, b1: float, b2: float, K: int, dev: str,
                       special_ids: list[int]):
    V, d = E.shape
    order = np.argsort(-counts.astype(np.float64), kind="stable")
    # special / template tokens always go to tier 1
    sp = set(special_ids)
    order = np.array([i for i in order if i in sp] + [i for i in order if i not in sp])
    tier = np.full(V, 2, dtype=np.uint8)
    tier[order[:n1]] = 0
    tier[order[n1:n1 + n2]] = 1
    out = {"embed.tier": torch.from_numpy(_pack2(tier))}
    stats = {}
    for t, bits in ((0, b1), (1, b2)):
        rows = np.where(tier == t)[0]
        name = f"embed.t{t + 1}"
        q, st = _quant_matrix(E[rows], name + ".weight", bits, None, dev)
        for k, v in q.items():
            out[name + k] = v
        stats[name] = st
    rows3 = np.where(tier == 2)[0]
    X = E[rows3].to(dev).float()
    C, idx = _kmeans(X, K)
    out["embed.t3.centroids"] = C.bfloat16().cpu()
    out["embed.t3.idx"] = idx.to(torch.int16).cpu()
    stats["embed.t3"] = dict(rel_err=((C[idx] - X).norm() / X.norm()).item())
    return out, stats


def restore_embedding(t: dict, V: int, d: int, dev: str) -> torch.Tensor:
    tier = _unpack2(t["embed.tier"].numpy(), V)
    E = torch.zeros(V, d, dtype=torch.bfloat16)
    for ti in (0, 1):
        rows = np.where(tier == ti)[0]
        name = f"embed.t{ti + 1}"
        sub = {k[len(name):]: v for k, v in t.items() if k.startswith(name + ".") and k[len(name):] in (".rans", ".cnt", ".scale", ".meta")}
        E[torch.from_numpy(rows)] = _dequant_matrix(sub, name + ".weight", dev).bfloat16().cpu()
    rows3 = np.where(tier == 2)[0]
    idx = t["embed.t3.idx"].to(torch.int32) & 0xFFFF
    E[torch.from_numpy(rows3)] = t["embed.t3.centroids"][idx.long()]
    return E


# ------------------------------------------------------------------- top level ---------------------------------------


def compress_model(model_dir: str, out_dir: str, plan: dict, hessians: dict | None, tok_counts: np.ndarray | None, dev: str = "cuda",
                   log=print):
    """plan: {"bits": {group: bits}, "embed": {"n1":..,"n2":..,"b1":..,"b2":..,"K":..}}"""
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    manifest = {"dropped": {}, "embed_shape": None, "plan": plan}
    stats = {}
    t0 = time.time()
    for name, W in read_all(model_dir):
        if name.startswith("mtp."):
            manifest["dropped"][name] = [list(W.shape), str(W.dtype).replace("torch.", "")]
        elif BODY_RE.match(name):
            g = group_of(name)
            bits = plan["bits"][g]
            H = hessians[hessian_key(name)] if hessians is not None else None
            q, st = _quant_matrix(W, name, bits, H, dev)
            for k, v in q.items():
                tensors[name + k] = v
            st["group"] = g
            stats[name] = st
            log(f"[{time.time() - t0:6.0f}s] {name} {g} target {bits} -> {st['bits']:.3f} bits, rel_err {st['rel_err']:.4f}")
        elif name == EMBED:
            manifest["embed_shape"] = list(W.shape)
            e = plan["embed"]
            sp = list(range(248044 - 0, 248070))  # special / chat-template token ids region (filtered below)
            if tok_counts is None:
                raise ValueError("embedding tiering needs token counts (--counts); refusing to tier by arbitrary order")
            counts = tok_counts
            q, st = compress_embedding(W, counts, e["n1"], e["n2"], e["b1"], e["b2"], e["K"], dev, [i for i in sp if i < W.shape[0]])
            tensors.update(q)
            stats["embed"] = st
            log(f"[{time.time() - t0:6.0f}s] embed {st}")
        else:
            tensors[name] = W.contiguous()
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    for f in model_dir.iterdir():
        if f.suffix in (".json", ".jinja", ".txt") and f.name != "model.safetensors.index.json":
            shutil.copy(f, out_dir / f.name)
    manifest["stats"] = stats
    (out_dir / "comp_manifest.json").write_text(json.dumps(manifest))
    return stats


def restore_model(comp_dir: str, out_dir: str, dev: str = "cuda", log=print):
    comp_dir, out_dir = Path(comp_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((comp_dir / "comp_manifest.json").read_text())
    raw = dict(read_all(comp_dir))
    out: dict[str, torch.Tensor] = {}
    groups: dict[str, dict] = {}
    for k, v in raw.items():
        m = re.match(r"^(.*\.weight)(\.rans|\.cnt|\.scale|\.meta)$", k)
        if m and not k.startswith("embed."):
            groups.setdefault(m.group(1), {})[m.group(2)] = v
        elif not k.startswith("embed."):
            out[k] = v
    for name, t in groups.items():
        out[name] = _dequant_matrix(t, name, dev).bfloat16().cpu()
    V, d = manifest["embed_shape"]
    out[EMBED] = restore_embedding({k: v for k, v in raw.items() if k.startswith("embed.")}, V, d, dev)
    for name, (shape, dt) in manifest["dropped"].items():
        out[name] = torch.zeros(shape, dtype=getattr(torch, dt))
    # shard like HF (single file is fine for vLLM); keep names identical to the original checkpoint
    save_file({k: v.contiguous() for k, v in out.items()}, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    for f in comp_dir.iterdir():
        if f.suffix in (".json", ".jinja", ".txt") and f.name != "comp_manifest.json":
            shutil.copy(f, out_dir / f.name)
    return out
