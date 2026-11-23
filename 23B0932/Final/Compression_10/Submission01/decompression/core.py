"""Core quantization primitives: random-sign block Hadamard, GPTQ with a uniform entropy-coded grid, rANS codec.

Everything here is deterministic given (name, shapes, stored tensors) so restore needs no side information
beyond the stored streams / scales / counts.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import torch

# ---------------------------------------------------------------- rotation ----------------------------------------------


def _pow2_block(n: int, cap: int = 4096) -> int:
    b = 1
    while n % (b * 2) == 0 and b * 2 <= cap:
        b *= 2
    return b


def _hadamard(b: int, device, dtype=torch.float32) -> torch.Tensor:
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < b:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(b)


def rotation_params(name: str, n: int):
    """(perm, signs) for an input dimension n, seeded by tensor name only."""
    g = np.random.default_rng(zlib.crc32(name.encode()))
    perm = torch.from_numpy(g.permutation(n))
    signs = torch.from_numpy(g.integers(0, 2, size=n) * 2 - 1).float()
    return perm, signs


def rotate_in(W: torch.Tensor, name: str) -> torch.Tensor:
    """W' = W R, R orthogonal on the input (last) dim: permute, random sign, block-Hadamard. float32 in/out."""
    n = W.shape[-1]
    perm, signs = rotation_params(name, n)
    b = _pow2_block(n)
    H = _hadamard(b, W.device)
    X = (W[..., perm.to(W.device)] * signs.to(W.device)).reshape(*W.shape[:-1], n // b, b)
    return (X @ H).reshape(W.shape)


def unrotate_in(Wp: torch.Tensor, name: str) -> torch.Tensor:
    """Exact inverse of rotate_in (R^T)."""
    n = Wp.shape[-1]
    perm, signs = rotation_params(name, n)
    b = _pow2_block(n)
    H = _hadamard(b, Wp.device)
    X = (Wp.reshape(*Wp.shape[:-1], n // b, b) @ H.T).reshape(Wp.shape) * signs.to(Wp.device)
    out = torch.empty_like(X)
    out[..., perm.to(Wp.device)] = X
    return out


def rotate_hessian(Hs: torch.Tensor, name: str) -> torch.Tensor:
    """H' = R^T H R for the layer-input second-moment matrix H (n x n)."""
    return rotate_in(rotate_in(Hs, name).T.contiguous(), name).T.contiguous()


# --------------------------------------------------------------------- GPTQ ---------------------------------------------


def gptq_uniform(Wp: torch.Tensor, Hp: torch.Tensor | None, delta: torch.Tensor, qmax: int = 4095,
                 blocksize: int = 128, damp: float = 0.01):
    """Quantize rotated weights W' (out x in) with per-row step `delta` (out,) and GPTQ error feedback.

    Returns integer codes (out x in, int32) such that W'_hat = codes * delta[:, None].
    Hp=None -> plain round-to-nearest.
    """
    W = Wp.float().clone()
    out_f, in_f = W.shape
    d = delta.float().unsqueeze(1)
    if Hp is None:
        return torch.clamp(torch.round(W / d), -qmax, qmax).to(torch.int32)
    H0 = Hp.double().clone()
    H0 = (H0 + H0.T) / 2  # rotation in fp32 can leave tiny asymmetry
    dead = torch.diag(H0) == 0
    H0[dead, dead] = 1.0
    W[:, dead] = 0
    # Cholesky in float64; if the (very ill-conditioned) Hessian still fails, escalate the damping x10 and log it.
    Hinv = None
    for k in range(6):
        dk = damp * (10 ** k)
        H = H0 + dk * torch.mean(torch.diag(H0)) * torch.eye(in_f, device=H0.device, dtype=H0.dtype)
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True).float()
            if k:
                print(f"  [gptq] damping escalated to {dk:g}", flush=True)
            break
        except torch.linalg.LinAlgError:
            continue
    if Hinv is None:
        raise RuntimeError("GPTQ: Hessian not positive definite even with 1e3x damping")
    Q = torch.zeros_like(W, dtype=torch.int32)
    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        W1 = W[:, i1:i2].clone()
        Err = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            q = torch.clamp(torch.round(w / d[:, 0]), -qmax, qmax)
            Q[:, i1 + i] = q.to(torch.int32)
            err = (w - q * d[:, 0]) / Hi[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hi[i, i:].unsqueeze(0)
            Err[:, i] = err
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q


def entropy_bits(codes: torch.Tensor) -> float:
    """Empirical zeroth-order entropy in bits/symbol of integer codes (single table per matrix)."""
    c = codes.flatten().long()
    c = c - c.min()
    cnt = torch.bincount(c).double()
    p = cnt[cnt > 0] / cnt.sum()
    return float(-(p * torch.log2(p)).sum())


def rtn_entropy(Wp: torch.Tensor, alpha: float) -> float:
    d = Wp.float().std(dim=1, keepdim=True).clamp_min(1e-8) * alpha
    return entropy_bits(torch.round(Wp.float() / d))


def alpha_for_rate(Wp: torch.Tensor, target_bits: float, lo: float = 0.02, hi: float = 20.0, iters: int = 30) -> float:
    """Bisect the step multiplier alpha (step = alpha * row_std) so RTN entropy == target_bits."""
    sub = Wp[:: max(1, Wp.shape[0] // 512)]
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if rtn_entropy(sub, mid) > target_bits:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# --------------------------------------------------------------------- rANS ---------------------------------------------


def _probs_from_counts(cnt: np.ndarray) -> np.ndarray:
    cnt = cnt.astype(np.float64)
    cnt = np.maximum(cnt, 0.0)
    return (cnt / cnt.sum()).astype(np.float32)


def rans_encode(codes: np.ndarray):
    """codes: int array (any shape). Returns (stream uint8, counts int32 over [min..max], minsym int)."""
    x = codes.reshape(-1).astype(np.int64)
    mn = int(x.min())
    sym = (x - mn).astype(np.int32)
    cnt = np.bincount(sym).astype(np.int64)
    # zero-count symbols inside the range need nonzero prob for Categorical(perfect=False): give them count 1 later
    cnt_model = np.maximum(cnt, 1)
    import constriction
    model = constriction.stream.model.Categorical(_probs_from_counts(cnt_model), perfect=False)
    enc = constriction.stream.stack.AnsCoder()
    enc.encode_reverse(sym, model)
    words = enc.get_compressed()  # uint32
    stream = np.frombuffer(words.tobytes(), dtype=np.uint8).copy()
    return stream, cnt_model.astype(np.int32), mn


def rans_decode(stream: np.ndarray, cnt: np.ndarray, minsym: int, n: int) -> np.ndarray:
    import constriction
    model = constriction.stream.model.Categorical(_probs_from_counts(cnt), perfect=False)
    words = np.frombuffer(stream.tobytes(), dtype=np.uint32).copy()
    dec = constriction.stream.stack.AnsCoder(words)
    sym = dec.decode(model, n)
    return sym.astype(np.int32) + minsym
