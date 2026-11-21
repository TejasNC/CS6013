"""One-layer test: 2-stage RVQ (dim 8) vs entropy-coded scalar quantization at equal bits, rotated weights, MSE (and Hessian-free)."""
import sys, math, torch
from safetensors import safe_open
from pathlib import Path
from .core import rotate_in, rtn_entropy, alpha_for_rate, entropy_bits
M = Path(open("../scratch_model_path.txt").read().strip()); dev = "cuda"
def get(name):
    for f in sorted(M.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            if name in sf.keys(): return sf.get_tensor(name)
def kmeans(X, K, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:K]].clone()
    for _ in range(iters):
        idx = torch.cat([torch.cdist(c, C).argmin(1) for c in X.split(65536)])
        S = torch.zeros_like(C).index_add_(0, idx, X); n = torch.bincount(idx, minlength=K)
        C = torch.where((n > 0).unsqueeze(1), S / n.clamp_min(1).unsqueeze(1), C)
    idx = torch.cat([torch.cdist(c, C).argmin(1) for c in X.split(65536)])
    return C, idx
for lname in sys.argv[1:]:
    W = get(lname).to(dev).float(); name = lname
    Wp = rotate_in(W, name); s = Wp.std(dim=1, keepdim=True)
    X = (Wp / s).reshape(-1, 8)  # unit-variance rows, 8-d vectors
    tot = (X ** 2).sum().item()
    print(lname.split("layers.")[1], "shape", tuple(W.shape))
    for bits in (1.0, 1.5, 2.0):
        # scalar entropy-coded (uniform grid, RTN) at this entropy
        a = alpha_for_rate(Wp, bits); d = s * a
        Q = torch.round(Wp / d) * d; mse_s = ((Wp - Q) ** 2 / s ** 2).sum().item() / tot
        # RVQ: total bits per 8-vector = 8*bits  -> split in two stages
        tb = int(round(8 * bits)); res = {}
        for k1 in range(max(tb // 2 - 2, 2), min(tb // 2 + 3, tb - 1)):
            k2 = tb - k1
            C1, i1 = kmeans(X, 2 ** k1); R1 = X - C1[i1]
            C2, i2 = kmeans(R1, 2 ** k2); R2 = R1 - C2[i2]
            res[(k1, k2)] = (R2 ** 2).sum().item() / tot
        best = min(res, key=res.get)
        print(f"  {bits} b/w: scalar-entropy mse {mse_s:.4f} ({10*math.log10(1/mse_s):.2f} dB) | RVQ best split {best} mse {res[best]:.4f} ({10*math.log10(1/res[best]):.2f} dB) | gain {100*(1-res[best]/mse_s):.1f}% | all {{{', '.join(f'{k}:{v:.4f}' for k, v in res.items())}}}", flush=True)
