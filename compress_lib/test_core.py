import torch, numpy as np, time
from compress_lib.core import *
torch.manual_seed(0)
for n in (2560, 4096, 9216):
    W = torch.randn(64, n); nm = f"layer.{n}.w"
    Wp = rotate_in(W, nm); R = unrotate_in(Wp, nm)
    print(n, "roundtrip err", (R - W).abs().max().item(), "norm ratio", (Wp.norm() / W.norm()).item())
# Hessian rotation consistency: tr(dW H dW^T) invariant
n = 2560; X = torch.randn(4000, n) @ torch.randn(n, n) * 0.05; H = X.T @ X / 4000
W = torch.randn(128, n) * 0.02; nm = "t"
Wp = rotate_in(W, nm); Hp = rotate_hessian(H, nm)
dW = torch.randn_like(W) * 1e-3
print("loss invariance", ((dW @ H) * dW).sum().item(), ((rotate_in(dW, nm) @ Hp) * rotate_in(dW, nm)).sum().item())
for bits in (3.67, 1.76):
    a = alpha_for_rate(Wp, bits); d = Wp.std(1) * a
    t = time.time(); Q = gptq_uniform(Wp, Hp, d); tg = time.time() - t
    Qr = gptq_uniform(Wp, None, d)
    def loss(Q): dW = Wp - Q.float() * d[:, None]; return ((dW @ Hp) * dW).sum().item()
    print(f"bits {bits} alpha {a:.3f} entropy gptq {entropy_bits(Q):.3f} rtn {entropy_bits(Qr):.3f} loss gptq {loss(Q):.4g} rtn {loss(Qr):.4g}  ({tg:.1f}s)")
    s, c, mn = rans_encode(Q.numpy()); y = rans_decode(s, c, mn, Q.numel())
    print("  rans bits/sym", s.size * 8 / Q.numel(), "roundtrip", (y.reshape(Q.shape) == Q.numpy()).all())
