import torch, json
from safetensors import safe_open
from compress_lib.model import _dequant_matrix
from compress_lib.lora_io import q8_rows
R = "/mnt/nas/tejaschaudhari/emir_data/ckpt/B2_v0/model.safetensors"
name = "model.language_model.layers.0.mlp.up_proj.weight"
t = {}
with safe_open(R, "pt") as f:
    for suf in (".rans", ".cnt", ".scale", ".meta"): t[suf] = f.get_tensor(name + suf)
W0 = _dequant_matrix(t, name, "cpu")
torch.manual_seed(0); r = 16
A = torch.randn(r, 2560) * 0.02; B = torch.randn(9216, r) * 0.02; sc = 1.0
Aq, As = q8_rows(A); Bq, Bs = q8_rows(B)
Ad = Aq.float() * As.float()[:, None]; Bd = Bq.float() * Bs.float()[:, None]
merged = W0 + sc * (Bd @ Ad)
exact = W0 + sc * (B @ A)
print("int8 LoRA delta rel err", ((merged - exact).norm() / (sc * (B @ A)).norm()).item())
print("stored bits", (Aq.numel() + Bq.numel()) * 8 + (As.numel() + Bs.numel()) * 16)
