"""Store trained LoRA adapters inside a compressed dir as int8 + per-row fp16 scales (counted by the size rule), merge at restore.

  python -m compress_lib.lora_io --comp DIR --adapters adapters.pt --out DIR2
Stored per adapted matrix <name>.weight:  .lora_A (int8 [r,in]) .lora_As (fp16 [r]) .lora_B (int8 [out,r]) .lora_Bs (fp16 [out])
Restore: W = dequant(Q) @ unrotate + scale * (B_dq @ A_dq), computed in fp32, then cast to bf16.
"""
import argparse, json, shutil
from pathlib import Path
import torch
from safetensors.torch import save_file
from .model import read_all


def q8_rows(X: torch.Tensor):
    s = (X.abs().amax(dim=1) / 127.0).clamp_min(1e-12).half()
    return torch.clamp(torch.round(X / s.float().unsqueeze(1)), -127, 127).to(torch.int8), s


def attach(comp: str, adapters: str, out: str):
    comp, out = Path(comp), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ad = torch.load(adapters, map_location="cpu")
    tensors = dict(read_all(comp))
    scale = None
    for name, d in ad.items():
        assert name + ".rans" in tensors, f"adapter {name} has no quantized base weight in {comp}"
        A, As = q8_rows(d["A"].float()); B, Bs = q8_rows(d["B"].float())
        tensors[name + ".lora_A"], tensors[name + ".lora_As"] = A, As
        tensors[name + ".lora_B"], tensors[name + ".lora_Bs"] = B, Bs
        assert scale is None or abs(scale - d["scale"]) < 1e-9
        scale = d["scale"]
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
    for f in comp.iterdir():
        if f.name not in ("model.safetensors", "comp_manifest.json") and f.is_file():
            shutil.copy(f, out / f.name)
    m = json.loads((comp / "comp_manifest.json").read_text()); m["lora_scale"] = scale
    (out / "comp_manifest.json").write_text(json.dumps(m))


