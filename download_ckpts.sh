#!/bin/bash
set -e

mkdir -p ./checkpoints/

hf download hoffman-lab/KnOTS-ViT-B-32_lora_R16_stanford_cars --local-dir ./checkpoints/stanford_cars_r16/adapter/
hf download hoffman-lab/KnOTS-ViT-B-32_lora_R16_dtd --local-dir ./checkpoints/dtd_r16/adapter/
hf download hoffman-lab/KnOTS-ViT-B-32_lora_R16_eurosat --local-dir ./checkpoints/eurosat_r16/adapter/
hf download hoffman-lab/KnOTS-ViT-B-32_lora_R16_gtsrb --local-dir ./checkpoints/gtsrb_r16/adapter/
hf download hoffman-lab/KnOTS-ViT-B-32_lora_R16_mnist --local-dir ./checkpoints/mnist_r16/adapter/
hf download hoffman-lab/KnOTS-ViT-B-32_lora_R16_resisc45 --local-dir ./checkpoints/resisc45_r16/adapter/
hf download hoffman-lab/KnOTS-ViT-B-32_lora_R16_sun397 --local-dir ./checkpoints/sun397_r16/adapter/
hf download hoffman-lab/KnOTS-ViT-B-32_lora_R16_svhn --local-dir ./checkpoints/svhn_r16/adapter/

python - <<'PY'
from pathlib import Path
import torch
from safetensors.torch import save_file

root = Path("./checkpoints")
for bin_path in root.glob("*/adapter/adapter_model.bin"):
    st_path = bin_path.with_suffix(".safetensors")
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    # KnOTS HF uploads were saved against CLIPVisionModel.encoder, but this
    # repo loads into CLIPVisionModelWithProjection.vision_model.encoder, so
    # PEFT silently fails to attach the LoRA without this prefix.
    state = {
        k.replace("base_model.model.", "base_model.model.vision_model.", 1): v.contiguous()
        for k, v in state.items()
    }
    save_file(state, st_path)
    bin_path.unlink()
    print(f"Converted {bin_path} -> {st_path}")
PY
