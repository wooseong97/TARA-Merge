import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor, SiglipModel

from src.utils import get_cliphead


class ViTSiglipClassifier(nn.Module):
    """Classification model with SigLIP"""

    def __init__(self, class_vector, model_id="google/siglip-base-patch16-224"):
        super().__init__()

        # Load pre-trained vision model of SigLIP
        # NOTE: SigLIP precision must be float32? I am not sure.
        full_model = AutoModel.from_pretrained(model_id, use_safetensors=True, dtype=torch.float32)
        self.vit = full_model.vision_model
        self._logit_scale = full_model.logit_scale
        self._logit_bias = full_model.logit_bias
        self.base_id = model_id
        if not isinstance(class_vector, torch.Tensor):
            class_vector = get_cliphead(class_vector)
        else:
            class_vector = class_vector.to(self.vit.config.dtype)
        self.register_buffer("class_vector", class_vector)

    def forward(self, x):
        # If we have a buggy return from processors, fix it
        if isinstance(x, torch.Tensor):
            x = {"pixel_values": x}
        if len(x["pixel_values"].shape) == 5:
            x["pixel_values"] = x["pixel_values"].squeeze(1)

        # encodings = self.vit.get_image_features(**x)
        encodings = self.vit(**x).pooler_output
        # Normalize the encodings
        encodings = F.normalize(encodings, p=2, dim=-1)
        logits = encodings @ self.class_vector.T
        logit_scale, logit_bias = (
            self._logit_scale.to(encodings.device),
            self._logit_bias.to(encodings.device),
        )
        logits = logits * logit_scale.exp() + logit_bias
        return logits


if __name__ == "__main__":
    # Example usage
    # --- 1. Load SigLIP model and processor ---
    model_name = "google/siglip-base-patch16-224"
    siglip_model = AutoModel.from_pretrained(model_name, use_safetensors=True, dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    # --- 2. Prepare class vectors from text ---
    class_names = ["a photo of a cat", "a photo of a dog"]
    text_inputs = processor(text=class_names, return_tensors="pt", padding=True)
    with torch.no_grad():
        class_vector = siglip_model.get_text_features(**text_inputs)

    # --- 3. Load and preprocess a real image ---
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt")

    # --- 4. Instantiate your model ---
    model = ViTSiglipClassifier(model_id=model_name, class_vector=class_vector)
    model.eval()

    # --- 5. Run forward pass ---
    with torch.no_grad():
        logits = model(inputs)

    # --- 6. Output predicted class ---
    pred_class = class_names[logits.argmax().item()]
    print("Overall Probs:", torch.softmax(logits, dim=-1))
    print("Predicted class:", pred_class)
