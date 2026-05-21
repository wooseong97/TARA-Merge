import ipdb  # noqa: F401
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModelWithProjection

from utils import get_cliphead


class ViTClipClassifier(nn.Module):
    """Classification model with CLIP"""

    def __init__(self, cliphead_path, model_id="openai/clip-vit-base-patch32"):
        super().__init__()

        # Load pre-trained vision model of CLIP
        self.vit = CLIPVisionModelWithProjection.from_pretrained(model_id, use_safetensors=True)
        self.base_id = model_id
        class_vector = get_cliphead(cliphead_path)
        self.register_buffer("class_vector", class_vector)

    def forward(self, x):
        # If we have a buggy return from processors, fix it
        if isinstance(x, torch.Tensor):
            x = {"pixel_values": x}
        if len(x["pixel_values"].shape) == 5:
            x["pixel_values"] = x["pixel_values"].squeeze(1)

        encodings = self.vit.vision_model(**x).pooler_output
        encodings = self.vit.visual_projection(encodings)
        # Normalize the encodings
        encodings = F.normalize(encodings, p=2, dim=-1)
        logits = encodings @ self.class_vector.T
        return logits


class ClassificationLoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, pred, target):
        # Classification does not require additional loss components
        return self.loss_fn(pred, target), None, None


def compute_classification_metrics(pred, target):
    """
    Computes top-k accuracy, precision, recall, and F1-score.

    Args:
        pred: logits tensor of shape (batch_size, num_classes)
        target: tensor of true class indices (batch_size,)

    Returns:
        dict with accuracy@k, precision, recall, f1
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    with torch.no_grad():
        pred_labels = pred.argmax(dim=1).cpu().numpy()
        target_labels = target.cpu().numpy()

        precision, recall, f1, _ = precision_recall_fscore_support(
            target_labels,
            pred_labels,
            average="macro",
            zero_division=0,  # type: ignore
        )

        acc = accuracy_score(target_labels, pred_labels)

        # For future use, compute the shannon entropy of the predictions
        pred_probs = F.softmax(pred, dim=1)
        entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-10), dim=1).mean().item()

        metrics = {
            "top1_acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "entropy": entropy,
        }

    return metrics


def compute_classification_metrics_joint(logits: torch.Tensor, target: torch.Tensor, ks=(1, 3, 5)):
    """
    Computes top-k accuracy for joint classification using pure PyTorch.

    Args:
        logits: (N, C) tensor of unnormalized scores.
        target: (N,) tensor of true class indices (long).
        ks: iterable of k values, e.g., (1,3,5)

    Returns:
        dict: {"top1_acc": float, "top3_acc": float, "top5_acc": float}
    """
    with torch.inference_mode():
        assert logits.ndim == 2, f"logits must be (N,C), got {logits.shape}"
        assert target.ndim == 1 and target.shape[0] == logits.shape[0], (
            f"target must be (N,), got {target.shape} vs logits {logits.shape}"
        )

        N, C = logits.shape
        if N == 0:
            return {f"top{k}_acc": float("nan") for k in ks}

        max_k = min(max(ks), C)
        # Get top-k indices once
        topk_idx = logits.topk(max_k, dim=1).indices  # (N, max_k)

        # Compare each row's top-k with its target
        hits = topk_idx.eq(target.view(-1, 1))  # (N, max_k), bool

        metrics = {}
        for k in ks:
            k_eff = min(k, C)  # guard when C < k
            acc = hits[:, :k_eff].any(dim=1).float().mean().item()
            metrics[f"top{k}_acc"] = acc

    return metrics


if __name__ == "__main__":
    # Example usage
    # --- 1. Load CLIP model and processor ---
    model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(model_name)

    # --- 2. Prepare class vectors from text ---
    class_names = ["a photo of a cat", "a photo of a dog"]
    text_inputs = processor(text=class_names, return_tensors="pt", padding=True)
    with torch.no_grad():
        class_vector = clip_model.get_text_features(**text_inputs)

    # --- 3. Load and preprocess a real image ---
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt")

    # --- 4. Instantiate your model ---
    model = ViTClipClassifier(model_id=model_name, class_vector=class_vector)
    model.eval()

    # --- 5. Run forward pass ---
    with torch.no_grad():
        logits = model(inputs)

    # --- 6. Output predicted class ---
    pred_class = class_names[logits.argmax().item()]
    print("Predicted class:", pred_class)
