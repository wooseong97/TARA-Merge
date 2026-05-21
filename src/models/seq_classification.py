import ipdb  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SequenceClassifier(nn.Module):
    """Classification model with CLIP"""

    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B", num_classes=3):
        super().__init__()
        self.num_classes = num_classes

        # Load pre-trained vision model of CLIP
        self.vit = AutoModel.from_pretrained(
            model_id,
            return_dict=True,
            torch_dtype=torch.bfloat16,
        )
        self.decoder = nn.Linear(self.vit.config.hidden_size, self.num_classes, bias=False)
        self.base_id = model_id

        # Basic configuration for the model
        self.vit.config.use_cache = False
        self.vit.config.pretraining_tp = 1

        self.vit.config.pad_token_id = 0
        self.pad_token_id = 0
        self.mask_class = None  # Mask class for classification, if needed

    def forward(self, *args, **kwargs):
        features = self.vit(*args, **kwargs).last_hidden_state
        pad_token_id = self.pad_token_id
        batch_size = features.shape[0]
        logits = self.decoder(features)

        input_ids = kwargs.get("input_ids", None)
        if pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(
                input_ids.shape[-1], device=logits.device, dtype=torch.int32
            )
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        return pooled_logits


class SequenceClassificationLoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, pred, target):
        # Classification does not require additional loss components
        return self.loss_fn(pred, target), None, None


def compute_sequence_classification_metrics(pred, target):
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
