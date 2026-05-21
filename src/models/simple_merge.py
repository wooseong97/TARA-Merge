import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from peft.utils import _get_submodules
from transformers import (
    AutoModel,
    CLIPVisionModelWithProjection,
    ViTModel,
)
from utils import get_base_model, get_criterion, load_lora_model

from .merging_utils import DiagonalLinear, LinearTrainable, LoraModelTrainableMerging

MERGE_TYPES = ["weight", "tara-a", "tara-b"]


def _merge_type_to_attr(merge_type: str) -> str:
    """Map CLI merge_type string to the corresponding LoraWrapper method suffix."""
    return merge_type.replace("-", "_")


class SimpleMergeModel(nn.Module):
    def __init__(
        self,
        cfgs,
        model_dirs,
        lora_dicts,
        preference,
        merge_type="weight",
        scaling_coeff=1.0,
        **kwargs,
    ):
        """
        SimpleMergeModel is a model that merges multiple LoRA adapters into a single model.
        Args:
            lora_dict (dict): A dictionary mapping task names to LoRA adapter paths.
            preference (torch.Tensor): A tensor representing the preference for each task.
            merge_type (str): The type of merging to perform. Options are:
                - "weight": Merge with a single coefficient per adapter (AdaMerging-style).
                - "tara-a": TARA Variant A — per-rank LoRA direction selection.
                - "tara-b": TARA Variant B — shared singular-direction selection (SVD basis).
        """
        super().__init__()
        assert merge_type in MERGE_TYPES, f"Unknown merge type: {merge_type}"
        self.prefix = "lora_"
        self.backbone_name = "merged_adapter"
        self.scaled_name = "scaled_adapter"
        self.cfgs = cfgs
        _models = [
            load_lora_model(
                get_base_model(cfg),
                lora_dicts[f"{cfg.data.name}-{cfg.task.name}"],
                os.path.join(model_dir, cfg.get("decoder_save_name", "")),
            )
            for cfg, model_dir in zip(cfgs, model_dirs)
        ]
        self.lora_dicts = lora_dicts
        self.scaling_coeff = scaling_coeff
        self.preference = preference
        self.merge_type = merge_type

        self.model_id = _model_id = _models[0].base_id
        assert all(model.base_id == _model_id for model in _models), (
            "Models must have the same pretrained model ID for merging."
        )
        _vit = self._get_multi_lora_model(_model_id, lora_dicts)

        # Compute adapter-wise number of coefficients and ranks.
        self.adapter_wise_num_coeffs = self._compute_adapter_wise_num_coeffs(_vit.peft_config)
        self.adapter_wise_rank = self._compute_adapter_wise_rank(_vit.peft_config)

        _vit_wrapper = LoraModelTrainableMerging(_vit)
        # Merge the LoRA adapters into a single model and share it with all task models.
        self.vit = self.merge(_vit_wrapper)

        _vit_wrapper.add_trainable_adapter(
            self.backbone_name,
            self.scaled_name,
            self.adapter_wise_num_coeffs,
            self.adapter_wise_rank,
            self.scaling_coeff,
            self.merge_type,
        )

        for model in _models:
            model.vit = self.vit  # type: ignore
        self.models = nn.ModuleList(_models)

        # Merged model does not require gradient updates for the parameters except for the
        # coefficients.
        self.requires_grad_(False)
        key_list = [key for key, _ in self.vit.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.vit.model, key)
            if isinstance(target, LinearTrainable):
                sub_target = target.lora_A[self.scaled_name][1]  # type: ignore
                if isinstance(sub_target, DiagonalLinear):
                    sub_target.requires_grad_(True)  # type: ignore

    def _compute_adapter_wise_rank(self, lora_configs):
        """
        Per-adapter rank for use in coefficient construction.
        """
        task_rank = [lora_configs[key].r for key in self.lora_dicts.keys()]
        return task_rank

    def _compute_adapter_wise_num_coeffs(self, lora_configs, merge_type=None):
        """
        Number of coefficients per adapter based on merge type.
            - "weight": 1 coefficient per adapter.
            - "tara-a": rank coefficients per adapter (Variant A).
            - "tara-b": total_rank coefficients per adapter (Variant B).
        """
        if merge_type is None:
            merge_type = self.merge_type
        task_rank = self._compute_adapter_wise_rank(lora_configs)
        if merge_type == "weight":
            task_rank = [1 for _ in task_rank]
        elif merge_type == "tara-b":
            total_rank = sum(task_rank)
            task_rank = [total_rank for _ in task_rank]
        return task_rank

    def _get_multi_lora_model(self, model_type, lora_dict):
        base_model = get_vit_from_id(model_type)
        first_task = list(lora_dict.keys())[0]
        first_lora_path = lora_dict[first_task]
        if "unseen" in self.cfgs[0] and self.cfgs[0].unseen:
            raise ValueError("Please load the LoRA adapters directly for the first task.")
        model = PeftModel.from_pretrained(base_model, first_lora_path, first_task)
        for task_name, lora_path, cfg in zip(
            list(lora_dict.keys())[1:], list(lora_dict.values())[1:], self.cfgs[1:]
        ):
            if "unseen" in cfg and cfg.unseen:
                continue
            model.load_adapter(lora_path, adapter_name=task_name)

        for lora_name in model.peft_config.keys():
            modules_to_save_status = model.peft_config[lora_name].modules_to_save
            if modules_to_save_status is not None:
                print(
                    f"Overriding modules_to_save for {lora_name} from {modules_to_save_status} to None"
                )
                model.peft_config[lora_name].modules_to_save = None  # type: ignore
        return model

    def merge(self, _vit_wrapper, merge_type=None):
        if merge_type is None:
            merge_type = self.merge_type
        merge_method = "concat_adapters_" + _merge_type_to_attr(merge_type)
        merge_fn = getattr(_vit_wrapper, merge_method, None)
        if merge_fn is None:
            raise ValueError(
                f"Merge type '{merge_type}' is not supported by LoraModelTrainableMerging."
            )
        merge_fn(
            list(self.lora_dicts.keys()),
            adapter_name=self.backbone_name,
        )
        _vit_wrapper.set_adapter(self.backbone_name)
        return _vit_wrapper.lora_model

    def _forward_helper(self, model, batch, result_accum, **kwargs):
        if isinstance(batch, dict):
            out = model(**batch)
        else:
            out = model(batch)
        result_accum.append(out)

    def forward(self, inputs_list, **kwargs):
        result = []
        for idx, model in enumerate(self.models):
            batch = inputs_list[idx]
            self._forward_helper(model, batch, result, **kwargs)

        return result

    def coeffs(self):
        coeffs = {}
        key_list = [key for key, _ in self.vit.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.vit.model, key)
            if isinstance(target, LinearTrainable):
                sub_target = target.lora_A[self.scaled_name][1]  # type: ignore
                if isinstance(sub_target, DiagonalLinear):
                    coeffs[key] = sub_target._logit_to_coeff().detach().cpu().numpy()
        return coeffs

    def coeffs_without_detach(self):
        coeffs = {}
        key_list = [key for key, _ in self.vit.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.vit.model, key)
            if isinstance(target, LinearTrainable):
                sub_target = target.lora_A[self.scaled_name][1]  # type: ignore
                if isinstance(sub_target, DiagonalLinear):
                    coeffs[key] = sub_target._logit_to_coeff()
        return coeffs

    def probs(self):
        probs = {}
        key_list = [key for key, _ in self.vit.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.vit.model, key)
            if isinstance(target, LinearTrainable):
                sub_target = target.lora_A[self.scaled_name][1]  # type: ignore
                if isinstance(sub_target, DiagonalLinear):
                    probs[key] = sub_target._logit_to_prob()
        return probs

    def reg_loss(self, lambda_reg=10.0):
        """Effective-rank regularization on the merged delta."""

        def compute_erank(matrix):
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
            s = s / (s.sum() + 1e-10)
            entropy = -(s * torch.log(s + 1e-10)).sum()
            erank = torch.exp(entropy)
            return erank

        loss_list = []
        key_list = [key for key, _ in self.vit.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.vit.model, key)
            if isinstance(target, LinearTrainable):
                delta = target.get_delta_weight()
                rank = target.get_rank()
                erank = compute_erank(delta)
                loss_list.append(1.0 - (erank / rank))
        loss = torch.stack(loss_list).mean()
        return lambda_reg * loss


class STCHLoss(nn.Module):
    """
    Smooth Tchebycheff Scalarization (STCH) loss used by TARA for
    preference-weighted entropy pseudo-loss optimization.
    """

    def __init__(
        self,
        criterions_cfgs,
        baseline_losses,
        preference,
        alpha=1.0,
        epsilon=0,
        loss_type="stch",
    ):
        super().__init__()
        self.criterions = [get_criterion(cfg) for cfg in criterions_cfgs]
        self.baselines = baseline_losses
        self.preference = preference
        self.alpha = alpha
        self.epsilon = epsilon
        self.loss_type = loss_type

    def forward(self, pred_lists, target_lists=None):
        if not (len(pred_lists) == len(self.criterions) == len(self.baselines)):
            raise ValueError(
                "pred_lists, criterions, and baseline_losses must have the same length."
            )
        if self.loss_type == "stch":
            return self.forward_stch(pred_lists, target_lists)
        elif self.loss_type == "default":
            return self.forward_default(pred_lists, target_lists)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Supported types are 'stch' and 'default'."
            )

    def forward_default(self, pred_lists, target_lists):
        loss = torch.zeros(1, device=pred_lists[0].device)
        for idx, (preference, pred, target) in enumerate(
            zip(self.preference, pred_lists, target_lists)
        ):
            loss += preference * (self._get_score(pred, target, idx) + self.epsilon)
        return loss

    def forward_stch(self, pred_lists, target_lists):
        loss = torch.zeros(1, device=pred_lists[0].device)
        for idx, (preference, pred, target) in enumerate(
            zip(self.preference, pred_lists, target_lists)
        ):
            loss_term = torch.exp(
                (1 / self.alpha) * preference * (self._get_score(pred, target, idx) + self.epsilon)
            )
            loss += loss_term
        stch_loss = torch.log(loss) * self.alpha
        return stch_loss

    def _get_score(self, pred, target, idx):
        def entropy(pred):
            probs = F.softmax(pred, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            return entropy.mean()

        if target is None:
            return entropy(pred) - self.baselines[idx]
        else:
            return self.criterions[idx](pred, target)[0] - self.baselines[idx]


def get_vit_from_id(model_id):
    if "clip" in model_id.lower():
        return CLIPVisionModelWithProjection.from_pretrained(model_id, use_safetensors=True)
    elif "google/vit" in model_id.lower():
        return ViTModel.from_pretrained(model_id, use_safetensors=True)
    elif "llama" in model_id.lower():
        model = AutoModel.from_pretrained(
            model_id, return_dict=True, torch_dtype=torch.bfloat16, use_safetensors=True
        )
        model.config.use_cache = False
        model.config.pretrained_tp = 1
        return model
    else:
        raise ValueError(f"Unknown model id: {model_id}. Supported models are CLIP and ViT.")
