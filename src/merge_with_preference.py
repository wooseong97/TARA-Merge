import argparse
import json
import os
from datetime import datetime
from importlib import import_module
from itertools import cycle

import numpy as np
import torch
import wandb
from baselines import retrieve_baselines
from models import SimpleMergeModel, STCHLoss
from models.merging_utils import merge_loras
from models.simple_merge import get_vit_from_id
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils import (
    get_base_model,
    get_criterion,
    get_dataloader,
    get_device,
    load_lora_model,
)

FILE_PATH = None
METRIC_MAP = {
    "classification": "top1_acc",
    "sequence_classification": "top1_acc",
}
CLIP_DATASET_NAMES = [
    "stanford_cars",
    "dtd",
    "eurosat",
    "gtsrb",
    "mnist",
    "resisc45",
    "sun397",
    "svhn",
]


def get_baseline_losses(baselines, cfgs):
    baseline_losses = [
        baseline["total_loss"] if "classification" not in cfg.task.name else baseline["entropy"]
        for cfg, baseline in zip(cfgs, baselines)
    ]
    return baseline_losses


def get_cfg_from_model_dir(model_dir):
    if os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "config.yaml")):
        return OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    elif os.path.isfile(model_dir) and model_dir.endswith(".yaml"):
        return OmegaConf.load(model_dir)
    else:
        raise ValueError(f"Invalid model_dir: {model_dir}")


def get_rep_metric(cfg):
    if cfg.task.name in ["classification", "sequence_classification"]:
        return "top1_acc"
    else:
        raise ValueError(
            f"Unsupported task for metric retrieval: {cfg.task.name} for dataset {cfg.data.name}"
        )


def get_validation_fn(cfg):
    if cfg.task.name in ["classification", "sequence_classification"]:
        utils = import_module("lora_vit_" + cfg.task.name)
        return utils.validate_epoch
    else:
        raise ValueError(f"Unsupported task for validation function retrieval: {cfg.task.name}")


def get_adapter_save_path(cfg, model_dir):
    if os.path.isfile(model_dir) and hasattr(cfg, "adapter_save_path"):
        return cfg.adapter_save_path
    elif os.path.isdir(model_dir):
        return os.path.join(model_dir, cfg.adapter_save_name)
    else:
        raise ValueError(f"Invalid model_dir: {model_dir}")


def get_decoder_save_path(cfg, model_dir):
    if os.path.isfile(model_dir):
        if hasattr(cfg, "decoder_save_path"):
            return cfg.decoder_save_path
        else:
            return None
    elif os.path.isdir(model_dir):
        decoder_save_name = cfg.get("decoder_save_name", None)
        if decoder_save_name is None:
            raise ValueError("decoder_save_name not found in cfg.")
        return os.path.join(model_dir, decoder_save_name)
    else:
        raise ValueError(f"Invalid model_dir: {model_dir}")


def dataloader_helper(cfg, batch_size=None, is_linsearch=False):
    """Get dataloaders for validation and testing."""
    if is_linsearch:
        cfg.dataloader.val_batch_size = 16
        cfg.dataloader.val_num_workers = 4
    cfg.dataloader.test_batch_size = 128
    cfg.dataloader.test_num_workers = 16
    if batch_size is not None:
        cfg.dataloader.val_batch_size = batch_size
    _, val_loader, test_loader = get_dataloader(cfg, val_shuffle=True)
    return (val_loader, test_loader)


def write_result(file_path, result):
    if file_path:
        with open(file_path, "a") as f:
            f.write(json.dumps(result) + "\n")
    else:
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"{key}: {value}")
        elif isinstance(result, list):
            for item in result:
                print(item)
        else:
            print(result)


def train_merged_model(
    args,
    cfgs,
    val_loaders,
    model_dirs,
    lora_dicts,
    preference,
    device,
    baselines_val,
):
    num_epochs = args.num_epochs
    stch_alpha = args.stch_alpha
    lambda_reg = args.lambda_reg
    merge_type = args.merge_type
    lamda = args.lamda
    loss_type = args.loss_type
    lr = args.learning_rate
    kwargs = {
        "cfgs": cfgs,
        "model_dirs": model_dirs,
        "lora_dicts": lora_dicts,
        "preference": preference,
        "merge_type": merge_type,
        "scaling_coeff": lamda,
    }
    merged_model = SimpleMergeModel(**kwargs)
    merged_model.to(device)
    params_to_update = [p for p in merged_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_update, lr=lr)
    print(f"Number of parameters to update: {sum([p.numel() for p in params_to_update])}")

    criterion_stch = STCHLoss(
        criterions_cfgs=cfgs,
        baseline_losses=get_baseline_losses(baselines_val, cfgs),
        preference=preference,
        alpha=stch_alpha,
        loss_type=loss_type,
    )
    task = cfgs[0].task.name

    cycled_val_loaders = [cycle(val_loader) for val_loader in val_loaders]
    if os.getenv("MAX_STEPS") is not None:
        max_steps = int(os.getenv("MAX_STEPS"))
        num_epochs = 1
    else:
        max_steps = min(50, min(len(loader) for loader in val_loaders))
    accumulate_step = int(os.getenv("ACCUMULATE_STEPS", 1))
    merged_model.eval()

    for epoch in range(num_epochs):
        total_loss = []
        print(f"Epoch {epoch + 1}/{num_epochs}")
        pbar = tqdm(range(max_steps), desc=f"Epoch {epoch + 1}", leave=False)
        optimizer.zero_grad()
        for i in pbar:
            samples = [next(loader) for loader in cycled_val_loaders]

            if task == "sequence_classification":
                inputs = [{k: v.to(device) for k, v in sample.items()} for sample in samples]
                targets = [None for _ in inputs]
            else:
                inputs = [sample["image"].to(device) for sample in samples]
                targets = [
                    sample[cfg.task.tgt].to(device) if cfg.task.name != "classification" else None
                    for sample, cfg in zip(samples, cfgs)
                ]

            loss = torch.tensor([0.0], device=device)
            if task == "sequence_classification":
                with autocast(dtype=torch.bfloat16):
                    outs = merged_model(inputs)
                    loss += criterion_stch(outs, targets)
                    if lambda_reg > 0:
                        loss += merged_model.reg_loss(lambda_reg=lambda_reg)
            else:
                outs = merged_model(inputs)
                loss += criterion_stch(outs, targets)
                if lambda_reg > 0:
                    loss += merged_model.reg_loss(lambda_reg=lambda_reg)

            if hasattr(loss, "grad_fn") and loss.grad_fn is not None:
                loss = loss / accumulate_step
                loss.backward()
                total_loss.append(loss.item())
                if (i + 1) % accumulate_step == 0 or (i + 1) == max_steps:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                total_loss.append(loss.item())

            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} Loss: {sum(total_loss) / len(total_loss):.4f}")
    return merged_model


def eval_indiv(
    cfgs,
    models,
    test_loaders,
    criterions,
    validate_fns,
    rep_metric_lists,
    baselines,
    device,
    final_results,
    file_path,
):
    avg_final_metric = 0.0
    avg_normalized_metric = 0.0
    with torch.no_grad():
        for cfg, model, test_loader, criterion, validate_fn, rep_metric, baseline in zip(
            cfgs,
            models,
            test_loaders,
            criterions,
            validate_fns,
            rep_metric_lists,
            baselines,  # type: ignore
        ):
            model.to(device)
            model.eval()
            final_metric = validate_fn(
                cfg,
                model,
                test_loader,
                criterion,
                device,
            )
            baseline_metric = baseline[rep_metric]
            final_model_metric = final_metric[rep_metric]
            result = {
                "rep_metric": rep_metric,
                "baseline_metric": f"{baseline_metric:.4f}",
                "merged_metric": f"{final_model_metric:.4f}",
                "normalized_metric": f"{100 * (final_model_metric / baseline_metric):.2f} %",
            }
            if "total_loss" in final_metric and "total_loss" in baseline:
                result["loss (delta)"] = final_metric["total_loss"] - baseline["total_loss"]
            if cfg.task.name == "classification":
                result["entropy (delta)"] = final_metric["entropy"] - baseline["entropy"]
            avg_final_metric += final_model_metric
            avg_normalized_metric += 100 * (final_model_metric / baseline_metric)
            final_results[cfg.run_name] = result  # type: ignore
        write_result(file_path, final_results)
        avg_final_metric /= len(cfgs)
        avg_normalized_metric /= len(cfgs)
        write_result(
            file_path,
            {
                "average_final_metric": f"{avg_final_metric:.4f}",
                "average_normalized_metric": f"{avg_normalized_metric:.2f} %",
            },
        )


def simple_merging(args, cfgs, val_loaders, test_loaders, simple_combination_type, **kwargs):
    hyperparams = {
        "run_name": args.run_name,
        "merging_type": args.merge_type,
        "preference": args.preference,
        "num_epochs": args.num_epochs,
        "stch_alpha": args.stch_alpha,
        "lambda_reg": args.lambda_reg,
        "model_dirs": args.model_dirs,
        "scaling_coeff": args.lamda,
        "loss_type": args.loss_type,
        "learning_rate": args.learning_rate,
    }
    write_result(FILE_PATH, hyperparams)
    model_dirs = args.model_dirs
    preference = torch.tensor(args.preference)
    FINAL_RESULTS = {
        "simple_combination_type": simple_combination_type,
        "preference": preference.tolist(),
    }

    validate_fns = [get_validation_fn(cfg) for cfg in cfgs]
    rep_metric_lists = [get_rep_metric(cfg) for cfg in cfgs]

    assert len(cfgs) == len(model_dirs), "Mismatch in number of cfgs and model directories"
    lora_dicts = {
        f"{cfg.data.name}-{cfg.task.name}": get_adapter_save_path(cfg, model_dir)
        for cfg, model_dir in zip(cfgs, model_dirs)
        if not ("unseen" in cfg and cfg.unseen)
    }

    assert all(cfg.device == cfgs[0].device for cfg in cfgs), "Devices must match for merging"
    device = get_device(cfgs[0].device)
    for cfg in cfgs:
        cfg.device = device

    criterions = [get_criterion(cfg) for cfg in cfgs]

    base_models = [get_base_model(cfg) for cfg in cfgs]
    models = [
        load_lora_model(
            base_model,
            lora_dicts[f"{cfg.data.name}-{cfg.task.name}"],
            get_decoder_save_path(cfg, model_dir),
        )
        if not ("unseen" in cfg and cfg.unseen)
        else base_model
        for base_model, cfg, model_dir in zip(base_models, cfgs, model_dirs)
    ]

    retrieved_baselines = retrieve_baselines(args, cfgs)
    if not retrieved_baselines:
        baselines_val, baselines_test = [], []
        for cfg, model, val_loader, test_loader, criterion, validate_fn in zip(
            cfgs, models, val_loaders, test_loaders, criterions, validate_fns
        ):
            model.to(device)
            with torch.no_grad():
                baselines_val.append(validate_fn(cfg, model, val_loader, criterion, device))
                baselines_test.append(validate_fn(cfg, model, test_loader, criterion, device))
            model.cpu()
            torch.cuda.empty_cache()
        breakpoint()
    else:
        baselines_val, baselines_test = retrieved_baselines

    base_model = get_vit_from_id(cfgs[0].model.id)
    model_preference = [
        (len(cfgs) / len(lora_dicts)) * preference[i] for i in range(len(lora_dicts))
    ]
    weights = [args.lamda * p.item() for p in model_preference]
    merged_vit = merge_loras(
        base_model,
        lora_dicts,
        weights,
        combination_type=simple_combination_type,
        **kwargs,
    )
    merged_vit = merged_vit.to(device)

    for model in models:
        model.vit = merged_vit  # type: ignore
    eval_indiv(
        cfgs,
        models,
        test_loaders if not args.linsearch else val_loaders,
        criterions,
        validate_fns,
        rep_metric_lists,
        baselines_test,
        device,
        FINAL_RESULTS,
        FILE_PATH,
    )


def trainable_merging(args, cfgs, val_loaders, test_loaders):
    hyperparams = {
        "run_name": args.run_name,
        "merging_type": args.merge_type,
        "preference": args.preference,
        "num_epochs": args.num_epochs,
        "stch_alpha": args.stch_alpha,
        "lambda_reg": args.lambda_reg,
        "model_dirs": args.model_dirs,
        "sclaing_coeff": args.lamda,
        "loss_type": args.loss_type,
        "learning_rate": args.learning_rate,
    }
    write_result(FILE_PATH, hyperparams)
    model_dirs = args.model_dirs
    preference = torch.tensor(args.preference)
    FINAL_RESULTS = {
        "preference": preference.tolist(),
    }

    validate_fns = [get_validation_fn(cfg) for cfg in cfgs]
    rep_metric_lists = [get_rep_metric(cfg) for cfg in cfgs]

    assert len(cfgs) == len(model_dirs), "Mismatch in number of cfgs and model directories"
    lora_dicts = {
        f"{cfg.data.name}-{cfg.task.name}": get_adapter_save_path(cfg, model_dir)
        for cfg, model_dir in zip(cfgs, model_dirs)
        if not ("unseen" in cfg and cfg.unseen)
    }

    assert all(cfg.device == cfgs[0].device for cfg in cfgs), "Devices must match for merging"
    device = get_device(cfgs[0].device)
    for cfg in cfgs:
        cfg.device = device

    criterions = [get_criterion(cfg) for cfg in cfgs]

    retrieved_baselines = retrieve_baselines(args, cfgs)
    if not retrieved_baselines:
        baselines_val, baselines_test = [], []
        base_models = [get_base_model(cfg) for cfg in cfgs]
        models = [
            load_lora_model(
                base_model,
                lora_dicts[f"{cfg.data.name}-{cfg.task.name}"],
                get_decoder_save_path(cfg, model_dir),
            )
            for base_model, cfg, model_dir in zip(base_models, cfgs, model_dirs)
            if not ("unseen" in cfg and cfg.unseen)
        ]
        for cfg, model, val_loader, test_loader, criterion, validate_fn in zip(
            cfgs, models, val_loaders, test_loaders, criterions, validate_fns
        ):
            model.to(device)
            with torch.no_grad():
                baselines_val.append(validate_fn(cfg, model, val_loader, criterion, device))
                baselines_test.append(validate_fn(cfg, model, test_loader, criterion, device))
            model.cpu()
            torch.cuda.empty_cache()
        breakpoint()
    else:
        baselines_val, baselines_test = retrieved_baselines

    torch.cuda.empty_cache()
    merged_model = train_merged_model(
        args,
        cfgs,
        val_loaders,
        model_dirs,
        lora_dicts,
        preference,
        device,
        baselines_val,
    )

    eval_indiv(
        cfgs=cfgs,
        models=merged_model.models,
        test_loaders=test_loaders,
        criterions=criterions,
        validate_fns=validate_fns,
        rep_metric_lists=rep_metric_lists,
        baselines=baselines_test,
        device=device,
        final_results=FINAL_RESULTS,
        file_path=FILE_PATH,
    )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="eight_vision_vit_b_32_rank_16")
    parser.add_argument(
        "--model_dirs",
        type=str,
        nargs="+",
        default=[],
        help="List of directories (or config yaml file) for the models to merge",
    )
    parser.add_argument("--simple_merging", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--stch_alpha", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=0.0)
    parser.add_argument("--preference", type=float, nargs="+", default=[0.5, 0.5])
    parser.add_argument("--output_file", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--merge_type",
        type=str,
        default="weight",
        choices=["weight", "tara-a", "tara-b"],
        help=(
            "Type of merging to perform: "
            "'weight' (AdaMerging-style), "
            "'tara-a' (TARA Variant A: per-rank LoRA direction selection), "
            "'tara-b' (TARA Variant B: shared singular-direction selection)."
        ),
    )
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="stch",
        choices=["stch", "default"],
        help="Loss aggregation: 'stch' (Smooth Tchebycheff) or 'default' (linear).",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ada", action="store_true", help="Enable preference-weighted training")
    parser.add_argument("--linsearch", action="store_true")
    return parser


def linsearch_task_arithmetic(args, cfgs, val_loaders, test_loaders):
    for scaling_factor in np.linspace(0.1, 1.0, num=10):
        args.lamda = scaling_factor
        args.preference = [scaling_factor for _ in range(len(cfgs))]
        simple_merging(
            args,
            cfgs,
            val_loaders,
            test_loaders,
            "ta",
            method="ta",
            scaling_factor=scaling_factor,
        )


def linsearch_ties(args, cfgs, val_loaders, test_loaders):
    scaling_factor = 0.4
    topK_list = list(np.linspace(0.1, 1.0, num=10))
    topK_list.reverse()
    for topK in topK_list:
        args.lamda = 1.0
        args.preference = [scaling_factor for _ in range(len(cfgs))]
        simple_merging(
            args,
            cfgs,
            val_loaders,
            test_loaders,
            "ties",
            method="ties",
            scaling_factor=scaling_factor,
            topK=topK,
        )


def linsearch_dare_ties(args, cfgs, val_loaders, test_loaders):
    scaling_factor = 0.3
    topK = 0.9
    for drop_p in np.linspace(0.1, 1.0, num=10):
        args.lamda = 1.0
        args.preference = [scaling_factor for _ in range(len(cfgs))]
        simple_merging(
            args,
            cfgs,
            val_loaders,
            test_loaders,
            "dare_ties",
            method="dare_ties",
            scaling_factor=scaling_factor,
            topK=topK,
            drop_p=drop_p,
        )


def linsearch_knots_ties(args, cfgs, val_loaders, test_loaders):
    scaling_factor = 0.4
    topK_list = list(np.linspace(0.1, 1.0, num=10))
    topK_list.reverse()
    for topK in topK_list:
        args.lamda = 1.0
        args.preference = [scaling_factor for _ in range(len(cfgs))]
        simple_merging(
            args,
            cfgs,
            val_loaders,
            test_loaders,
            "ties",
            method="knots_ties",
            scaling_factor=scaling_factor,
            topK=topK,
        )


def linsearch_iso_c(args, cfgs, val_loaders, test_loaders):
    os.makedirs("merge_cache", exist_ok=True)
    for c in np.linspace(0.0, 10.0, num=21)[1:]:
        print(f"Running with scaling_factor: {c} for iso_c")
        args.lamda = 1.0
        args.preference = [c for _ in range(len(cfgs))]
        simple_merging(
            args,
            cfgs,
            val_loaders,
            test_loaders,
            "iso_c",
            method="iso_c",
            scaling_factor=c,
            cache_path=os.path.join("merge_cache", f"iso_c_{args.run_name}.safetensors"),
            profile=False,
        )


def linsearch_iso_cts(args, cfgs, val_loaders, test_loaders):
    os.makedirs("merge_cache", exist_ok=True)
    for c in np.linspace(0.0, 4.0, num=21)[1:]:
        print(f"Running with scaling_factor: {c} for iso_cts")
        args.lamda = 1.0
        args.preference = [c for _ in range(len(cfgs))]
        simple_merging(
            args,
            cfgs,
            val_loaders,
            test_loaders,
            "iso_cts",
            method="iso_cts",
            scaling_factor=c,
            cache_path=os.path.join("merge_cache", f"iso_cts_{args.run_name}.safetensors"),
            profile=False,
        )


def linsearch_tsvm(args, cfgs, val_loaders, test_loaders):
    os.makedirs("merge_cache", exist_ok=True)
    for c in np.linspace(0.0, 3.0, num=31):
        print(f"Running with scaling_factor: {c} for tsvm")
        args.lamda = 1.0
        args.preference = [c for _ in range(len(cfgs))]
        simple_merging(
            args,
            cfgs,
            val_loaders,
            test_loaders,
            "tsvm",
            method="tsvm",
            scaling_factor=c,
            cache_path=os.path.join("merge_cache", f"tsvm_{args.run_name}.safetensors"),
            profile=False,
        )


def linsearch_robustmerge(args, cfgs, val_loaders, test_loaders):
    os.makedirs("merge_cache", exist_ok=True)
    for c in np.linspace(0.0, 2.0, num=11)[1:]:
        print(f"Running with scaling_factor: {c} for robustmerge")
        args.lamda = 1.0
        args.preference = [c for _ in range(len(cfgs))]
        simple_merging(
            args,
            cfgs,
            val_loaders,
            test_loaders,
            "ta",
            method="robustmerge",
            scaling_factor=c,
            cache_path=os.path.join("merge_cache", f"robustmerge_{args.run_name}.safetensors"),
            profile=False,
        )


def linsearch_emr(args, cfgs, val_loaders, test_loaders):
    os.makedirs("merge_cache", exist_ok=True)
    for c in np.linspace(0.0, 1.0, num=11)[1:]:
        print(f"Running with scaling_factor: {c} for emr")
        args.lamda = 1.0
        args.preference = [c for _ in range(len(cfgs))]
        simple_merging(
            args,
            cfgs,
            val_loaders,
            test_loaders,
            "emr",
            method="emr",
            scaling_factor=c,
        )


def linsearch_free(args, cfgs, val_loaders, test_loaders):
    os.makedirs("merge_cache", exist_ok=True)
    for c in np.linspace(0.0, 2.0, num=11)[1:]:
        print(f"Running with scaling_factor: {c} for free")
        args.lamda = 1.0
        args.preference = [c for _ in range(len(cfgs))]
        simple_merging(
            args,
            cfgs,
            val_loaders,
            test_loaders,
            "free",
            method="free",
            scaling_factor=c,
            cache_path=os.path.join("merge_cache", f"free_{args.run_name}.safetensors"),
            profile=False,
        )


if __name__ == "__main__":
    wandb.init()
    parser = get_parser()
    args = parser.parse_args()

    cfgs = [get_cfg_from_model_dir(model_dir) for model_dir in args.model_dirs]

    data_loaders = [
        dataloader_helper(
            cfg,
            batch_size=args.batch_size,
            is_linsearch=args.linsearch,
        )
        for cfg in cfgs
    ]
    val_loaders = [dl[0] for dl in data_loaders]
    test_loaders = [dl[1] for dl in data_loaders]

    if args.output_file:
        FILE_PATH = (
            f"{args.run_name}_"
            f"alpha_{args.stch_alpha}_"
            f"merge_type_{args.merge_type}_"
            f"lambda_reg_{args.lambda_reg}_"
            f"eps_{args.num_epochs}_"
            f"scaling_coeff_{args.lamda}_"
            f"loss_type_{args.loss_type}_"
            f"lr_{args.learning_rate}_"
            f"ada_{args.ada}_"
            ".ndjson"
        )

    if len(args.preference) != len(cfgs):
        print(
            f"Warning: Preference length {len(args.preference)} "
            f"does not match number of models {len(cfgs)}. Using equal preference."
        )
        args.preference = [1.0 / len(cfgs) for _ in range(len(cfgs))]

    date = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.debug:
        print("Debug mode is enabled. Running with ipdb...")
        breakpoint()

    if args.linsearch and args.simple_merging:
        linsearch_ties(args, cfgs, val_loaders, test_loaders)
    elif args.simple_merging:
        print("Running simple merging...")
        simple_merging(
            args,
            cfgs,
            val_loaders,
            test_loaders,
            "ta",
            method="ta",
            scaling_factor=0.4,
        )
    else:
        print(f"Running trainable merging with provided preference...{args.preference}")
        trainable_merging(args, cfgs, val_loaders, test_loaders)
