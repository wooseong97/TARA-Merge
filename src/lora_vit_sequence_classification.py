import logging
import os
import sys
import warnings
from datetime import datetime

import ipdb  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast
from tqdm import tqdm

from models import compute_sequence_classification_metrics
from utils import (
    get_args,
    get_base_model,
    get_config,
    get_criterion,
    get_dataloader,
    get_device,
    get_optimizer,
    init_lora_model,
    load_lora_model,
    save_lora_model,
)

warnings.filterwarnings("ignore")


def train_epoch(cfg, model, dataloader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    step = 0
    total_steps = min(2000, len(dataloader))

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for _, sample in enumerate(pbar):
        step += 1
        sample = {k: v.to(device) for k, v in sample.items()}
        labels = sample["labels"]

        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):
            # Forward pass
            outputs = model(**sample)

            # SAFELY mask logits (float16 can't represent -1e10)
            if hasattr(model, "mask_class") and model.mask_class is not None:
                min_val = torch.finfo(outputs.dtype).min
                outputs[:, model.mask_class] = min_val  # or -1e4

            loss, _, _ = criterion(outputs, labels)

        # backward with gradient scaling
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Use .item() after unscaling
        total_loss += loss.detach().float().item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        if step >= total_steps:
            break

    return {"total_loss": total_loss / step}


def validate_epoch(cfg, model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Validation"):
            sample = {k: v.to(device) for k, v in sample.items()}
            labels = sample["labels"]
            with autocast(dtype=torch.bfloat16):
                outputs = model(**sample)
                if model.mask_class is not None:
                    outputs[:, model.mask_class] = -np.inf
                loss, _, _ = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.append(outputs)
            all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_sequence_classification_metrics(all_preds, all_targets)

    return {"total_loss": total_loss / len(dataloader), **metrics}


def main(cfg: DictConfig):
    """Main training and testing pipeline"""
    # Prepare configuration and logging
    cfg.adapter_save_path = os.path.join(cfg.output_dir, cfg.adapter_save_name)
    cfg.decoder_save_path = os.path.join(cfg.output_dir, cfg.decoder_save_name)
    cfg.device = get_device(cfg.device)
    logger.info(f"Using device: {cfg.device}")

    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloader(cfg)

    # Model
    model = init_lora_model(cfg)
    model = model.to(cfg.device)

    # Loss function
    criterion = get_criterion(cfg)

    # Optimizer and scheduler
    optimizer, scheduler = get_optimizer(
        cfg,
        model,
        num_warmup_steps=0.06 * len(train_loader) * cfg.train.num_epochs,
        num_training_steps=len(train_loader) * cfg.train.num_epochs,
    )

    # Training loop
    best_val_loss = float("inf")
    prev_val_loss = float("inf")
    early_stopping_counter = 0
    early_stopping_patience = (
        cfg.train.early_stopping_patience if "early_stopping_patience" in cfg.train else 3
    )
    train_losses = []
    val_losses = []

    logger.info("Starting training...")

    for epoch in range(cfg.train.num_epochs):
        # Train
        train_metrics = train_epoch(
            cfg, model, train_loader, optimizer, scheduler, criterion, cfg.device, epoch
        )

        # Validate
        val_metrics = validate_epoch(cfg, model, val_loader, criterion, cfg.device)

        # Log metrics
        logger.info(f"Epoch {epoch + 1}/{cfg.train.num_epochs}:")
        logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['total_loss']:.4f}")
        logger.info(f"  Val Top-1 Acc: {val_metrics['top1_acc']:.4f}")
        logger.info(f"  Val Precision: {val_metrics['precision']:.4f}")
        logger.info(f"  Val Recall: {val_metrics['recall']:.4f}")
        logger.info(f"  Val F1: {val_metrics['f1']:.4f}")

        train_losses.append(train_metrics["total_loss"])
        val_losses.append(val_metrics["total_loss"])

        if val_metrics["total_loss"] < prev_val_loss:
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logger.info(
                f"  Early stopping counter: {early_stopping_counter}/{early_stopping_patience}"
            )
            if early_stopping_counter >= early_stopping_patience:
                logger.info("  Early stopping triggered.")
                break
        prev_val_loss = val_metrics["total_loss"]
        # Save best model and check for early stopping
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            save_lora_model(model, cfg.adapter_save_path, cfg.decoder_save_path)
            logger.info(f"  Saved best model with val loss: {best_val_loss:.4f}")

    # Training complete. Clean up GPU memory.
    model.cpu()
    torch.cuda.empty_cache()
    # Plot train curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(cfg.output_dir, "training_curves.png"),
        dpi=150,
        bbox_inches="tight",
    )

    # Load best model for testing
    model = get_base_model(cfg)
    model = load_lora_model(model, cfg.adapter_save_path, cfg.decoder_save_path)
    model = model.to(cfg.device)

    # Test evaluation
    logger.info("Evaluating on test set...")
    test_metrics = validate_epoch(cfg, model, test_loader, criterion, cfg.device)

    logger.info("Final Test Results:")
    logger.info(f"  Test Loss: {test_metrics['total_loss']:.4f}")
    logger.info(f"  Test Top-1 Acc: {test_metrics['top1_acc']:.4f}")
    logger.info(f"  Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  Test F1: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    # Get configuration
    args = get_args()

    main_cfg = OmegaConf.load(args.config)
    main_cfg.task = "sequence_classification"
    # Load configuration
    cfg = get_config(main_cfg, args)
    # Configure output directory and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.output_dir = f"outputs/{cfg.run_name}/{timestamp}"
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_file = os.path.join(cfg.output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    # Save the configuration
    with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    main(cfg)  # type: ignore
