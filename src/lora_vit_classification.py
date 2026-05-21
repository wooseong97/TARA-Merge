import logging
import os
import sys
import warnings
from datetime import datetime

import ipdb  # noqa: F401
import matplotlib.pyplot as plt
import torch
from models import compute_classification_metrics, compute_classification_metrics_joint
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
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

from data import Transforms

warnings.filterwarnings("ignore")


def get_classes(ds):
    if hasattr(ds, "classes"):
        return ds.classes
    if hasattr(ds, "dataset"):
        return get_classes(ds.dataset)
    return None


def train_epoch(cfg, model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for _, sample in enumerate(pbar):
        images = sample[cfg.task.src].to(device)
        labels = sample[cfg.task.tgt].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss, _, _ = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return {"total_loss": total_loss / len(dataloader)}


def validate_epoch(cfg, model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Validation"):
            images = sample[cfg.task.src].to(device)
            labels = sample[cfg.task.tgt].to(device)

            logits = model(images)
            loss, _, _ = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.append(logits)
            all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_classification_metrics(all_preds, all_targets)

    return {"total_loss": total_loss / len(dataloader), **metrics}


def validate_joint_classification(cfg, model, dataloader, device, label_map):
    model.eval()
    total_loss = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Validation"):
            images = sample[cfg.task.src].to(device)
            labels = label_map[sample[cfg.task.tgt]].to(device)

            logits = model(images)

            all_preds.append(logits)
            all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_classification_metrics_joint(all_preds, all_targets)

    return {"total_loss": total_loss / len(dataloader), **metrics}


# TOOD: Better visualization
def visualize_predictions(cfg, model, dataloader, device, class_names, num_samples=4):
    """Visualize classification predictions"""
    model.eval()
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 8))

    with torch.no_grad():
        i = 0
        for _, sample in enumerate(dataloader):
            image_batch = sample[cfg.task.src].to(device)
            label_batch = sample[cfg.task.tgt].to(device)

            for image, label in zip(image_batch, label_batch):
                if i >= num_samples:
                    break
                logits = model(image.unsqueeze(0))
                pred = torch.argmax(logits, dim=1).item()

                img_np = image.cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 0.5 + 0.5).clip(0, 1)

                axes[i].imshow(img_np)
                axes[i].set_title(f"GT: {class_names[label]}\nPred: {class_names[pred]}")
                axes[i].axis("off")
                i += 1
            if i >= num_samples:
                break

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "classification_preds.png"), dpi=150)


def main(cfg: DictConfig):
    """Main training and testing pipeline"""
    # Prepare configuration and logging
    cfg.adapter_save_path = os.path.join(cfg.output_dir, cfg.adapter_save_name)
    cfg.decoder_save_path = os.path.join(cfg.output_dir, cfg.decoder_save_name)
    cfg.device = get_device(cfg.device)
    logger.info(f"Using device: {cfg.device}")

    # Dataloaders
    transformation = Transforms(cfg)
    # TODO
    train_loader, val_loader, test_loader = get_dataloader(
        cfg,
        train_transform=transformation.get(split="train"),
        val_transform=transformation.get(split="val"),
    )
    classes = get_classes(train_loader.dataset)
    logger.info(f"Number of classes: {len(classes)}")
    logger.info(f"Number of training samples: {len(train_loader.dataset)}")
    logger.info(f"Number of validation samples: {len(val_loader.dataset)}")
    logger.info(f"Number of test samples: {len(test_loader.dataset)}")

    # Model
    model = init_lora_model(cfg)
    model = model.to(cfg.device)

    # Loss function
    criterion = get_criterion(cfg)

    # Optimizer and scheduler
    optimizer, scheduler = get_optimizer(cfg, model)

    # Training loop
    best_val_loss = float("inf")
    prev_val_loss = float("inf")
    early_stopping_counter = 0
    early_stopping_patience = (
        cfg.train.early_stopping_patience if "early_stopping_patience" in cfg.train else 5
    )
    train_losses = []
    val_losses = []

    logger.info("Starting training...")

    for epoch in range(cfg.train.num_epochs):
        # Train
        train_metrics = train_epoch(
            cfg, model, train_loader, optimizer, criterion, cfg.device, epoch
        )

        # Validate
        val_metrics = validate_epoch(cfg, model, val_loader, criterion, cfg.device)

        # Update scheduler
        scheduler.step()

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

    # Visualize predictions
    logger.info("Generating prediction visualizations...")
    visualize_predictions(cfg, model, val_loader, cfg.device, classes)

    logger.info("Training and evaluation completed!")


if __name__ == "__main__":
    # Get configuration
    args = get_args()

    main_cfg = OmegaConf.load(args.config)
    main_cfg.task = "classification"
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
