#!/usr/bin/env python3
"""NAS-style regression suite across models, losses, optimizers, and hyperparameters."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from models.model_factory import (
    AUTOENCODER_MODELS,
    CLASSIFICATION_MODELS,
    GAN_MODELS,
    LARGE_IMAGE_MODELS,
    ModelFactory,
)
from utils.autoencoder_trainer import AutoencoderEvaluator, AutoencoderTrainer
from utils.data_loader import DataLoaderFactory
from utils.dataset_config import DatasetSpec, get_dataset_spec
from utils.early_stopping import EarlyStopping
from utils.evaluator import Evaluator
from utils.gantrainer import GANTrainer
from utils.cgantrainer import CGANTrainer
from utils.wgantrainer import WGANtrainer
from utils.trainer import Trainer
from utils.training_factory import (
    AUTOENCODER_LOSSES,
    CLASSIFICATION_LOSSES,
    GAN_LOSSES,
    OPTIMIZERS,
)

# Model lists are defined in models.model_factory (single source of truth).
ALL_MODELS = CLASSIFICATION_MODELS + AUTOENCODER_MODELS + GAN_MODELS

SEARCH_SPACE = {
    "lr": [1e-4, 1e-3, 1e-2],
    "batch_size": [8, 16, 32],
    "weight_decay": [0.0, 1e-4, 1e-3],
}

DEFAULT_WORKERS = 8
MIN_BATCH_SIZE = 4


@dataclass
class TrialResult:
    """Result of a single training trial."""

    model: str
    loss: str
    optimizer: str
    hyperparameters: dict[str, Any]
    metric_name: str
    metric_value: float
    train_loss: float
    test_loss: float
    epochs_run: int
    epochs_to_convergence: int
    status: str = "ok"
    error: str = ""


@dataclass
class RegressionConfig:
    """Configuration for the regression suite."""

    output_dir: str = "outputs/regression"
    report_path: str = "REPORT.md"
    quick_test: bool = True
    max_epochs: int = 200
    patience: int = 5
    min_delta: float = 0.1
    nas_trials: int = 20
    models: List[str] = field(default_factory=list)
    device: str = "auto"
    seed: int = 42
    workers: int = DEFAULT_WORKERS
    max_batch_size: int = 32
    dataset: str = "mnist"
    data_root: str = ""


def parse_args() -> RegressionConfig:
    parser = argparse.ArgumentParser(description="MNIST model regression / NAS suite")
    parser.add_argument("--output-dir", type=str, default="outputs/regression")
    parser.add_argument("--report-path", type=str, default="REPORT.md")
    parser.add_argument("--quick-test", action="store_true", default=False)
    parser.add_argument("--full", action="store_true", help="Use full MNIST (disables quick-test)")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.1)
    parser.add_argument("--nas-trials", type=int, default=20)
    parser.add_argument("--models", type=str, default="", help="Comma-separated model subset")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel workers (default: {DEFAULT_WORKERS}, capped by GPUs/CPUs). "
        "Use 1 for sequential, 0 or --parallel for auto.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run models in parallel (auto worker count)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=0,
        help="Cap NAS batch size (0=auto: 16 if parallel else 32)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset: mnist | strawberry | plant_village_raspberry | plant_village_orange",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="",
        help="Override dataset root directory (e.g. data/Strawberry/strawberries)",
    )
    args = parser.parse_args()

    quick = args.quick_test and not args.full
    if quick:
        max_epochs, patience, nas_trials = min(args.max_epochs, 10), min(args.patience, 3), min(args.nas_trials, 2)
    else:
        max_epochs, patience, nas_trials = args.max_epochs, args.patience, args.nas_trials

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        models = list(ALL_MODELS)

    workers = resolve_workers(args.workers, args.parallel, len(models), args.device)
    if args.max_batch_size > 0:
        max_batch_size = args.max_batch_size
    elif workers > 1:
        max_batch_size = 16
    else:
        max_batch_size = 32

    return RegressionConfig(
        output_dir=args.output_dir,
        report_path=args.report_path,
        quick_test=quick,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=args.min_delta,
        nas_trials=nas_trials,
        models=models,
        device=args.device,
        seed=args.seed,
        workers=workers,
        max_batch_size=max_batch_size,
        dataset=args.dataset,
        data_root=args.data_root,
    )


def available_parallel_slots(device: str, cap: int = DEFAULT_WORKERS) -> int:
    """Max parallel workers (one job per GPU when using CUDA)."""
    if device == "cpu":
        return min(cap, os.cpu_count() or 1)
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        # Never run more concurrent GPU workers than physical devices.
        n_gpu = max(torch.cuda.device_count(), 1)
        return min(cap, n_gpu)
    return min(cap, os.cpu_count() or 1)


def resolve_workers(requested: int, parallel_flag: bool, num_models: int, device: str) -> int:
    """Choose worker count for parallel regression."""
    if requested == 1:
        return 1
    if parallel_flag or requested == 0:
        return min(num_models, available_parallel_slots(device))
    return min(requested, num_models, available_parallel_slots(device, cap=requested))


def config_to_dict(config: RegressionConfig) -> Dict[str, Any]:
    """Serialize config for multiprocessing workers."""
    return asdict(config)


def config_from_dict(data: Dict[str, Any]) -> RegressionConfig:
    """Restore config from a plain dict."""
    return RegressionConfig(**data)


def get_worker_device(config: RegressionConfig, worker_id: int) -> torch.device:
    """Assign a device per worker when running in parallel."""
    if config.device == "cpu":
        return torch.device("cpu")
    if config.device == "cuda":
        n_gpu = max(torch.cuda.device_count(), 1)
        return torch.device(f"cuda:{worker_id % n_gpu}")
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            return torch.device(f"cuda:{worker_id % n_gpu}")
    return torch.device("cpu")


def release_cuda_memory(device: torch.device) -> None:
    """Free cached GPU memory between trials."""
    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    except RuntimeError:
        # GPU may be in a bad state after a device-side assert; skip cleanup.
        pass


def is_oom_error(exc: BaseException) -> bool:
    """Return True if the exception is CUDA OOM."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message


def init_worker_cuda(device: torch.device) -> None:
    """Initialize CUDA in a spawned worker before creating models."""
    if device.type == "cuda":
        torch.cuda.set_device(device)
        release_cuda_memory(device)
        _ = torch.zeros(1, device=device)
    torch.set_num_threads(1)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def model_category(model_name: str) -> str:
    if model_name in CLASSIFICATION_MODELS:
        return "classifier"
    if model_name in AUTOENCODER_MODELS:
        return "autoencoder"
    if model_name in GAN_MODELS:
        return "gan"
    raise ValueError(f"Unknown model: {model_name}")


def valid_losses(model_name: str) -> tuple[str, ...]:
    cat = model_category(model_name)
    if cat == "classifier":
        return CLASSIFICATION_LOSSES
    if cat == "autoencoder":
        return AUTOENCODER_LOSSES
    return GAN_LOSSES


def valid_optimizers(model_name: str) -> tuple[str, ...]:
    if model_name == "wgan":
        return ("rmsprop", "adam")
    return OPTIMIZERS


def sample_hyperparameters(rng: random.Random, config: RegressionConfig) -> dict[str, Any]:
    """Sample hyperparameters respecting the configured batch-size cap."""
    batch_choices = [b for b in SEARCH_SPACE["batch_size"] if b <= config.max_batch_size]
    if not batch_choices:
        batch_choices = [config.max_batch_size]
    return {
        "lr": rng.choice(SEARCH_SPACE["lr"]),
        "batch_size": rng.choice(batch_choices),
        "weight_decay": rng.choice(SEARCH_SPACE["weight_decay"]),
    }


def resolve_dataset_spec(config: RegressionConfig) -> DatasetSpec:
    """Build dataset spec from regression config."""
    return get_dataset_spec(config.dataset, config.data_root or None)


def get_image_size(model_name: str, dataset_spec: DatasetSpec) -> int:
    if dataset_spec.name != "mnist":
        return dataset_spec.default_image_size
    return 224 if model_name in LARGE_IMAGE_MODELS else 28


def build_model_kwargs(model_name: str, image_size: int) -> dict[str, Any]:
    """Default architecture kwargs per model (matches run.sh defaults)."""
    if model_name == "alexnet":
        return {"dropout": 0.5}
    if model_name == "simple_cnn":
        return {"channels": [32, 64, 64], "input_size": image_size}
    if model_name == "vgg":
        return {"cfg": "A", "input_size": image_size}
    if model_name == "resnet":
        return {"num_blocks": [2, 2, 2, 2]}
    if model_name == "densenet":
        return {"growth_rate": 12, "block_config": (3, 6, 12, 8)}
    if model_name == "mobilenet":
        return {"width_multiplier": 1.0}
    if model_name == "mlp":
        return {"hidden_sizes": [512, 256, 128], "input_size": image_size}
    if model_name == "vit":
        return {
            "img_size": image_size,
            "patch_size": 7 if image_size >= 224 else 4,
            "embed_dim": 128,
            "depth": 4,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
        }
    if model_name == "xception":
        return {"num_blocks": 8}
    if model_name == "efficientnet":
        return {"width_mult": 1.0, "depth_mult": 1.0, "dropout_rate": 0.2, "reduction": 4}
    if model_name == "squeezenet":
        return {"version": 1.1}
    if model_name == "lenet":
        return {"dropout": 0.5}
    if model_name == "nin":
        return {"channels": [96, 256, 384], "dropout": 0.5}
    if model_name == "googlenet":
        return {"dropout": 0.4}
    if model_name == "shufflenet":
        return {"stages": [4, 8, 4], "base_channels": 24}
    if model_name == "se_resnet":
        return {"num_blocks": [2, 2, 2, 2], "reduction": 16}
    if model_name == "wide_resnet":
        return {"depth": 28, "widen_factor": 10, "dropout": 0.3}
    if model_name == "convnext":
        return {"depths": [2, 2, 4, 2], "dims": [48, 96, 192, 384]}
    if model_name == "repvgg":
        return {"width_mult": 1.0}
    if model_name == "regnet":
        return {"widths": [32, 64, 128, 256], "depths": [1, 2, 6, 2]}
    if model_name == "ghostnet":
        return {"width_mult": 1.0}
    if model_name == "resnext":
        return {"num_blocks": [2, 2, 2, 2], "cardinality": 32, "width_per_group": 4}
    if model_name == "res2net":
        return {"num_blocks": [2, 2, 2, 2], "scale": 4, "base_width": 26}
    if model_name == "cbam_resnet":
        return {"num_blocks": [2, 2, 2, 2], "reduction": 16}
    if model_name == "mobilenetv3":
        return {"width_mult": 1.0}
    if model_name == "mnasnet":
        return {"width_mult": 1.0}
    if model_name == "eca_resnet":
        return {"num_blocks": [2, 2, 2, 2], "k_size": 3}
    if model_name == "sknet":
        return {"num_blocks": [2, 2, 2, 2], "reduction": 16}
    if model_name == "dpn":
        return {"num_blocks": [2, 2, 2, 2], "dense_channels": 32}
    if model_name == "lcnet":
        return {"width_mult": 1.0}
    if model_name == "capsnet":
        return {
            "primary_caps": 32,
            "primary_dim": 8,
            "digit_dim": 16,
            "routing_iters": 3,
            "input_size": image_size,
        }
    if model_name == "coord_resnet":
        return {"num_blocks": [2, 2, 2, 2], "reduction": 32}
    if model_name == "hardnet":
        return {"growth_rate": 16}
    if model_name == "cspnet":
        return {"channels": [64, 128, 256, 512], "blocks": [2, 2, 2, 2]}
    if model_name == "van":
        return {"dims": [32, 64, 128, 256], "depths": [2, 2, 4, 2]}
    if model_name == "poolformer":
        return {"dims": [32, 64, 128, 256], "depths": [2, 2, 4, 2], "drop": 0.0}
    if model_name == "darknet":
        return {"width_mult": 1.0}
    if model_name == "inception_resnet":
        return {"blocks": [3, 4, 4, 2], "channels": 192}
    if model_name == "repghost":
        return {"width_mult": 1.0}
    if model_name == "hrnet":
        return {"base_channels": 32}
    if model_name == "swin_tiny":
        return {
            "img_size": image_size,
            "patch_size": 4,
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 7,
        }
    if model_name == "mobilenetv2":
        return {"width_mult": 1.0}
    if model_name == "efficientnetv2":
        return {"width_mult": 1.0}
    if model_name == "deit":
        return {
            "img_size": image_size,
            "patch_size": 16,
            "embed_dim": 128,
            "depth": 4,
            "num_heads": 4,
            "drop_rate": 0.0,
        }
    if model_name == "coatnet":
        return {"img_size": image_size, "dims": [32, 64, 128, 256], "depths": [2, 2, 4, 2]}
    if model_name == "vim_tiny":
        return {"img_size": image_size, "embed_dim": 128, "depths": [2, 4, 6, 2]}
    seq_length = image_size * image_size
    if model_name == "bert":
        return {
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "max_seq_length": seq_length,
        }
    if model_name == "gpt":
        return {
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "max_seq_length": seq_length,
        }
    if model_name in ("lstm", "gru"):
        return {"hidden_size": 128, "num_layers": 2, "dropout": 0.2, "bidirectional": False}
    if model_name == "vanilla_gan":
        return {
            "latent_dim": 100,
            "generator_hidden": 256,
            "discriminator_hidden": 256,
            "image_size": image_size,
        }
    if model_name in ("dcgan", "wgan", "cgan"):
        return {
            "latent_dim": 100,
            "generator_channels": 64,
            "discriminator_channels": 64,
            "image_size": image_size,
        }
    if model_name == "simple_ae":
        return {
            "latent_dim": 32,
            "hidden_dims": [128, 64],
            "input_size": image_size,
            "channels": 1,
        }
    if model_name == "conv_ae":
        return {"latent_dim": 32, "channels": [32, 64, 128], "input_size": image_size}
    if model_name == "vae":
        return {"latent_dim": 32, "hidden_dims": [128, 64], "input_size": image_size}
    if model_name == "denoising_ae":
        return {"noise_factor": 0.3, "hidden_dims": [128, 64], "input_size": image_size}
    return {}


def get_data_loaders(
    model_name: str,
    batch_size: int,
    quick_test: bool,
    image_size: int,
    dataset_spec: DatasetSpec,
    loader_workers: int = 4,
    seed: int = 42,
):
    del model_name  # image size already resolved per model + dataset
    loader_workers = max(loader_workers, 0)
    num_workers = min(loader_workers, 2) if quick_test else loader_workers
    return DataLoaderFactory.get_loaders_for_dataset(
        spec=dataset_spec,
        batch_size=min(batch_size, 32) if quick_test else batch_size,
        num_workers=num_workers,
        image_size=image_size,
        quick_test=quick_test,
        seed=seed,
    )


def run_trial(
    model_name: str,
    loss_name: str,
    optimizer_name: str,
    hyperparams: dict[str, Any],
    config: RegressionConfig,
    device: torch.device,
    loader_workers: int = 4,
) -> TrialResult:
    cat = model_category(model_name)
    dataset_spec = resolve_dataset_spec(config)
    image_size = get_image_size(model_name, dataset_spec)
    model = None

    try:
        train_loader, test_loader = get_data_loaders(
            model_name,
            hyperparams["batch_size"],
            config.quick_test,
            image_size,
            dataset_spec,
            loader_workers=loader_workers,
            seed=config.seed,
        )

        model_kwargs = build_model_kwargs(model_name, image_size)
        model = ModelFactory.create_model(
            model_name,
            num_classes=dataset_spec.num_classes,
            enable_logging=False,
            **model_kwargs,
        )
        model = model.to(device)

        lr = hyperparams["lr"]
        wd = hyperparams["weight_decay"]

        if cat == "classifier":
            trainer = Trainer(model, device, lr, loss_name, optimizer_name, wd)
            evaluator = Evaluator(model, device, loss_name)
            early = EarlyStopping(config.patience, config.min_delta, mode="max")
            metric_name = "test_accuracy"
        elif cat == "autoencoder":
            trainer = AutoencoderTrainer(model, device, lr, loss_name, optimizer_name, wd)
            evaluator = AutoencoderEvaluator(model, device, loss_name)
            early = EarlyStopping(config.patience, min_delta=1e-4, mode="min")
            metric_name = "test_reconstruction_loss"
        else:
            if model_name == "wgan":
                trainer = WGANtrainer(model, device, learning_rate=lr, optimizer_name=optimizer_name)
            elif model_name == "cgan":
                trainer = CGANTrainer(model, device, learning_rate=lr, optimizer_name=optimizer_name)
            else:
                trainer = GANTrainer(model, device, learning_rate=lr, optimizer_name=optimizer_name)
            evaluator = None
            early = EarlyStopping(config.patience, min_delta=1e-3, mode="min")
            metric_name = "generator_loss"

        last_train_loss = 0.0
        last_test_loss = 0.0
        epochs_run = 0
        monitor = 0.0

        for epoch in range(config.max_epochs):
            epochs_run = epoch + 1
            if cat == "gan":
                g_loss, d_loss = trainer.train_epoch(train_loader)
                last_train_loss = g_loss
                last_test_loss = d_loss
                monitor = g_loss
            else:
                last_train_loss, _ = trainer.train_epoch(train_loader)
                last_test_loss, monitor, _, _ = evaluator.evaluate(test_loader)

            if early.step(monitor, epoch + 1):
                break

        final_metric = early.best_metric if early.best_metric is not None else monitor

        return TrialResult(
            model=model_name,
            loss=loss_name,
            optimizer=optimizer_name,
            hyperparameters=hyperparams,
            metric_name=metric_name,
            metric_value=final_metric,
            train_loss=last_train_loss,
            test_loss=last_test_loss,
            epochs_run=epochs_run,
            epochs_to_convergence=early.best_epoch if early.best_epoch > 0 else epochs_run,
        )
    finally:
        if model is not None:
            del model
        release_cuda_memory(device)


def search_best_config(
    model_name: str,
    loss_name: str,
    optimizer_name: str,
    config: RegressionConfig,
    device: torch.device,
    rng: random.Random,
    loader_workers: int = 4,
) -> TrialResult:
    best: Optional[TrialResult] = None
    last_error = ""
    higher_is_better = model_category(model_name) == "classifier"

    for _ in range(config.nas_trials):
        hyperparams = sample_hyperparameters(rng, config)
        result: Optional[TrialResult] = None
        oom_attempts = 0

        while result is None:
            try:
                result = run_trial(
                    model_name,
                    loss_name,
                    optimizer_name,
                    hyperparams,
                    config,
                    device,
                    loader_workers=loader_workers,
                )
            except Exception as exc:
                if is_oom_error(exc) and hyperparams["batch_size"] > MIN_BATCH_SIZE:
                    oom_attempts += 1
                    hyperparams["batch_size"] = max(
                        MIN_BATCH_SIZE, hyperparams["batch_size"] // 2
                    )
                    release_cuda_memory(device)
                    print(
                        f"OOM for {model_name} ({loss_name}/{optimizer_name}); "
                        f"retry with batch_size={hyperparams['batch_size']}",
                        flush=True,
                    )
                    if oom_attempts >= 4:
                        result = TrialResult(
                            model=model_name,
                            loss=loss_name,
                            optimizer=optimizer_name,
                            hyperparameters=hyperparams,
                            metric_name="error",
                            metric_value=0.0,
                            train_loss=0.0,
                            test_loss=0.0,
                            epochs_run=0,
                            epochs_to_convergence=0,
                            status="failed",
                            error=f"CUDA OOM after batch-size retries: {exc}",
                        )
                    continue

                result = TrialResult(
                    model=model_name,
                    loss=loss_name,
                    optimizer=optimizer_name,
                    hyperparameters=hyperparams,
                    metric_name="error",
                    metric_value=0.0,
                    train_loss=0.0,
                    test_loss=0.0,
                    epochs_run=0,
                    epochs_to_convergence=0,
                    status="failed",
                    error=str(exc),
                )
                traceback.print_exc()

        if result.status == "failed":
            if result.error:
                last_error = result.error
            continue
        if best is None:
            best = result
            continue
        if higher_is_better:
            if result.metric_value > best.metric_value:
                best = result
        elif result.metric_value < best.metric_value:
            best = result

    if best is None:
        return TrialResult(
            model=model_name,
            loss=loss_name,
            optimizer=optimizer_name,
            hyperparameters={},
            metric_name="error",
            metric_value=0.0,
            train_loss=0.0,
            test_loss=0.0,
            epochs_run=0,
            epochs_to_convergence=0,
            status="failed",
            error=last_error or "All trials failed",
        )
    return best


def format_hyperparameters(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


def escape_md_cell(text: str) -> str:
    """Escape characters that break markdown table cells."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def format_metric(value: float, precision: int = 4) -> str:
    """Format a numeric metric for markdown tables (handles nan/inf/huge values)."""
    if not math.isfinite(value):
        return "—"
    if abs(value) >= 1_000_000 or (0 < abs(value) < 1e-4):
        return f"{value:.4e}"
    return f"{value:.{precision}f}"


def markdown_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    """Build a GFM markdown table; each row is a single line."""
    header_line = "| " + " | ".join(escape_md_cell(h) for h in headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [
        "| " + " | ".join(escape_md_cell(cell) for cell in row) + " |" for row in rows
    ]
    return [header_line, separator_line, *body_lines]


def best_per_model(results: List[TrialResult], higher_is_better: bool) -> List[TrialResult]:
    """Keep the single best trial per model name."""
    best_map: Dict[str, TrialResult] = {}
    for result in results:
        current = best_map.get(result.model)
        if current is None:
            best_map[result.model] = result
            continue
        if not math.isfinite(result.metric_value):
            continue
        if not math.isfinite(current.metric_value):
            best_map[result.model] = result
            continue
        if higher_is_better:
            if result.metric_value > current.metric_value:
                best_map[result.model] = result
        elif result.metric_value < current.metric_value:
            best_map[result.model] = result
    ranked = list(best_map.values())
    ranked.sort(key=lambda r: r.metric_value, reverse=higher_is_better)
    return ranked


def _trial_row_classifier(rank: int, result: TrialResult) -> List[str]:
    return [
        str(rank),
        result.model,
        result.loss,
        result.optimizer,
        format_hyperparameters(result.hyperparameters),
        format_metric(result.metric_value, precision=2),
        format_metric(result.test_loss),
        str(result.epochs_run),
        str(result.epochs_to_convergence),
    ]


def _trial_row_autoencoder(rank: int, result: TrialResult) -> List[str]:
    return [
        str(rank),
        result.model,
        result.loss,
        result.optimizer,
        format_hyperparameters(result.hyperparameters),
        format_metric(result.metric_value),
        str(result.epochs_run),
        str(result.epochs_to_convergence),
    ]


def _trial_row_gan(rank: int, result: TrialResult) -> List[str]:
    return [
        str(rank),
        result.model,
        result.loss,
        result.optimizer,
        format_hyperparameters(result.hyperparameters),
        format_metric(result.train_loss),
        format_metric(result.test_loss),
        str(result.epochs_run),
        str(result.epochs_to_convergence),
    ]


def run_models_subset(
    model_names: List[str],
    config: RegressionConfig,
    worker_id: int = 0,
) -> List[TrialResult]:
    """Run regression for a subset of models (one worker)."""
    device = get_worker_device(config, worker_id)
    loader_workers = 0 if config.workers > 1 else 4
    rng = random.Random(config.seed + worker_id * 10007)
    results: List[TrialResult] = []

    for model_name in model_names:
        if model_name not in ModelFactory.get_available_models():
            print(f"[worker {worker_id}] Skipping unknown model: {model_name}", flush=True)
            continue
        print(f"[worker {worker_id}] Model: {model_name} on {device}", flush=True)
        for loss_name in valid_losses(model_name):
            for optimizer_name in valid_optimizers(model_name):
                print(
                    f"[worker {worker_id}]   {model_name}: {loss_name} + {optimizer_name}",
                    flush=True,
                )
                result = search_best_config(
                    model_name,
                    loss_name,
                    optimizer_name,
                    config,
                    device,
                    rng,
                    loader_workers=loader_workers,
                )
                results.append(result)
                if result.status == "ok":
                    print(
                        f"[worker {worker_id}]     -> {result.metric_name}="
                        f"{result.metric_value:.4f}",
                        flush=True,
                    )
                else:
                    print(f"[worker {worker_id}]     -> FAILED: {result.error}", flush=True)
    return results


def _worker_entry(args: Tuple[int, List[str], Dict[str, Any]]) -> str:
    """ProcessPool worker: run models and write partial results JSON."""
    worker_id, model_names, config_dict = args
    config = config_from_dict(config_dict)
    device = get_worker_device(config, worker_id)
    init_worker_cuda(device)
    results = run_models_subset(model_names, config, worker_id)
    partial_dir = os.path.join(config.output_dir, "partials")
    os.makedirs(partial_dir, exist_ok=True)
    path = os.path.join(partial_dir, f"results_worker_{worker_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    return path


def split_models(model_names: List[str], num_workers: int) -> List[List[str]]:
    """Split model list into balanced chunks for workers."""
    chunks: List[List[str]] = [[] for _ in range(num_workers)]
    for idx, name in enumerate(model_names):
        chunks[idx % num_workers].append(name)
    return [c for c in chunks if c]


def load_partial_results(partial_paths: List[str]) -> List[TrialResult]:
    """Load and merge partial result files from workers."""
    merged: List[TrialResult] = []
    for path in partial_paths:
        with open(path, encoding="utf-8") as f:
            for row in json.load(f):
                merged.append(TrialResult(**row))
    return merged


def run_parallel(config: RegressionConfig) -> List[TrialResult]:
    """Run regression across workers in separate processes."""
    chunks = split_models(config.models, config.workers)
    config_dict = config_to_dict(config)
    tasks = [(i, chunk, config_dict) for i, chunk in enumerate(chunks)]

    # CUDA requires spawn (not fork) so each worker gets a fresh CUDA context.
    mp_ctx = mp.get_context("spawn")
    print(
        f"Parallel regression: {config.workers} workers, {len(chunks)} chunks "
        f"(start method=spawn)",
        flush=True,
    )
    partial_paths: List[str] = []
    with ProcessPoolExecutor(max_workers=config.workers, mp_context=mp_ctx) as executor:
        futures = {executor.submit(_worker_entry, task): task[0] for task in tasks}
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                path = future.result()
                partial_paths.append(path)
                print(f"Worker {worker_id} finished -> {path}", flush=True)
            except Exception as exc:
                print(f"Worker {worker_id} failed: {exc}", flush=True)
                traceback.print_exc()

    if not partial_paths:
        return []
    return load_partial_results(partial_paths)


def generate_report(results: List[TrialResult], config: RegressionConfig, elapsed: float) -> str:
    classifiers = [r for r in results if model_category(r.model) == "classifier" and r.status == "ok"]
    autoencoders = [r for r in results if model_category(r.model) == "autoencoder" and r.status == "ok"]
    gans = [r for r in results if model_category(r.model) == "gan" and r.status == "ok"]
    failed = [r for r in results if r.status == "failed"]

    classifiers.sort(key=lambda r: r.metric_value, reverse=True)
    autoencoders.sort(key=lambda r: r.metric_value)
    gans.sort(key=lambda r: r.metric_value)

    classifier_best = best_per_model(classifiers, higher_is_better=True)
    autoencoder_best = best_per_model(autoencoders, higher_is_better=False)
    gan_best = best_per_model(gans, higher_is_better=False)

    dataset_spec = resolve_dataset_spec(config)
    lines = [
        f"# {dataset_spec.name.title()} Regression Report",
        "",
        "## Run configuration",
        "",
        *markdown_table(
            ["Setting", "Value"],
            [
                ["Generated", datetime.now().isoformat(timespec="seconds")],
                ["Dataset", dataset_spec.name],
                ["Classes", str(dataset_spec.num_classes)],
                ["Class names", ", ".join(dataset_spec.class_names)],
                ["Mode", "quick-test" if config.quick_test else "full dataset"],
                ["Max epochs", str(config.max_epochs)],
                ["Early-stop patience", str(config.patience)],
                ["Min delta", str(config.min_delta)],
                ["NAS trials per config", str(config.nas_trials)],
                ["Workers", str(config.workers)],
                ["Max batch size", str(config.max_batch_size)],
                ["Total wall time", f"{elapsed:.1f}s"],
            ],
        ),
        "",
        "Training stops when validation metric shows no significant improvement "
        f"for {config.patience} consecutive epochs.",
        "",
        "## Classification — best per model",
        "",
        *markdown_table(
            [
                "Rank",
                "Model",
                "Loss",
                "Optimizer",
                "Hyperparameters",
                "Test Acc (%)",
                "Test Loss",
                "Epochs Run",
                "Convergence Epoch",
            ],
            [_trial_row_classifier(i, r) for i, r in enumerate(classifier_best, 1)],
        ),
        "",
        f"## Classification — all trials ({len(classifiers)} rows)",
        "",
        *markdown_table(
            [
                "Rank",
                "Model",
                "Loss",
                "Optimizer",
                "Hyperparameters",
                "Test Acc (%)",
                "Test Loss",
                "Epochs Run",
                "Convergence Epoch",
            ],
            [_trial_row_classifier(i, r) for i, r in enumerate(classifiers, 1)],
        ),
        "",
        "## Autoencoder — best per model",
        "",
        *markdown_table(
            [
                "Rank",
                "Model",
                "Loss",
                "Optimizer",
                "Hyperparameters",
                "Recon Loss",
                "Epochs Run",
                "Convergence Epoch",
            ],
            [_trial_row_autoencoder(i, r) for i, r in enumerate(autoencoder_best, 1)],
        ),
        "",
        f"## Autoencoder — all trials ({len(autoencoders)} rows)",
        "",
        *markdown_table(
            [
                "Rank",
                "Model",
                "Loss",
                "Optimizer",
                "Hyperparameters",
                "Recon Loss",
                "Epochs Run",
                "Convergence Epoch",
            ],
            [_trial_row_autoencoder(i, r) for i, r in enumerate(autoencoders, 1)],
        ),
        "",
        "## GAN — best per model",
        "",
        *markdown_table(
            [
                "Rank",
                "Model",
                "Loss",
                "Optimizer",
                "Hyperparameters",
                "G Loss",
                "D Loss",
                "Epochs Run",
                "Convergence Epoch",
            ],
            [_trial_row_gan(i, r) for i, r in enumerate(gan_best, 1)],
        ),
        "",
        f"## GAN — all trials ({len(gans)} rows)",
        "",
        *markdown_table(
            [
                "Rank",
                "Model",
                "Loss",
                "Optimizer",
                "Hyperparameters",
                "G Loss",
                "D Loss",
                "Epochs Run",
                "Convergence Epoch",
            ],
            [_trial_row_gan(i, r) for i, r in enumerate(gans, 1)],
        ),
    ]

    if failed:
        lines.extend(["", "## Failed Configurations", ""])
        for r in failed:
            lines.append(f"- **{r.model}** / {r.loss} / {r.optimizer}: {r.error}")

    lines.extend(
        [
            "",
            "## Search Space",
            "",
            f"- Loss (classification): {', '.join(CLASSIFICATION_LOSSES)}",
            f"- Loss (autoencoder): {', '.join(AUTOENCODER_LOSSES)}",
            f"- Loss (GAN): {', '.join(GAN_LOSSES)} (informational; GANs use fixed objectives)",
            f"- Optimizers: {', '.join(OPTIMIZERS)}",
            f"- Hyperparameters: {json.dumps(SEARCH_SPACE)}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    config = parse_args()
    random.seed(config.seed)

    dataset_spec = resolve_dataset_spec(config)
    if not config.quick_test:
        dataset_spec.validate()
    elif dataset_spec.name == "mnist" and not os.path.isfile(dataset_spec.quick_train_path):
        print("Quick-test MNIST subset missing; run: python3 utils/create_test_data.py")
    elif dataset_spec.train_source == "pv_style":
        dataset_spec.validate()

    os.makedirs(config.output_dir, exist_ok=True)
    partial_dir = os.path.join(config.output_dir, "partials")
    if config.workers > 1:
        if os.path.isdir(partial_dir):
            shutil.rmtree(partial_dir)
        os.makedirs(partial_dir, exist_ok=True)

    mode = "parallel" if config.workers > 1 else "sequential"
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(
        f"Regression suite ({mode}, dataset={dataset_spec.name}, "
        f"classes={dataset_spec.num_classes}, workers={config.workers}, "
        f"quick_test={config.quick_test}, max_batch={config.max_batch_size})"
    )
    print(f"Models: {len(config.models)}")
    if config.workers > 1 and n_gpu > 0:
        print(f"GPUs available: {n_gpu} (one worker per GPU)")

    t0 = time.time()
    if config.workers > 1:
        all_results = run_parallel(config)
    else:
        device = get_device(config.device)
        print(f"Device: {device}")
        all_results = run_models_subset(config.models, config, worker_id=0)

    elapsed = time.time() - t0

    results_path = os.path.join(config.output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    report = generate_report(all_results, config, elapsed)
    with open(config.report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nResults saved to {results_path}")
    print(f"Report written to {config.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
