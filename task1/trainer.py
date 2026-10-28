"""
Unified training engine for Task 1:
- Supports Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, and OPLoRA-16.
- Enforces all shared training invariants:
  * 40,000 examples, 1,250 optimizer updates.
  * Microbatch 2, gradient accumulation 16 (effective batch 32).
  * AdamW with weight decay 0.01 excluding 1D/biases, cosine decay with 38 warmup steps.
  * FP16 mixed precision with dynamic GradScaler.
  * Global gradient clipping 1.0 with unscaling.
  * Effective batch loss normalization.
  * L2-SP starting point penalty Omega(theta, theta_0) = 0.5 * sum (theta_j - theta_{0,j})^2.
  * Recovery checkpoints at updates 312, 625, 937, 1250 with full state preservation.
"""

import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from task1.utils import (
    CUDATimer,
    get_peak_vram_mb,
    reset_peak_vram,
    count_parameters,
    snapshot_model_parameters,
    compute_parameter_displacement,
    get_checkpoint_size_mb,
)
from task1.models import build_optimizer_parameter_groups
from task1.data import DynamicPaddingCollator, compute_normalized_effective_batch_loss


class CosineWarmupLRScheduler:
    """Cosine learning rate decay with linear warmup."""
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.peak_lr * (self.current_step / max(1, self.warmup_steps))
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.current_step = state_dict["current_step"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.total_steps = state_dict["total_steps"]
        self.peak_lr = state_dict["peak_lr"]
        self.min_lr = state_dict["min_lr"]


def compute_l2sp_penalty(
    model: nn.Module,
    theta_0: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, float]:
    """
    Compute Omega(theta, theta_0) = 0.5 * sum_j (theta_j - theta_{0,j})^2
    Summed over all unique parameters (counting tied parameters once).
    Do NOT average Omega over parameters.
    Returns:
        (penalty_tensor, penalty_float_value)
    """
    visited_ids = set()
    total_penalty = torch.tensor(0.0, device=next(model.parameters()).device, dtype=torch.float32)

    for name, param in model.named_parameters():
        param_id = id(param)
        if param_id in visited_ids or name not in theta_0:
            continue
        visited_ids.add(param_id)

        ref_val = theta_0[name].to(device=param.device, dtype=torch.float32)
        diff = param.float() - ref_val
        total_penalty = total_penalty + 0.5 * torch.sum(diff ** 2)

    return total_penalty, float(total_penalty.item())


def train_variant(
    variant_name: str,
    model: nn.Module,
    dataset: Any,
    config: Any,
    device: torch.device,
    output_checkpoint_dir: str = "checkpoints/task1",
    resume_from_checkpoint: Optional[str] = None,
    checkpoint_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute full training for one of the five variants.
    Returns training logs and metadata.
    """
    print(f"\n{'='*70}\n[Trainer] Starting training for: {variant_name.upper()}\n{'='*70}")
    method_cfg = config.methods.get(variant_name, {})
    peak_lr = method_cfg.get("peak_lr", 2e-5)
    lambda_l2sp = method_cfg.get("lambda_l2sp", 0.0)

    # 1. Parameter snapshot for L2-SP or displacement
    theta_0: Optional[Dict[str, torch.Tensor]] = None
    ref_storage_mb = 0.0
    if variant_name in ["full_ft", "full_ft_l2sp"]:
        t0_start = time.perf_counter()
        theta_0 = snapshot_model_parameters(model, device="cpu")
        ref_storage_mb = sum(t.numel() * 4 for t in theta_0.values()) / (1024.0 * 1024.0)
        ref_transfer_sec = time.perf_counter() - t0_start
        print(f"[{variant_name}] Reference theta_0 saved to CPU: {ref_storage_mb:.2f} MB in {ref_transfer_sec:.2f}s")

    # 2. Parameter counts
    total_params, trainable_params = count_parameters(model)
    print(f"[{variant_name}] Parameters: Total = {total_params:,} | Trainable = {trainable_params:,}")

    # 3. Parameter groups & Optimizer
    param_groups = build_optimizer_parameter_groups(model, weight_decay=config.training.optimizer.weight_decay)
    use_fused = config.training.optimizer.fused and device.type == "cuda"
    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=peak_lr,
            betas=tuple(config.training.optimizer.betas),
            eps=config.training.optimizer.eps,
            fused=use_fused,
        )
    except Exception:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=peak_lr,
            betas=tuple(config.training.optimizer.betas),
            eps=config.training.optimizer.eps,
        )

    # 4. Scheduler & Scaler
    scheduler = CosineWarmupLRScheduler(
        optimizer=optimizer,
        warmup_steps=config.training.warmup_updates,
        total_steps=config.training.total_updates,
        peak_lr=peak_lr,
    )
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=(config.training.fp16 and device.type == "cuda"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(config.training.fp16 and device.type == "cuda"))

    # 5. DataLoader with dynamic collator (adaptive workers for Kaggle / workstation)
    collator = DynamicPaddingCollator(pad_token_id=0, ignore_index=-100)
    num_cpus = os.cpu_count() or 1
    is_kaggle = os.path.exists("/kaggle") or ("KAGGLE_KERNEL_RUN_TYPE" in os.environ)
    max_workers = 2 if is_kaggle else 6
    num_workers = min(max_workers, max(1, num_cpus - 1)) if device.type == "cuda" else 0
    data_loader = DataLoader(
        dataset,
        batch_size=config.training.micro_batch_size,
        shuffle=False,  # STRICT INVARIANT: Do not shuffle retained examples!
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # 6. Resume state if requested
    start_update = 0
    start_batch_idx = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"[{variant_name}] Resuming from checkpoint: {resume_from_checkpoint}")
        ckpt = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_update = ckpt["update"]
        start_batch_idx = ckpt["batch_idx"]
        torch.set_rng_state(ckpt["rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"])

    # 7. Training tracking
    reset_peak_vram()
    training_timer = CUDATimer(name="train_loop")
    logs = {
        "variant": variant_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "history": [],
        "checkpoints": {},
        "displacements": {},
        "skipped_updates": 0,
        "total_train_time_sec": 0.0,
        "peak_vram_mb": 0.0,
        "ref_storage_mb": ref_storage_mb,
    }

    model.train()
    if config.training.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    grad_accum_steps = config.training.gradient_accumulation_steps
    target_updates = config.training.total_updates
    current_update = start_update

    ckpt_save_dir = os.path.join(output_checkpoint_dir, variant_name)
    os.makedirs(ckpt_save_dir, exist_ok=True)

    # Redundant logging: streaming JSONL and CSV files flushed on every single step
    live_jsonl_path = os.path.join(ckpt_save_dir, "training_steps.jsonl")
    live_csv_path = os.path.join(ckpt_save_dir, "training_progress.csv")
    if not os.path.exists(live_csv_path):
        with open(live_csv_path, "w", encoding="utf-8") as f_csv_init:
            f_csv_init.write("update,sft_loss,total_loss,omega,lambda_omega,lr,grad_norm,valid_tokens,sec_per_step\n")

    microbatch_buffer = []
    last_print_time = time.time()
    last_print_update = current_update

    # Initialize tqdm progress bar for real-time visual progress
    pbar = tqdm(
        total=target_updates,
        initial=start_update,
        desc=f"[{variant_name.upper()}]",
        unit="step",
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.0,
    )

    with training_timer:
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx < start_batch_idx:
                continue
            if current_update >= target_updates:
                break

            microbatch_buffer.append(batch)

            # Process accumulation window (effective batch = 32 examples)
            if len(microbatch_buffer) == grad_accum_steps or (batch_idx + 1) == len(data_loader):
                # Calculate total valid response tokens across this effective batch upfront
                accum_tokens = sum((mb["labels"] != -100).sum().item() for mb in microbatch_buffer)
                norm_denominator = max(1, accum_tokens)
                total_sft_loss_val = 0.0

                # Microbatch forward & immediate backward to discard activation graphs right away
                for mb in microbatch_buffer:
                    input_ids = mb["input_ids"].to(device, non_blocking=True)
                    labels = mb["labels"].to(device, non_blocking=True)
                    attention_mask = mb["attention_mask"].to(device, non_blocking=True)

                    if hasattr(torch.amp, "autocast"):
                        autocast_ctx = torch.amp.autocast("cuda", enabled=(config.training.fp16 and device.type == "cuda"), dtype=torch.float16)
                    else:
                        autocast_ctx = torch.cuda.amp.autocast(enabled=(config.training.fp16 and device.type == "cuda"), dtype=torch.float16)

                    with autocast_ctx:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        sum_loss, _ = compute_normalized_effective_batch_loss(outputs.logits, labels)
                        mb_scaled_loss = sum_loss / norm_denominator

                    total_sft_loss_val += mb_scaled_loss.item()
                    scaler.scale(mb_scaled_loss).backward()
                    del outputs, sum_loss, mb_scaled_loss, input_ids, labels, attention_mask

                # Free microbatch buffer immediately
                microbatch_buffer.clear()

                # Add L2-SP penalty if enabled
                omega_val = 0.0
                lambda_omega_val = 0.0
                if lambda_l2sp > 0.0 and theta_0 is not None:
                    penalty_tensor, omega_val = compute_l2sp_penalty(model, theta_0)
                    lambda_omega_val = lambda_l2sp * omega_val
                    scaler.scale(lambda_l2sp * penalty_tensor).backward()

                # Optimizer update
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=config.training.gradient_clipping,
                )

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                # Check if update was skipped due to inf/nan
                if scale_after < scale_before:
                    logs["skipped_updates"] += 1

                current_lr = scheduler.step()
                current_update += 1
                optimizer.zero_grad(set_to_none=True)

                total_loss_val = total_sft_loss_val + lambda_omega_val
                step_log = {
                    "update": current_update,
                    "sft_loss": float(total_sft_loss_val),
                    "total_loss": float(total_loss_val),
                    "omega": float(omega_val),
                    "lambda_omega": float(lambda_omega_val),
                    "lr": float(current_lr),
                    "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
                    "valid_tokens": accum_tokens,
                }
                now = time.time()
                dt = now - last_print_time
                d_steps = max(1, current_update - last_print_update)
                sec_per_step = dt / d_steps if (last_print_update > 0 or current_update > 1) else dt
                last_print_time = now
                last_print_update = current_update

                step_log["sec_per_step"] = float(sec_per_step)
                logs["history"].append(step_log)

                # Update tqdm progress bar on every single step
                pbar.update(1)
                pbar.set_postfix({
                    "sft_loss": f"{total_sft_loss_val:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "grad": f"{step_log['grad_norm']:.2f}",
                    "toks": accum_tokens,
                    "s/it": f"{sec_per_step:.2f}",
                })
                sys.stdout.flush()

                # Redundant streaming persistence on every single update
                try:
                    with open(live_jsonl_path, "a", encoding="utf-8") as f_j:
                        f_j.write(json.dumps(step_log) + "\n")
                        f_j.flush()
                        os.fsync(f_j.fileno())
                    with open(live_csv_path, "a", encoding="utf-8") as f_c:
                        f_c.write(
                            f"{current_update},{total_sft_loss_val:.6f},{total_loss_val:.6f},"
                            f"{omega_val:.6f},{lambda_omega_val:.6f},{current_lr:.6e},"
                            f"{step_log['grad_norm']:.6f},{accum_tokens},{sec_per_step:.4f}\n"
                        )
                        f_c.flush()
                        os.fsync(f_c.fileno())
                except Exception as log_err:
                    pbar.write(f"[{variant_name}] Warning: Failed to append to live log files: {log_err}")

                # Checkpoint saving schedule: 312, 625, 937, 1250
                if current_update in config.training.checkpoint_updates:
                    ckpt_file = os.path.join(ckpt_save_dir, f"checkpoint_update_{current_update}.pt")
                    ckpt_payload = {
                        "update": current_update,
                        "batch_idx": batch_idx + 1,
                        "variant": variant_name,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                    }
                    torch.save(ckpt_payload, ckpt_file)
                    ckpt_size = get_checkpoint_size_mb(ckpt_file)
                    logs["checkpoints"][str(current_update)] = {
                        "path": ckpt_file,
                        "size_mb": ckpt_size,
                    }
                    pbar.write(f"[{variant_name}] Saved recovery checkpoint at update {current_update} ({ckpt_size:.2f} MB)")

                    # Full-model displacement recording
                    if variant_name in ["full_ft", "full_ft_l2sp"] and theta_0 is not None:
                        if (variant_name == "full_ft" and current_update in [625, 1250]) or \
                           (variant_name == "full_ft_l2sp" and current_update == 1250):
                            disp = compute_parameter_displacement(model, theta_0)
                            logs["displacements"][str(current_update)] = disp
                            pbar.write(f"[{variant_name}] ||theta - theta_0||_2 at update {current_update}: {disp:.6f}")

                    # Periodically save training summary so checkpoint state is never lost
                    partial_summary_file = os.path.join(ckpt_save_dir, "training_summary.json")
                    try:
                        with open(partial_summary_file, "w", encoding="utf-8") as f_sum:
                            json.dump({
                                "variant": variant_name,
                                "trainable_params": logs["trainable_params"],
                                "total_params": logs["total_params"],
                                "total_train_time_sec": training_timer.elapsed,
                                "peak_vram_mb": get_peak_vram_mb(),
                                "checkpoints": logs["checkpoints"],
                                "displacements": logs["displacements"],
                                "skipped_updates": logs.get("skipped_updates", 0),
                                "history": logs["history"],
                            }, f_sum, indent=2)
                            f_sum.flush()
                            os.fsync(f_sum.fileno())
                    except Exception:
                        pass

                    # Trigger external callback if registered (e.g. Modal Volume commit)
                    if checkpoint_callback is not None and callable(checkpoint_callback):
                        try:
                            checkpoint_callback(current_update, ckpt_file)
                        except Exception as cb_err:
                            pbar.write(f"[{variant_name}] Warning: checkpoint callback error: {cb_err}")

    pbar.close()

    logs["total_train_time_sec"] = training_timer.elapsed
    logs["peak_vram_mb"] = get_peak_vram_mb()
    print(f"[{variant_name}] Training finished: {logs['total_train_time_sec']:.2f}s | Peak VRAM: {logs['peak_vram_mb']:.2f} MB")

    summary_file = os.path.join(ckpt_save_dir, "training_summary.json")
    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "variant": variant_name,
                "trainable_params": logs["trainable_params"],
                "total_params": logs["total_params"],
                "total_train_time_sec": logs["total_train_time_sec"],
                "peak_vram_mb": logs["peak_vram_mb"],
                "checkpoints": logs["checkpoints"],
                "displacements": logs["displacements"],
                "skipped_updates": logs.get("skipped_updates", 0),
                "history": logs["history"],
            }, f, indent=2)
        print(f"[{variant_name}] Saved training summary to {summary_file}")
    except Exception as e:
        print(f"[Warning] Failed to write training summary {summary_file}: {e}")

    return logs
