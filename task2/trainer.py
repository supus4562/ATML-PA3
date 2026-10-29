"""
Unified 3-stage curriculum training engine for Task 2:
- Stage 1: Caption Alignment (2,250 updates, Connector only)
- Stage 2: Visual Instruction Tuning (1,500 updates, Connector + LoRA)
- Stage 3: Natural VQA Specialization (750 updates, Connector + LoRA with lower LR)
- Per-stage 5% warmup and cosine decay to 10% peak LR
- Resumable checkpointing every 250 updates and stage endpoints (2250, 3750, 4500)
- Redundant streaming JSONL and CSV logging with live tqdm progress bar
- Periodic dev evaluation every 500 updates tracking caption loss, answer NLL, and delta NLL.
"""

import copy
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from tqdm import tqdm

from task2.config import Task2Config
from task2.models import VLMModel, apply_lora_to_vlm
from task2.loss import compute_example_weighted_loss, compute_reference_answer_nll
from task2.data import (
    VLMExample,
    DynamicVLMCollator,
    StageCurriculumSampler,
)


class StageCosineLRScheduler:
    """
    Per-stage 5% warmup, then cosine decay to 10% of peak LR:
    eta(t) = eta_min + 0.5 * (eta_peak - eta_min) * (1 + cos(pi * (t - t_warmup) / (t_stage - t_warmup)))
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_ratio: float = 0.05,
        min_lr_ratio: float = 0.10,
        peak_lrs: Optional[List[float]] = None,
    ):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, int(self.total_steps * warmup_ratio))
        self.min_lr_ratio = min_lr_ratio
        self.peak_lrs = peak_lrs or [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> List[float]:
        self.current_step += 1
        t = self.current_step
        lrs = []

        for pg, peak_lr in zip(self.optimizer.param_groups, self.peak_lrs):
            min_lr = peak_lr * self.min_lr_ratio
            if t <= self.warmup_steps:
                cur_lr = peak_lr * (t / self.warmup_steps)
            elif t <= self.total_steps:
                progress = (t - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
                cur_lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
            else:
                cur_lr = min_lr

            pg["lr"] = cur_lr
            lrs.append(cur_lr)

        return lrs


def build_stage_optimizer(
    model: VLMModel,
    stage: int,
    config: Task2Config,
    device: torch.device,
) -> Tuple[torch.optim.Optimizer, StageCosineLRScheduler]:
    """Build optimizer with separate parameter groups for connector and LoRA."""
    weight_decay = config.training.optimizer.weight_decay

    connector_decay = []
    connector_no_decay = []
    lora_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "connector" in name:
            if param.ndim < 2 or "bias" in name:
                connector_no_decay.append(param)
            else:
                connector_decay.append(param)
        elif "lora_" in name:
            lora_params.append(param)

    if stage == 1:
        s1_cfg = config.training.stage1
        param_groups = [
            {"params": connector_decay, "weight_decay": weight_decay, "lr": s1_cfg.connector_peak_lr},
            {"params": connector_no_decay, "weight_decay": 0.0, "lr": s1_cfg.connector_peak_lr},
        ]
        total_steps = s1_cfg.updates
        peak_lrs = [s1_cfg.connector_peak_lr, s1_cfg.connector_peak_lr]

    elif stage == 2:
        s2_cfg = config.training.stage2
        param_groups = [
            {"params": connector_decay, "weight_decay": weight_decay, "lr": s2_cfg.connector_peak_lr},
            {"params": connector_no_decay, "weight_decay": 0.0, "lr": s2_cfg.connector_peak_lr},
            {"params": lora_params, "weight_decay": 0.0, "lr": s2_cfg.lora_peak_lr},
        ]
        total_steps = s2_cfg.updates
        peak_lrs = [s2_cfg.connector_peak_lr, s2_cfg.connector_peak_lr, s2_cfg.lora_peak_lr]

    else:
        s3_cfg = config.training.stage3
        param_groups = [
            {"params": connector_decay, "weight_decay": weight_decay, "lr": s3_cfg.connector_peak_lr},
            {"params": connector_no_decay, "weight_decay": 0.0, "lr": s3_cfg.connector_peak_lr},
            {"params": lora_params, "weight_decay": 0.0, "lr": s3_cfg.lora_peak_lr},
        ]
        total_steps = s3_cfg.updates
        peak_lrs = [s3_cfg.connector_peak_lr, s3_cfg.connector_peak_lr, s3_cfg.lora_peak_lr]

    use_fused = config.training.optimizer.fused and device.type == "cuda"
    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=tuple(config.training.optimizer.betas),
            eps=config.training.optimizer.eps,
            fused=use_fused,
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=tuple(config.training.optimizer.betas),
            eps=config.training.optimizer.eps,
        )

    scheduler = StageCosineLRScheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=config.training.warmup_ratio,
        min_lr_ratio=config.training.min_lr_ratio,
        peak_lrs=peak_lrs,
    )

    return optimizer, scheduler


@torch.no_grad()
def evaluate_development_suite(
    model: VLMModel,
    data_splits: Dict[str, Any],
    collator_normal: DynamicVLMCollator,
    collator_noimg: DynamicVLMCollator,
    device: torch.device,
    im_end_id: int,
) -> Dict[str, float]:
    """
    Evaluates development suites every 500 updates and stage endpoints per PA spec:
    - Caption reference loss on 96 dev captions
    - Reference-answer NLL on Grounding and Natural VQAv2 under 3 conditions:
      1. Correct image (I_correct)
      2. Complementary swap (I_partner)
      3. No image
    - Computes image-swap penalty: delta_NLL = NLL(y | q, I_partner) - NLL(y | q, I_correct).
    """
    model.eval()

    # Helper: chunked evaluation for dev caption loss
    def _eval_caption_loss(examples: List[VLMExample]) -> float:
        if not examples:
            return 0.0
        batch_size = 16
        total_weighted_loss = 0.0
        total_valid = 0
        for i in range(0, len(examples), batch_size):
            chunk = examples[i:i + batch_size]
            b = collator_normal(chunk)
            vis = b["visual_features"].to(device) if b["visual_features"] is not None else None
            out = model.forward_multimodal(
                b["prefix_ids"].to(device),
                vis,
                b["suffix_ids"].to(device),
                b["attention_mask"].to(device),
            )
            _, ex_losses, _ = compute_example_weighted_loss(out.logits, b["labels"].to(device))
            for l_val in ex_losses:
                total_weighted_loss += float(l_val.item())
                total_valid += 1
        return (total_weighted_loss / total_valid) if total_valid > 0 else 0.0

    # Helper: chunked evaluation for reference answer NLL
    def _eval_nll(examples: List[VLMExample], collator: DynamicVLMCollator) -> float:
        if not examples:
            return 0.0
        batch_size = 16
        total_weighted_nll = 0.0
        total_valid = 0
        for i in range(0, len(examples), batch_size):
            chunk = examples[i:i + batch_size]
            b = collator(chunk)
            vis = b["visual_features"].to(device) if b["visual_features"] is not None else None
            out = model.forward_multimodal(
                b["prefix_ids"].to(device),
                vis,
                b["suffix_ids"].to(device),
                b["attention_mask"].to(device),
            )
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = b["labels"][:, 1:].to(device).contiguous()
            if im_end_id is not None:
                shift_labels = shift_labels.clone()
                shift_labels[shift_labels == im_end_id] = -100

            B, S, V = shift_logits.shape
            loss_flat = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            )
            loss_mat = loss_flat.view(B, S)
            valid_mask = (shift_labels != -100).float()
            tokens_per_ex = valid_mask.sum(dim=-1)
            denom = tokens_per_ex.clamp(min=1.0)
            ex_nll = (loss_mat * valid_mask).sum(dim=-1) / denom
            for idx in range(B):
                if tokens_per_ex[idx] > 0:
                    total_weighted_nll += float(ex_nll[idx].item())
                    total_valid += 1
        return (total_weighted_nll / total_valid) if total_valid > 0 else 0.0

    # 1. Dev Captions
    caption_dev = data_splits.get("coco_captions_dev", [])
    cap_loss_val = _eval_caption_loss(caption_dev)

    # 2. VQA Dev Evaluation on Grounding and Natural VQA
    suites = {
        "grounding": data_splits.get("grounding_dev_pairs", []),
        "natural_vqa": data_splits.get("natural_vqa_dev_pairs", []),
    }

    metrics = {"dev_caption_loss": cap_loss_val}

    for suite_name, pairs in suites.items():
        if not pairs:
            continue
        # Unpack pairs: ex_a (img_a), ex_b (img_b)
        all_ex = [p[0] for p in pairs] + [p[1] for p in pairs]

        # Condition 1: Correct Image
        nll_corr = _eval_nll(all_ex, collator_normal)

        # Condition 2: Complementary Swap (swapped images within each pair)
        swapped_ex = []
        for p in pairs:
            swapped_a = copy.copy(p[0])
            swapped_a.image_id = p[0].partner_image_id
            swapped_b = copy.copy(p[1])
            swapped_b.image_id = p[1].partner_image_id
            swapped_ex.extend([swapped_a, swapped_b])
        nll_swap = _eval_nll(swapped_ex, collator_normal)

        # Condition 3: No Image
        nll_noimg = _eval_nll(all_ex, collator_noimg)

        delta_nll = nll_swap - nll_corr

        metrics[f"{suite_name}_nll_correct"] = nll_corr
        metrics[f"{suite_name}_nll_swap"] = nll_swap
        metrics[f"{suite_name}_nll_noimg"] = nll_noimg
        metrics[f"{suite_name}_delta_nll"] = delta_nll

    return metrics


def train_task2(
    model: VLMModel,
    data_splits: Dict[str, Any],
    tokenizer: Any,
    config: Task2Config,
    device: torch.device,
    stage_to_run: str = "all",
    checkpoint_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes the 3-stage curriculum training loop.
    Logs each step, periodically evaluates development metrics, and saves checkpoints.
    """
    ckpt_dir = config.paths.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    live_jsonl = os.path.join(ckpt_dir, "training_steps.jsonl")
    live_csv = os.path.join(ckpt_dir, "training_progress.csv")
    file_mode = "w" if (stage_to_run == "all" or stage_to_run == "stage1" or stage_to_run == ["stage1"]) else "a"
    f_live_jsonl = open(live_jsonl, file_mode, encoding="utf-8", buffering=1)
    f_live_csv = open(live_csv, file_mode, encoding="utf-8", buffering=1)
    if file_mode == "w" or not os.path.exists(live_csv) or os.path.getsize(live_csv) == 0:
        f_live_csv.write("global_update,stage,stage_update,loss,lr_connector,lr_lora,grad_norm,tokens,sec_per_step\n")

    collator_normal = DynamicVLMCollator(
        tokenizer=tokenizer,
        feature_cache_dir=config.data.feature_cache_dir,
        num_patches=config.vision.num_patches,
        hidden_dim=config.vision.hidden_dim,
        question_token_cap=config.training.question_token_cap,
        caption_target_cap=config.training.caption_target_cap,
        short_answer_target_cap=config.training.short_answer_target_cap,
        no_image=False,
    )
    collator_noimg = DynamicVLMCollator(
        tokenizer=tokenizer,
        num_patches=0,
        hidden_dim=config.vision.hidden_dim,
        question_token_cap=config.training.question_token_cap,
        caption_target_cap=config.training.caption_target_cap,
        short_answer_target_cap=config.training.short_answer_target_cap,
        no_image=True,
    )

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    training_logs = {
        "step_history": [],
        "dev_eval_history": [],
        "checkpoints": {},
    }

    # Evaluate at Step 0 before training per PA specification
    print("\n[Evaluation] Running Pre-Training Development Evaluation (Update 0)...")
    dev_0 = evaluate_development_suite(model, data_splits, collator_normal, collator_noimg, device, im_end_id)
    training_logs["dev_eval_history"].append({"global_update": 0, "stage": 0, **dev_0})
    print(f"[Evaluation @ 0] Dev Caption Loss: {dev_0.get('dev_caption_loss', 0.0):.4f} | "
          f"Grounding delta_NLL: {dev_0.get('grounding_delta_nll', 0.0):.4f} | "
          f"Natural VQA delta_NLL: {dev_0.get('natural_vqa_delta_nll', 0.0):.4f}")

    global_update = 0
    microbatch_size = config.training.micro_batch_size
    accum_steps = config.training.gradient_accumulation_steps
    effective_batch = microbatch_size * accum_steps

    if stage_to_run == "all":
        stages_to_execute = [1, 2, 3]
    elif isinstance(stage_to_run, (list, tuple)):
        stages_to_execute = sorted([int(str(s).replace("stage", "")) for s in stage_to_run if "stage" in str(s) or str(s) in ("1", "2", "3")])
    elif isinstance(stage_to_run, str) and (" " in stage_to_run or "," in stage_to_run):
        tokens = stage_to_run.replace(",", " ").split()
        stages_to_execute = sorted([int(s.replace("stage", "")) for s in tokens if "stage" in s or s in ("1", "2", "3")])
    elif isinstance(stage_to_run, str) and (stage_to_run.startswith("stage") or stage_to_run in ("1", "2", "3")):
        stages_to_execute = [int(stage_to_run.replace("stage", ""))]
    else:
        stages_to_execute = [1, 2, 3]

    for stage_idx in stages_to_execute:
        print("\n" + "=" * 80)
        stage_names = {1: "STAGE 1: CAPTION ALIGNMENT", 2: "STAGE 2: VISUAL INSTRUCTION TUNING", 3: "STAGE 3: NATURAL VQA SPECIALIZATION"}
        print(stage_names[stage_idx])
        print("=" * 80)

        # In Stage 2 or 3, enable LoRA adapters on Qwen if not already applied
        if stage_idx in (2, 3) and not getattr(model, "lora_applied", False):
            print("[LoRA] Enabling rank-16 LoRA adapters on Qwen attention projections...")
            apply_lora_to_vlm(
                model,
                rank=config.training.stage2.lora_rank,
                alpha=config.training.stage2.lora_alpha,
                dropout=config.training.stage2.lora_dropout,
                target_modules=config.training.stage2.target_modules,
            )
            model.lora_applied = True

        model.to(device)
        model.train()

        # Build stage-specific optimizer and scheduler
        optimizer, scheduler = build_stage_optimizer(model, stage_idx, config, device)
        if hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=(config.training.fp16 and device.type == "cuda"))
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=(config.training.fp16 and device.type == "cuda"))

        # Setup data sampler for the stage
        if stage_idx == 1:
            stage_updates = config.training.stage1.updates
            # 2 passes over 18,000 caption examples = 36,000 examples / 16 = 2,250 updates
            caption_pool = data_splits["coco_captions_train"]
            cur_pool = caption_pool + caption_pool  # 2 passes
            sampler = None
        elif stage_idx == 2:
            stage_updates = config.training.stage2.updates
            sampler = StageCurriculumSampler(
                stage=2,
                captions=data_splits["coco_captions_train"],
                presence_pairs=data_splits["grounding_train_presence_pairs"],
                counting_pairs=data_splits["grounding_train_counting_pairs"],
                seed=config.training.seed + 2,
            )
        else:
            stage_updates = config.training.stage3.updates
            sampler = StageCurriculumSampler(
                stage=3,
                captions=data_splits["coco_captions_train"],
                presence_pairs=data_splits["grounding_train_presence_pairs"],
                counting_pairs=data_splits["grounding_train_counting_pairs"],
                natural_vqa_pairs=data_splits["natural_vqa_train_pairs"],
                seed=config.training.seed + 3,
            )

        pbar = tqdm(
            total=stage_updates,
            desc=f"[Stage {stage_idx}]",
            unit="step",
            dynamic_ncols=True,
            file=sys.stdout,
            mininterval=0.0,
        )

        stage_update = 0
        pool_idx = 0
        last_time = time.time()

        while stage_update < stage_updates:
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            accum_tokens = 0

            # Accumulation loop (effective batch = 16 = microbatch 8 x 2 steps)
            for acc in range(accum_steps):
                if stage_idx == 1:
                    batch_ex = cur_pool[pool_idx:pool_idx + microbatch_size]
                    pool_idx = (pool_idx + microbatch_size) % len(cur_pool)
                else:
                    batch_ex = []
                    # Draw microbatch_size // 2 pairs (each draw gives 2 examples)
                    for _ in range(microbatch_size // 2):
                        batch_ex.extend(sampler.draw_pair())

                batch = collator_normal(batch_ex)
                prefix_ids = batch["prefix_ids"].to(device, non_blocking=True)
                vis_feats = batch["visual_features"].to(device, non_blocking=True) if batch["visual_features"] is not None else None
                suffix_ids = batch["suffix_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(config.training.fp16 and device.type == "cuda")):
                    outputs = model.forward_multimodal(prefix_ids, vis_feats, suffix_ids, attention_mask)
                    loss, _, n_tokens = compute_example_weighted_loss(outputs.logits, labels)
                    loss_scaled = loss / accum_steps

                scaler.scale(loss_scaled).backward()
                accum_loss += loss.item()
                accum_tokens += n_tokens

            # Unscale and clip gradients
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=config.training.gradient_clipping,
            )

            scaler.step(optimizer)
            scaler.update()

            lrs = scheduler.step()
            stage_update += 1
            global_update += 1

            now = time.time()
            sec_per_step = now - last_time
            last_time = now

            step_loss = accum_loss / accum_steps
            step_record = {
                "global_update": global_update,
                "stage": stage_idx,
                "stage_update": stage_update,
                "loss": step_loss,
                "lr_connector": lrs[0],
                "lr_lora": lrs[-1] if len(lrs) > 2 else 0.0,
                "grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
                "tokens": accum_tokens,
                "sec_per_step": sec_per_step,
            }
            training_logs["step_history"].append(step_record)

            # Update live tqdm progress bar
            pbar.set_postfix({
                "loss": f"{step_loss:.4f}",
                "lr_conn": f"{lrs[0]:.2e}",
                "grad": f"{step_record['grad_norm']:.2f}",
                "s/it": f"{sec_per_step:.2f}",
            }, refresh=False)
            pbar.update(1)
            sys.stdout.flush()

            # Live persistence without blocking fsync
            try:
                f_live_jsonl.write(json.dumps(step_record) + "\n")
                f_live_csv.write(f"{global_update},{stage_idx},{stage_update},{step_loss:.6f},{lrs[0]:.6e},{step_record['lr_lora']:.6e},{step_record['grad_norm']:.4f},{accum_tokens},{sec_per_step:.4f}\n")
            except Exception:
                pass

            # Resumable checkpointing every 250 main-run updates
            if global_update % config.training.checkpoint_frequency == 0 or global_update in config.training.endpoint_updates:
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_update_{global_update}.pt")
                ckpt_state = {
                    "global_update": global_update,
                    "stage": stage_idx,
                    "stage_update": stage_update,
                    "connector_state": model.connector.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_step": scheduler.current_step,
                    "scaler_state": scaler.state_dict(),
                    "rng_state": torch.get_rng_state(),
                }
                if stage_idx >= 2:
                    lora_state = {k: v for k, v in model.language_model.named_parameters() if "lora_" in k}
                    ckpt_state["lora_state"] = lora_state

                torch.save(ckpt_state, ckpt_path)
                training_logs["checkpoints"][str(global_update)] = ckpt_path
                pbar.write(f"[Checkpoint] Saved resumable checkpoint at global update {global_update} -> {ckpt_path}")
                if checkpoint_callback:
                    try:
                        checkpoint_callback(global_update, ckpt_path)
                    except Exception:
                        pass

            # Periodic dev evaluation every 500 updates or stage endpoints
            if global_update % config.training.dev_eval_frequency == 0 or global_update in config.training.endpoint_updates:
                dev_res = evaluate_development_suite(model, data_splits, collator_normal, collator_noimg, device, im_end_id)
                dev_entry = {"global_update": global_update, "stage": stage_idx, **dev_res}
                training_logs["dev_eval_history"].append(dev_entry)
                pbar.write(f"[Dev Eval @ {global_update}] Caption Loss: {dev_res.get('dev_caption_loss', 0.0):.4f} | "
                           f"Grounding delta_NLL: {dev_res.get('grounding_delta_nll', 0.0):.4f} | "
                           f"Natural VQA delta_NLL: {dev_res.get('natural_vqa_delta_nll', 0.0):.4f}")
                model.train()

        pbar.close()

        # Save stage endpoint weights
        stage_weights_path = os.path.join(ckpt_dir, f"stage_{stage_idx}_endpoint_weights.pt")
        stage_payload = {"stage": stage_idx, "connector": model.connector.state_dict()}
        if stage_idx >= 2:
            stage_payload["lora"] = {k: v for k, v in model.language_model.named_parameters() if "lora_" in k}
        torch.save(stage_payload, stage_weights_path)
        print(f"[Stage {stage_idx}] Saved endpoint weights to {stage_weights_path}")

    try:
        f_live_jsonl.close()
        f_live_csv.close()
    except Exception:
        pass

    # Summary save
    summary_path = os.path.join(ckpt_dir, "task2_training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(training_logs, f, indent=2)

    return training_logs
