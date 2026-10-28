"""
Master orchestration script for ATML PA3 - Task 1: Large Language Model Fine-Tuning.
Executes the complete experimental pipeline:
1. Environment & Blackwell GPU setup.
2. Dataset loading, leak filtering, and tokenization cache.
3. Baseline evaluation of unchanged Qwen3-0.6B (Update 0).
4. Five independent training runs:
   - Full FT (midpoint eval @ 625, final @ 1250)
   - Full FT + L2-SP (final @ 1250)
   - LoRA-4 (final @ 1250)
   - LoRA-16 (final @ 1250)
   - OPLoRA-16 (final @ 1250)
5. Weight diagnostics (rho_k, eta) and Activation probes (R(z)) across layers 0, 13, 27.
6. Qualitative evaluation across 24 prompts (168 responses) and 42-response annotation sheet.
7. Result synthesis: consolidated metrics_summary.json, learning_curves.csv, and percentage-point deltas.
"""

import argparse
import copy
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

def get_modal_checkpoint_callback():
    if "MODAL_VOLUME_COMMIT" in os.environ:
        def _cb(update: int, ckpt_file: str):
            try:
                import modal
                vol_name = os.environ.get("MODAL_VOLUME_NAME", "pa3-volume")
                vol = modal.Volume.from_name(vol_name)
                vol.commit()
                print(f"[Modal Volume] Committed checkpoint {update} to volume '{vol_name}'", flush=True)
            except Exception as e:
                print(f"[Modal Volume] Warning: commit error: {e}", flush=True)
        return _cb
    return None

def trigger_modal_stage_commit(stage_name: str):
    if "MODAL_VOLUME_COMMIT" in os.environ:
        try:
            import modal
            vol_name = os.environ.get("MODAL_VOLUME_NAME", "pa3-volume")
            vol = modal.Volume.from_name(vol_name)
            vol.commit()
            print(f"[Modal Volume] Committed stage '{stage_name}' to volume '{vol_name}'", flush=True)
        except Exception as e:
            print(f"[Modal Volume] Warning: stage commit error: {e}", flush=True)

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from task1.config import Task1Config
from task1.utils import (
    set_seed,
    get_peak_vram_mb,
    reset_peak_vram,
    count_parameters,
    snapshot_model_parameters,
    compute_parameter_displacement,
)
from task1.data import prepare_and_filter_dataset
from task1.models import (
    apply_lora_to_model,
    apply_oplora_to_model,
    merge_adapters_into_backbone,
    compute_layer_svd,
)
from task1.trainer import train_variant
from task1.evaluator import evaluate_checkpoint_condition
from task1.diagnostics import (
    compute_weight_space_metrics,
    sample_activation_probe_set,
    ActivationProbeCacher,
    compute_activation_ratio,
)
from task1.qualitative import (
    load_qualitative_prompts,
    generate_response,
    generate_qualitative_suite,
)


def load_fresh_qwen_model(config: Task1Config, device: torch.device, is_adapter: bool = False) -> Tuple[nn.Module, Any]:
    """
    Load a fresh copy of Qwen/Qwen3-0.6B with thinking disabled and SDPA.
    Per PDF invariant:
    - Full FT variants use FP32 master parameters.
    - Adapter variants use a frozen FP16 backbone and FP32 adapters.
    """
    model_name = config.model.model_name_or_path
    print(f"\n[Model] Loading fresh checkpoint: {model_name} (is_adapter={is_adapter})...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Invariant: Full FT variants use FP32 master parameters; adapter variants use a frozen FP16 backbone
    if is_adapter and config.training.fp16 and device.type == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            attn_implementation=config.model.attn_implementation,
            trust_remote_code=True,
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=config.model.attn_implementation,
            trust_remote_code=True,
        ).to(device)

    if hasattr(model.config, "enable_thinking"):
        model.config.enable_thinking = False

    return model, tokenizer


def load_condition_model(
    condition_name: str,
    config: Task1Config,
    device: torch.device,
) -> Tuple[nn.Module, Any]:
    """
    Load a model corresponding to a given experimental condition directly from disk checkpoints.
    If LoRA/OPLoRA, merges the adapter into the backbone so downstream diagnostics and qualitative
    generation receive standard linear layers.
    """
    ckpt_dir = config.paths.checkpoint_dir

    if condition_name == "baseline":
        return load_fresh_qwen_model(config, device)

    if condition_name == "full_ft_midpoint":
        model, tokenizer = load_fresh_qwen_model(config, device)
        ckpt_path = os.path.join(ckpt_dir, "full_ft", "checkpoint_update_625.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        return model, tokenizer

    if condition_name == "full_ft_final":
        model, tokenizer = load_fresh_qwen_model(config, device)
        ckpt_path = os.path.join(ckpt_dir, "full_ft", "checkpoint_update_1250.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        return model, tokenizer

    if condition_name == "full_ft_l2sp_final":
        model, tokenizer = load_fresh_qwen_model(config, device)
        ckpt_path = os.path.join(ckpt_dir, "full_ft_l2sp", "checkpoint_update_1250.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        return model, tokenizer

    if condition_name == "lora_4_final":
        model, tokenizer = load_fresh_qwen_model(config, device, is_adapter=True)
        lora_cfg = config.methods.get("lora_4", {})
        apply_lora_to_model(
            model,
            rank=lora_cfg.get("rank", 4),
            alpha=lora_cfg.get("alpha", 8),
            dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", None),
        )
        ckpt_path = os.path.join(ckpt_dir, "lora_4", "checkpoint_update_1250.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        merged = merge_adapters_into_backbone(model)
        return merged, tokenizer

    if condition_name == "lora_16_final":
        model, tokenizer = load_fresh_qwen_model(config, device, is_adapter=True)
        lora_cfg = config.methods.get("lora_16", {})
        apply_lora_to_model(
            model,
            rank=lora_cfg.get("rank", 16),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", None),
        )
        ckpt_path = os.path.join(ckpt_dir, "lora_16", "checkpoint_update_1250.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        merged = merge_adapters_into_backbone(model)
        return merged, tokenizer

    if condition_name == "oplora_16_final":
        model, tokenizer = load_fresh_qwen_model(config, device, is_adapter=True)
        oplora_cfg = config.methods.get("oplora_16", {})
        model, _ = apply_oplora_to_model(
            model,
            rank=oplora_cfg.get("rank", 16),
            alpha=oplora_cfg.get("alpha", 32),
            protection_rank_k=oplora_cfg.get("protection_rank_k", 128),
            dropout=oplora_cfg.get("dropout", 0.05),
            target_modules=oplora_cfg.get("target_modules", None),
            device=device,
        )
        ckpt_path = os.path.join(ckpt_dir, "oplora_16", "checkpoint_update_1250.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        merged = merge_adapters_into_backbone(model)
        return merged, tokenizer

    raise ValueError(f"Unknown condition name: {condition_name}")


def main():
    parser = argparse.ArgumentParser(description="ATML PA3 Task 1 Orchestrator")
    parser.add_argument("--config", type=str, default="configs/task1_config.yaml", help="Path to config YAML")
    parser.add_argument("--stages", type=str, nargs="+", default=["all"],
                        help="Stages to run: [all, baseline, train_full_ft, train_l2sp, train_lora4, train_lora16, train_oplora, diagnostics, qualitative, synthesize]")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cuda:0", "cuda:1", "cpu"],
                        help="Target device (auto, cuda, cuda:0, cuda:1, cpu)")
    parser.add_argument("--single-gpu", action="store_true", help="Force single-GPU execution without parallel subprocess orchestration")
    parser.add_argument("--skip-eval", action="store_true", help="Skip benchmark evaluation (for debugging)")
    args = parser.parse_args()

    # Normalize args.stages to cleanly handle multi-stage strings or comma-separated lists
    expanded_stages = []
    for s in args.stages:
        for piece in s.replace(",", " ").split():
            expanded_stages.append(piece.strip())
    args.stages = expanded_stages if expanded_stages else ["all"]

    config = Task1Config.from_yaml(args.config)
    set_seed(config.model.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[Device] Using compute device: {device}")
    if device.type == "cuda":
        dev_idx = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(dev_idx)
        vram_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024**3)
        num_gpus = torch.cuda.device_count()
        print(f"[Hardware] GPU {dev_idx}: {gpu_name} ({vram_gb:.1f} GB VRAM) | Total GPUs Available: {num_gpus}")
        # Performance optimizations
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    os.makedirs(config.paths.output_dir, exist_ok=True)
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)

    # 1. Prepare Data
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION & CACHING")
    print("="*80)
    temp_tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path, trust_remote_code=True)
    if temp_tokenizer.pad_token_id is None:
        temp_tokenizer.pad_token_id = temp_tokenizer.eos_token_id

    cache_file = os.path.join(config.data.cache_dir, "metamath_sft_filtered_40k.pt")
    dataset = prepare_and_filter_dataset(
        tokenizer=temp_tokenizer,
        dataset_name=config.data.dataset_name,
        max_examples=config.data.max_examples,
        max_seq_length=config.data.max_seq_length,
        filter_gsm8k_leakage=config.data.filter_gsm8k_leakage,
        cache_path=cache_file,
        seed=config.model.seed,
    )

    all_eval_results: Dict[str, Dict[str, float]] = {}
    training_histories: Dict[str, List[Dict[str, Any]]] = {}
    resource_stats: Dict[str, Dict[str, Any]] = {}
    diagnostic_results: Dict[str, Any] = {}
    models_for_qualitative: Dict[str, Tuple[nn.Module, Any]] = {}
    baseline_W0: Dict[int, torch.Tensor] = {}
    cached_probe_z: Dict[int, List[torch.Tensor]] = {}

    run_all = "all" in args.stages

    # 2. Stage: Baseline Evaluation
    if run_all or "baseline" in args.stages:
        print("\n" + "="*80)
        print("STEP 2: BASELINE EVALUATION (UPDATE 0 - UNCHANGED STARTING MODEL)")
        print("="*80)
        base_model, tokenizer = load_fresh_qwen_model(config, device)
        total_p, train_p = count_parameters(base_model)
        resource_stats["baseline"] = {
            "trainable_params": train_p,
            "total_params": total_p,
            "training_time_sec": 0.0,
            "peak_vram_mb": get_peak_vram_mb(),
            "checkpoint_size_mb": 0.0,
        }

        if not args.skip_eval:
            b_eval = evaluate_checkpoint_condition(base_model, tokenizer, "baseline", device, config, config.paths.output_dir)
            all_eval_results["baseline"] = b_eval["scores"]

        print("[Baseline] Caching probe activations for activation diagnostics...")
        probe_set = sample_activation_probe_set(seed=config.diagnostics.seed)
        cacher = ActivationProbeCacher(target_layers=config.diagnostics.target_layers)
        cacher.register(base_model)

        base_model.eval()
        with torch.no_grad():
            for p_item in probe_set["math"] + probe_set["non_math"]:
                prompt_txt = p_item.get("question", p_item.get("context", ""))
                toks = tokenizer(prompt_txt, return_tensors="pt").to(device)
                base_model(**toks)

        cached_probe_z = {l: [z.clone() for z in cacher.cached_z[l]] for l in config.diagnostics.target_layers}
        cacher.clear()

        baseline_W0 = {}
        for l in config.diagnostics.target_layers:
            mod = base_model.model.layers[l].mlp.down_proj
            baseline_W0[l] = mod.weight.detach().cpu().float().clone()

        cache_path = os.path.join(config.paths.checkpoint_dir, "baseline_cache.pt")
        try:
            torch.save({
                "baseline_W0": baseline_W0,
                "cached_probe_z": cached_probe_z,
            }, cache_path)
            print(f"[Baseline] Saved probe cache and down_proj W0 to {cache_path}")
        except Exception as e:
            print(f"[Warning] Failed to save baseline cache to {cache_path}: {e}")

        base_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        models_for_qualitative["baseline"] = (base_model, tokenizer)
        trigger_modal_stage_commit("baseline")

    can_parallelize_training = (
        torch.cuda.device_count() >= 2
        and not args.single_gpu
        and device.type == "cuda"
        and device.index is None
    )

    if run_all and can_parallelize_training:
        print("\n" + "="*80)
        print("DUAL-GPU ORCHESTRATION ACTIVATED (2x NVIDIA Tesla T4 DETECTED)")
        print("Running independent training variants concurrently across cuda:0 and cuda:1")
        print("="*80)

        eval_flag = ["--skip-eval"] if args.skip_eval else []

        print("\n" + "="*80)
        print("STEP 3 & 4: PARALLEL TRAINING PAIR 1 (cuda:0: Full FT || cuda:1: Full FT + L2-SP)")
        print("="*80)
        p1 = subprocess.Popen([sys.executable, "-m", "task1.run_all", "--config", args.config, "--stages", "train_full_ft", "--device", "cuda:0", "--single-gpu"] + eval_flag)
        p2 = subprocess.Popen([sys.executable, "-m", "task1.run_all", "--config", args.config, "--stages", "train_l2sp", "--device", "cuda:1", "--single-gpu"] + eval_flag)
        exit1 = p1.wait()
        exit2 = p2.wait()
        if exit1 != 0 or exit2 != 0:
            raise RuntimeError(f"Parallel training pair 1 failed (exit1={exit1}, exit2={exit2})")

        print("\n" + "="*80)
        print("STEP 5 & 6: PARALLEL TRAINING PAIR 2 (cuda:0: LoRA-4 || cuda:1: LoRA-16)")
        print("="*80)
        p3 = subprocess.Popen([sys.executable, "-m", "task1.run_all", "--config", args.config, "--stages", "train_lora4", "--device", "cuda:0", "--single-gpu"] + eval_flag)
        p4 = subprocess.Popen([sys.executable, "-m", "task1.run_all", "--config", args.config, "--stages", "train_lora16", "--device", "cuda:1", "--single-gpu"] + eval_flag)
        exit3 = p3.wait()
        exit4 = p4.wait()
        if exit3 != 0 or exit4 != 0:
            raise RuntimeError(f"Parallel training pair 2 failed (exit3={exit3}, exit4={exit4})")

        print("\n" + "="*80)
        print("STEP 7: TRAINING OPLoRA-16 (cuda:0)")
        print("="*80)
        p5 = subprocess.Popen([sys.executable, "-m", "task1.run_all", "--config", args.config, "--stages", "train_oplora", "--device", "cuda:0", "--single-gpu"] + eval_flag)
        exit5 = p5.wait()
        if exit5 != 0:
            raise RuntimeError(f"Training OPLoRA failed (exit5={exit5})")

    else:
        # 3. Stage: Full FT
        if run_all or "train_full_ft" in args.stages:
            print("\n" + "="*80)
            print("STEP 3: FULL FINE-TUNING (FULL FT)")
            print("="*80)
            model, tokenizer = load_fresh_qwen_model(config, device)
            ckpt_cb = get_modal_checkpoint_callback()
            logs = train_variant("full_ft", model, dataset, config, device, config.paths.checkpoint_dir, checkpoint_callback=ckpt_cb)
            training_histories["full_ft"] = logs["history"]
            resource_stats["full_ft"] = {
                "trainable_params": logs["trainable_params"],
                "total_params": logs["total_params"],
                "training_time_sec": logs["total_train_time_sec"],
                "peak_vram_mb": logs["peak_vram_mb"],
                "checkpoint_size_mb": logs["checkpoints"].get("1250", {}).get("size_mb", 0.0),
                "displacements": logs["displacements"],
            }

            if "625" in logs["checkpoints"] and not args.skip_eval:
                mid_ckpt = logs["checkpoints"]["625"]["path"]
                mid_model, _ = load_fresh_qwen_model(config, device)
                mid_payload = torch.load(mid_ckpt, map_location=device)
                mid_model.load_state_dict(mid_payload["model_state_dict"])
                mid_eval = evaluate_checkpoint_condition(mid_model, tokenizer, "full_ft_midpoint", device, config, config.paths.output_dir)
                all_eval_results["full_ft_midpoint"] = mid_eval["scores"]
                mid_model.cpu()
                models_for_qualitative["full_ft_midpoint"] = (mid_model, tokenizer)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not args.skip_eval:
                final_eval = evaluate_checkpoint_condition(model, tokenizer, "full_ft_final", device, config, config.paths.output_dir)
                all_eval_results["full_ft_final"] = final_eval["scores"]
            model.cpu()
            models_for_qualitative["full_ft_final"] = (model, tokenizer)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trigger_modal_stage_commit("full_ft")

        # 4. Stage: Full FT + L2-SP
        if run_all or "train_l2sp" in args.stages:
            print("\n" + "="*80)
            print("STEP 4: FULL FT WITH L2-SP REGULARIZATION")
            print("="*80)
            model, tokenizer = load_fresh_qwen_model(config, device)
            ckpt_cb = get_modal_checkpoint_callback()
            logs = train_variant("full_ft_l2sp", model, dataset, config, device, config.paths.checkpoint_dir, checkpoint_callback=ckpt_cb)
            training_histories["full_ft_l2sp"] = logs["history"]
            resource_stats["full_ft_l2sp"] = {
                "trainable_params": logs["trainable_params"],
                "total_params": logs["total_params"],
                "training_time_sec": logs["total_train_time_sec"],
                "peak_vram_mb": logs["peak_vram_mb"],
                "checkpoint_size_mb": logs["checkpoints"].get("1250", {}).get("size_mb", 0.0),
                "displacements": logs["displacements"],
                "ref_storage_mb": logs["ref_storage_mb"],
            }
            if not args.skip_eval:
                final_eval = evaluate_checkpoint_condition(model, tokenizer, "full_ft_l2sp_final", device, config, config.paths.output_dir)
                all_eval_results["full_ft_l2sp_final"] = final_eval["scores"]
            model.cpu()
            models_for_qualitative["full_ft_l2sp_final"] = (model, tokenizer)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trigger_modal_stage_commit("full_ft_l2sp")

        # 5. Stage: LoRA-4
        if run_all or "train_lora4" in args.stages:
            print("\n" + "="*80)
            print("STEP 5: LoRA AT RANK 4")
            print("="*80)
            model, tokenizer = load_fresh_qwen_model(config, device, is_adapter=True)
            lora_cfg = config.methods.get("lora_4", {})
            apply_lora_to_model(
                model,
                rank=lora_cfg.get("rank", 4),
                alpha=lora_cfg.get("alpha", 8),
                dropout=lora_cfg.get("dropout", 0.05),
                target_modules=lora_cfg.get("target_modules", None),
            )
            ckpt_cb = get_modal_checkpoint_callback()
            logs = train_variant("lora_4", model, dataset, config, device, config.paths.checkpoint_dir, checkpoint_callback=ckpt_cb)
            training_histories["lora_4"] = logs["history"]
            resource_stats["lora_4"] = {
                "trainable_params": logs["trainable_params"],
                "total_params": logs["total_params"],
                "training_time_sec": logs["total_train_time_sec"],
                "peak_vram_mb": logs["peak_vram_mb"],
                "checkpoint_size_mb": logs["checkpoints"].get("1250", {}).get("size_mb", 0.0),
            }
            merged_model = merge_adapters_into_backbone(copy.deepcopy(model), verify_logits_parity=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not args.skip_eval:
                final_eval = evaluate_checkpoint_condition(merged_model, tokenizer, "lora_4_final", device, config, config.paths.output_dir)
                all_eval_results["lora_4_final"] = final_eval["scores"]
            merged_model.cpu()
            models_for_qualitative["lora_4_final"] = (merged_model, tokenizer)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trigger_modal_stage_commit("lora_4")

        # 6. Stage: LoRA-16
        if run_all or "train_lora16" in args.stages:
            print("\n" + "="*80)
            print("STEP 6: LoRA AT RANK 16")
            print("="*80)
            model, tokenizer = load_fresh_qwen_model(config, device, is_adapter=True)
            lora_cfg = config.methods.get("lora_16", {})
            apply_lora_to_model(
                model,
                rank=lora_cfg.get("rank", 16),
                alpha=lora_cfg.get("alpha", 32),
                dropout=lora_cfg.get("dropout", 0.05),
                target_modules=lora_cfg.get("target_modules", None),
            )
            ckpt_cb = get_modal_checkpoint_callback()
            logs = train_variant("lora_16", model, dataset, config, device, config.paths.checkpoint_dir, checkpoint_callback=ckpt_cb)
            training_histories["lora_16"] = logs["history"]
            resource_stats["lora_16"] = {
                "trainable_params": logs["trainable_params"],
                "total_params": logs["total_params"],
                "training_time_sec": logs["total_train_time_sec"],
                "peak_vram_mb": logs["peak_vram_mb"],
                "checkpoint_size_mb": logs["checkpoints"].get("1250", {}).get("size_mb", 0.0),
            }
            merged_model = merge_adapters_into_backbone(copy.deepcopy(model), verify_logits_parity=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not args.skip_eval:
                final_eval = evaluate_checkpoint_condition(merged_model, tokenizer, "lora_16_final", device, config, config.paths.output_dir)
                all_eval_results["lora_16_final"] = final_eval["scores"]
            merged_model.cpu()
            models_for_qualitative["lora_16_final"] = (merged_model, tokenizer)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trigger_modal_stage_commit("lora_16")

        # 7. Stage: OPLoRA-16
        if run_all or "train_oplora" in args.stages:
            print("\n" + "="*80)
            print("STEP 7: OPLoRA AT RANK 16 (PROTECTION RANK 128)")
            print("="*80)
            model, tokenizer = load_fresh_qwen_model(config, device, is_adapter=True)
            oplora_cfg = config.methods.get("oplora_16", {})
            model, svd_stats = apply_oplora_to_model(
                model,
                rank=oplora_cfg.get("rank", 16),
                alpha=oplora_cfg.get("alpha", 32),
                protection_rank_k=oplora_cfg.get("protection_rank_k", 128),
                dropout=oplora_cfg.get("dropout", 0.05),
                target_modules=oplora_cfg.get("target_modules", None),
                device=device,
            )
            ckpt_cb = get_modal_checkpoint_callback()
            logs = train_variant("oplora_16", model, dataset, config, device, config.paths.checkpoint_dir, checkpoint_callback=ckpt_cb)
            training_histories["oplora_16"] = logs["history"]
            resource_stats["oplora_16"] = {
                "trainable_params": logs["trainable_params"],
                "total_params": logs["total_params"],
                "training_time_sec": logs["total_train_time_sec"],
                "peak_vram_mb": logs["peak_vram_mb"],
                "checkpoint_size_mb": logs["checkpoints"].get("1250", {}).get("size_mb", 0.0),
                "svd_preprocessing_time_sec": svd_stats["preprocessing_time_sec"],
                "basis_storage_mb": svd_stats["basis_storage_mb"],
            }
            merged_model = merge_adapters_into_backbone(copy.deepcopy(model), verify_logits_parity=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not args.skip_eval:
                final_eval = evaluate_checkpoint_condition(merged_model, tokenizer, "oplora_16_final", device, config, config.paths.output_dir)
                all_eval_results["oplora_16_final"] = final_eval["scores"]
            merged_model.cpu()
            models_for_qualitative["oplora_16_final"] = (merged_model, tokenizer)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trigger_modal_stage_commit("oplora_16")

    # 8. Stage: Diagnostics (Weight-space & Activation)
    if run_all or "diagnostics" in args.stages:
        print("\n" + "="*80)
        print("STEP 8: WEIGHT-SPACE & ACTIVATION DIAGNOSTICS")
        print("="*80)
        cache_path = os.path.join(config.paths.checkpoint_dir, "baseline_cache.pt")
        if (not baseline_W0 or not cached_probe_z) and os.path.exists(cache_path):
            print(f"[Diagnostics] Loading cached baseline W0 and probes from {cache_path}...")
            cache_payload = torch.load(cache_path, map_location="cpu")
            baseline_W0 = cache_payload.get("baseline_W0", {})
            cached_probe_z = cache_payload.get("cached_probe_z", {})

        required_conditions = ["full_ft_midpoint", "full_ft_final", "full_ft_l2sp_final", "lora_4_final", "lora_16_final", "oplora_16_final"]
        for cond_name in required_conditions:
            if cond_name not in models_for_qualitative:
                try:
                    c_model, c_tok = load_condition_model(cond_name, config, device=torch.device("cpu"))
                    models_for_qualitative[cond_name] = (c_model, c_tok)
                except Exception as e:
                    print(f"[Diagnostics] Note: {cond_name} not yet available ({e})")

        diag_data = {}
        for cond_name, (cond_model, _) in models_for_qualitative.items():
            if cond_name == "baseline":
                continue
            diag_data[cond_name] = {"weight_space": {}, "activation_ratio": {}}

            for layer_idx in config.diagnostics.target_layers:
                mod = cond_model.model.layers[layer_idx].mlp.down_proj
                curr_w = mod.weight.detach().cpu().float()
                w0 = baseline_W0.get(layer_idx, None)
                if w0 is None:
                    continue
                delta_w = curr_w - w0

                U_k, _ = compute_layer_svd(w0, k=config.diagnostics.protection_rank_k)
                w_metrics = compute_weight_space_metrics(w0, delta_w, U_k)
                diag_data[cond_name]["weight_space"][f"layer_{layer_idx}"] = w_metrics

                if layer_idx in cached_probe_z:
                    z_math = cached_probe_z[layer_idx][:config.diagnostics.num_probe_gsm8k]
                    z_nonmath = cached_probe_z[layer_idx][config.diagnostics.num_probe_gsm8k:]
                    r_math = compute_activation_ratio(w0, delta_w, z_math)
                    r_nonmath = compute_activation_ratio(w0, delta_w, z_nonmath)

                    diag_data[cond_name]["activation_ratio"][f"layer_{layer_idx}"] = {
                        "R_z_math": r_math,
                        "R_z_non_math": r_nonmath,
                    }

        diag_path = os.path.join(config.paths.output_dir, "diagnostics.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag_data, f, indent=2)
        print(f"[Diagnostics] Saved diagnostic metrics to {diag_path}")
        diagnostic_results = diag_data
        trigger_modal_stage_commit("diagnostics")

    # 9. Stage: Qualitative Suite & 42-Response Annotation
    if run_all or "qualitative" in args.stages:
        print("\n" + "="*80)
        print("STEP 9: QUALITATIVE PROMPT EVALUATION & 42-RESPONSE ANNOTATION")
        print("="*80)
        all_qual_conditions = ["baseline", "full_ft_midpoint", "full_ft_final", "full_ft_l2sp_final", "lora_4_final", "lora_16_final", "oplora_16_final"]
        for cond_name in all_qual_conditions:
            if cond_name not in models_for_qualitative:
                try:
                    c_model, c_tok = load_condition_model(cond_name, config, device=torch.device("cpu"))
                    models_for_qualitative[cond_name] = (c_model, c_tok)
                except Exception as e:
                    print(f"[Qualitative] Note: {cond_name} not yet available ({e})")

        generate_qualitative_suite(
            models_dict=models_for_qualitative,
            prompts_path=config.qualitative.prompts_file,
            max_new_tokens=config.qualitative.max_new_tokens,
            output_json_path=os.path.join(config.paths.output_dir, "qualitative_responses.json"),
            annotation_md_path=os.path.join(config.paths.output_dir, "qualitative_annotations.md"),
            device=device,
        )
        trigger_modal_stage_commit("qualitative")

    # 10. Stage: Synthesis & Consolidated Reporting
    if run_all or "synthesize" in args.stages:
        print("\n" + "="*80)
        print("STEP 10: SYNTHESIS & REPORTING")
        print("="*80)
        for cond in ["baseline", "full_ft_midpoint", "full_ft_final", "full_ft_l2sp_final", "lora_4_final", "lora_16_final", "oplora_16_final"]:
            if cond not in all_eval_results:
                e_file = os.path.join(config.paths.output_dir, f"eval_summary_{cond}.json")
                if os.path.exists(e_file):
                    try:
                        with open(e_file, "r", encoding="utf-8") as f:
                            edata = json.load(f)
                            all_eval_results[cond] = edata.get("scores", {})
                    except Exception:
                        pass

        for var in ["full_ft", "full_ft_l2sp", "lora_4", "lora_16", "oplora_16"]:
            t_file = os.path.join(config.paths.checkpoint_dir, var, "training_summary.json")
            if os.path.exists(t_file):
                try:
                    with open(t_file, "r", encoding="utf-8") as f:
                        tdata = json.load(f)
                        if var not in training_histories:
                            training_histories[var] = tdata.get("history", [])
                        if var not in resource_stats:
                            resource_stats[var] = {
                                "trainable_params": tdata.get("trainable_params", 0),
                                "total_params": tdata.get("total_params", 0),
                                "training_time_sec": tdata.get("total_train_time_sec", 0.0),
                                "peak_vram_mb": tdata.get("peak_vram_mb", 0.0),
                                "checkpoint_size_mb": tdata.get("checkpoints", {}).get("1250", {}).get("size_mb", 0.0),
                                "displacements": tdata.get("displacements", {}),
                            }
                except Exception:
                    pass

        diag_path = os.path.join(config.paths.output_dir, "diagnostics.json")
        if not diagnostic_results and os.path.exists(diag_path):
            try:
                with open(diag_path, "r", encoding="utf-8") as f:
                    diagnostic_results = json.load(f)
            except Exception:
                pass

        baseline_scores = all_eval_results.get("baseline", {})
        summary_table = []

        for cond, scores in all_eval_results.items():
            row = {"condition": cond}
            for bench in ["gsm8k", "hellaswag", "boolq", "arc_easy"]:
                score = scores.get(bench, 0.0)
                base = baseline_scores.get(bench, 0.0)
                delta = score - base
                row[f"{bench}_score"] = score
                row[f"{bench}_delta_pp"] = delta if cond != "baseline" else 0.0
            summary_table.append(row)

        summary_df = pd.DataFrame(summary_table)
        print("\n--- BENCHMARK RESULTS & BASELINE DELTAS (PERCENTAGE POINTS) ---")
        print(summary_df.to_string(index=False))

        final_summary = {
            "benchmarks": all_eval_results,
            "benchmark_deltas_pp": summary_table,
            "resources": resource_stats,
            "diagnostics": diagnostic_results,
        }
        sum_path = os.path.join(config.paths.output_dir, "metrics_summary.json")
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2)
        print(f"\n[Synthesis] Saved consolidated metrics summary to {sum_path}")

        all_curves = []
        for variant, hist in training_histories.items():
            for entry in hist:
                row = {"variant": variant, **entry}
                all_curves.append(row)
        if all_curves:
            curves_df = pd.DataFrame(all_curves)
            curves_path = os.path.join(config.paths.output_dir, "learning_curves.csv")
            curves_df.to_csv(curves_path, index=False)
            print(f"[Synthesis] Saved training learning curves to {curves_path}")

        trigger_modal_stage_commit("synthesize")

        print("\n" + "="*80)
        print("ALL TASK 1 EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)


if __name__ == "__main__":
    main()
