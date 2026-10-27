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
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

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


def main():
    parser = argparse.ArgumentParser(description="ATML PA3 Task 1 Orchestrator")
    parser.add_argument("--config", type=str, default="configs/task1_config.yaml", help="Path to config YAML")
    parser.add_argument("--stages", type=str, nargs="+", default=["all"],
                        help="Stages to run: [all, baseline, train_full_ft, train_l2sp, train_lora4, train_lora16, train_oplora, diagnostics, qualitative, synthesize]")
    parser.add_argument("--skip-eval", action="store_true", help="Skip benchmark evaluation (for debugging)")
    args = parser.parse_args()

    config = Task1Config.from_yaml(args.config)
    set_seed(config.model.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using compute device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Hardware] GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) - Ampere/Hopper/Blackwell Tensor Cores active.")
        # Ampere/Hopper/Blackwell high-throughput performance optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

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

        models_for_qualitative["baseline"] = (base_model, tokenizer)

    # 3. Stage: Full FT
    if run_all or "train_full_ft" in args.stages:
        print("\n" + "="*80)
        print("STEP 3: FULL FINE-TUNING (FULL FT)")
        print("="*80)
        model, tokenizer = load_fresh_qwen_model(config, device)
        logs = train_variant("full_ft", model, dataset, config, device, config.paths.checkpoint_dir)
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
            models_for_qualitative["full_ft_midpoint"] = (mid_model, tokenizer)

        if not args.skip_eval:
            final_eval = evaluate_checkpoint_condition(model, tokenizer, "full_ft_final", device, config, config.paths.output_dir)
            all_eval_results["full_ft_final"] = final_eval["scores"]
        models_for_qualitative["full_ft_final"] = (model, tokenizer)

    # 4. Stage: Full FT + L2-SP
    if run_all or "train_l2sp" in args.stages:
        print("\n" + "="*80)
        print("STEP 4: FULL FT WITH L2-SP REGULARIZATION")
        print("="*80)
        model, tokenizer = load_fresh_qwen_model(config, device)
        logs = train_variant("full_ft_l2sp", model, dataset, config, device, config.paths.checkpoint_dir)
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
        models_for_qualitative["full_ft_l2sp_final"] = (model, tokenizer)

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
        logs = train_variant("lora_4", model, dataset, config, device, config.paths.checkpoint_dir)
        training_histories["lora_4"] = logs["history"]
        resource_stats["lora_4"] = {
            "trainable_params": logs["trainable_params"],
            "total_params": logs["total_params"],
            "training_time_sec": logs["total_train_time_sec"],
            "peak_vram_mb": logs["peak_vram_mb"],
            "checkpoint_size_mb": logs["checkpoints"].get("1250", {}).get("size_mb", 0.0),
        }
        merged_model = merge_adapters_into_backbone(copy.deepcopy(model), verify_logits_parity=True)
        if not args.skip_eval:
            final_eval = evaluate_checkpoint_condition(merged_model, tokenizer, "lora_4_final", device, config, config.paths.output_dir)
            all_eval_results["lora_4_final"] = final_eval["scores"]
        models_for_qualitative["lora_4_final"] = (merged_model, tokenizer)

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
        logs = train_variant("lora_16", model, dataset, config, device, config.paths.checkpoint_dir)
        training_histories["lora_16"] = logs["history"]
        resource_stats["lora_16"] = {
            "trainable_params": logs["trainable_params"],
            "total_params": logs["total_params"],
            "training_time_sec": logs["total_train_time_sec"],
            "peak_vram_mb": logs["peak_vram_mb"],
            "checkpoint_size_mb": logs["checkpoints"].get("1250", {}).get("size_mb", 0.0),
        }
        merged_model = merge_adapters_into_backbone(copy.deepcopy(model), verify_logits_parity=True)
        if not args.skip_eval:
            final_eval = evaluate_checkpoint_condition(merged_model, tokenizer, "lora_16_final", device, config, config.paths.output_dir)
            all_eval_results["lora_16_final"] = final_eval["scores"]
        models_for_qualitative["lora_16_final"] = (merged_model, tokenizer)

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
        logs = train_variant("oplora_16", model, dataset, config, device, config.paths.checkpoint_dir)
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
        if not args.skip_eval:
            final_eval = evaluate_checkpoint_condition(merged_model, tokenizer, "oplora_16_final", device, config, config.paths.output_dir)
            all_eval_results["oplora_16_final"] = final_eval["scores"]
        models_for_qualitative["oplora_16_final"] = (merged_model, tokenizer)

    # 8. Stage: Diagnostics (Weight-space & Activation)
    if run_all or "diagnostics" in args.stages:
        print("\n" + "="*80)
        print("STEP 8: WEIGHT-SPACE & ACTIVATION DIAGNOSTICS")
        print("="*80)
        diag_data = {}
        for cond_name, (cond_model, _) in models_for_qualitative.items():
            if cond_name == "baseline":
                continue
            diag_data[cond_name] = {"weight_space": {}, "activation_ratio": {}}

            for layer_idx in config.diagnostics.target_layers:
                mod = cond_model.model.layers[layer_idx].mlp.down_proj
                curr_w = mod.weight.detach().cpu().float()
                w0 = baseline_W0[layer_idx]
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

    # 9. Stage: Qualitative Suite & 42-Response Annotation
    if run_all or "qualitative" in args.stages:
        print("\n" + "="*80)
        print("STEP 9: QUALITATIVE PROMPT EVALUATION & 42-RESPONSE ANNOTATION")
        print("="*80)
        generate_qualitative_suite(
            models_dict=models_for_qualitative,
            prompts_path=config.qualitative.prompts_file,
            max_new_tokens=config.qualitative.max_new_tokens,
            output_json_path=os.path.join(config.paths.output_dir, "qualitative_responses.json"),
            annotation_md_path=os.path.join(config.paths.output_dir, "qualitative_annotations.md"),
        )

    # 10. Stage: Synthesis & Consolidated Reporting
    if run_all or "synthesize" in args.stages:
        print("\n" + "="*80)
        print("STEP 10: SYNTHESIS & REPORTING")
        print("="*80)
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

        print("\n" + "="*80)
        print("ALL TASK 1 EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)


if __name__ == "__main__":
    main()
