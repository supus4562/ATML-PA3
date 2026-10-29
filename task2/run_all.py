"""
Master orchestrator for ATML PA3 Task 2: Vision-Language Models and Visual Grounding.
Executes data preparation, SigLIP feature caching, connector calibration,
3-stage curriculum training, 4-condition grounding evaluation with 4,000 pair bootstraps,
and consolidated report synthesis.
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List, Optional, Any

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from task2.config import Task2Config
from task2.vision import SigLIPFeatureExtractor, cache_image_features, cache_real_coco_features
from task2.models import VisualConnector, VLMModel, calibrate_connector_scale
from task2.data import prepare_task2_dataset_splits
from task2.trainer import train_task2
from task2.evaluator import evaluate_test_suite_4_conditions


def set_seed(seed: int = 6304):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trigger_modal_stage_commit(stage_name: str):
    """If running on Modal cloud, trigger volume commit after each stage."""
    try:
        import modal
        vol = modal.Volume.from_name("pa3-volume")
        vol.commit()
        print(f"[Modal Volume] Committed stage '{stage_name}' to cloud storage.", flush=True)
    except Exception:
        pass


def get_modal_checkpoint_callback():
    """Returns a callback to commit the cloud volume after each checkpoint."""
    def callback(update_step: int, checkpoint_path: str):
        try:
            import modal
            vol = modal.Volume.from_name("pa3-volume")
            vol.commit()
            print(f"[Modal Volume] Committed checkpoint {update_step} to volume 'pa3-volume'", flush=True)
        except Exception:
            pass
    return callback


def main():
    parser = argparse.ArgumentParser(description="ATML PA3 Task 2 Orchestrator (VLM & Grounding)")
    parser.add_argument("--config", type=str, default="configs/task2_config.yaml", help="Path to config YAML")
    parser.add_argument("--stages", type=str, nargs="+", default=["all"],
                        help="Stages to run: [all, data, stage1, stage2, stage3, evaluate, synthesize]")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cuda:0", "cpu"],
                        help="Target compute device")
    parser.add_argument("--single-gpu", action="store_true", help="Single GPU flag")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    args = parser.parse_args()

    # Normalize stages
    expanded_stages = []
    for s in args.stages:
        for piece in s.replace(",", " ").split():
            expanded_stages.append(piece.strip())
    args.stages = expanded_stages if expanded_stages else ["all"]

    config = Task2Config.from_yaml(args.config)
    set_seed(config.training.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("\n" + "=" * 80)
    print("ATML PA3 - TASK 2: VISION-LANGUAGE MODELS & VISUAL GROUNDING")
    print("=" * 80)
    print(f"[Device] Using compute device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Hardware] GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(config.paths.output_dir, exist_ok=True)
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)

    run_all = "all" in args.stages

    # 1. Stage: Data Preparation & Split Isolation
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPARATION & SPLIT ISOLATION")
    print("=" * 80)
    data_splits = prepare_task2_dataset_splits(
        cache_dir=config.data.cache_dir,
        seed=config.data.seed,
    )
    trigger_modal_stage_commit("data")

    if args.stages == ["data"]:
        print("[Task 2] Data preparation complete. Exiting as requested.")
        return

    # 2. Stage: Vision Feature Caching (SigLIP)
    if run_all or any(s in args.stages for s in ["stage1", "stage2", "stage3", "evaluate", "cache_features"]):
        print("\n" + "=" * 80)
        print("STEP 2: SIGLIP FEATURE EXTRACTION & CACHING (FP16)")
        print("=" * 80)
        os.makedirs(config.data.feature_cache_dir, exist_ok=True)
        cached_count = len([f for f in os.listdir(config.data.feature_cache_dir) if f.endswith(".pt")])
        print(f"[Vision Cache] Found {cached_count} pre-cached feature files in {config.data.feature_cache_dir}.")

        all_ids = data_splits["all_image_ids"]
        split_map = data_splits["image_split_map"]
        suffix = "_comp49.pt" if config.vision.compress_2x2 else "_196.pt"
        missing_count = sum(1 for img_id in all_ids if not os.path.exists(os.path.join(config.data.feature_cache_dir, f"img_{img_id}{suffix}")))

        if missing_count > 0:
            print(f"[Vision Cache] Extracting real SigLIP features for {missing_count} missing images on {device}...")
            extractor = SigLIPFeatureExtractor(
                model_name=config.vision.model_name_or_path,
                device=device,
                dtype=torch.float16,
            )
            cache_real_coco_features(
                image_ids=all_ids,
                image_split_map=split_map,
                cache_dir=config.data.feature_cache_dir,
                extractor=extractor,
                batch_size=64,
                max_workers=32,
                compress=config.vision.compress_2x2,
            )
            extractor.unload()
        else:
            print(f"[Vision Cache] All {len(all_ids)} required image features are already cached!")

        trigger_modal_stage_commit("vision_cache")

    # 3. Load Qwen and Visual Connector
    print("\n" + "=" * 80)
    print("STEP 3: INITIALIZING QWEN-0.6B LANGUAGE MODEL & VISUAL CONNECTOR")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(
        config.language_model.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.float16 if (config.training.fp16 and device.type == "cuda") else torch.float32
    qwen = AutoModelForCausalLM.from_pretrained(
        config.language_model.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=config.language_model.attn_implementation,
        trust_remote_code=True,
    ).to(device)

    if hasattr(qwen.config, "enable_thinking"):
        qwen.config.enable_thinking = False

    qwen.eval()
    for p in qwen.parameters():
        p.requires_grad = False

    connector = VisualConnector(
        in_features=config.connector.in_features,
        hidden_features=config.connector.hidden_features,
        out_features=config.connector.out_features,
    ).to(device)

    sample_vis = torch.randn(16, config.vision.num_patches, config.vision.hidden_dim, device=device)
    scale_factor = calibrate_connector_scale(
        connector=connector,
        embed_tokens=qwen.model.embed_tokens,
        sample_visual_features=sample_vis,
        device=device,
    )

    vlm = VLMModel(language_model=qwen, connector=connector)

    t0_pipeline = time.time()
    total_p = sum(p.numel() for p in vlm.parameters())
    train_p = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    print(f"\n[VLM] Total Model Parameters: {total_p:,} | Initial Trainable (Connector only): {train_p:,}")

    # 4. Training Curriculum
    ckpt_cb = get_modal_checkpoint_callback()

    training_logs = None
    if run_all or any(s in args.stages for s in ["stage1", "stage2", "stage3"]):
        selected = [s for s in args.stages if s in ["stage1", "stage2", "stage3"]]
        stage_target = "all" if run_all or not selected else selected
        training_logs = train_task2(
            model=vlm,
            data_splits=data_splits,
            tokenizer=tokenizer,
            config=config,
            device=device,
            stage_to_run=stage_target,
            checkpoint_callback=ckpt_cb,
        )
        trigger_modal_stage_commit("training")

    # 5. Final Grounding Evaluation (4 Conditions)
    eval_results = {}
    if (run_all or "evaluate" in args.stages) and not args.skip_eval:
        print("\n" + "=" * 80)
        print("STEP 6: FINAL VISUAL GROUNDING EVALUATION (4 CONDITIONS)")
        print("=" * 80)

        print("[Evaluation] Evaluating Grounding Test Suite across 4 conditions...")
        grounding_results = evaluate_test_suite_4_conditions(
            model=vlm,
            suite_name="grounding",
            pairs=data_splits["grounding_test_pairs"],
            tokenizer=tokenizer,
            device=device,
            feature_cache_dir=config.data.feature_cache_dir,
            num_patches=config.vision.num_patches,
            hidden_dim=config.vision.hidden_dim,
            batch_size=config.evaluation.batch_size,
            num_bootstrap=config.evaluation.bootstrap_resamples,
            seed=config.training.seed,
        )
        eval_results["grounding"] = grounding_results

        print("[Evaluation] Evaluating Natural VQAv2 Test Suite across 4 conditions...")
        natural_vqa_results = evaluate_test_suite_4_conditions(
            model=vlm,
            suite_name="natural_vqa",
            pairs=data_splits["natural_vqa_test_pairs"],
            tokenizer=tokenizer,
            device=device,
            feature_cache_dir=config.data.feature_cache_dir,
            num_patches=config.vision.num_patches,
            hidden_dim=config.vision.hidden_dim,
            batch_size=config.evaluation.batch_size,
            num_bootstrap=config.evaluation.bootstrap_resamples,
            seed=config.training.seed,
        )
        eval_results["natural_vqa"] = natural_vqa_results

        # Save per-suite evaluation outputs
        with open(os.path.join(config.paths.output_dir, "grounding_eval_results.json"), "w", encoding="utf-8") as f:
            json.dump(grounding_results, f, indent=2)
        with open(os.path.join(config.paths.output_dir, "natural_vqa_eval_results.json"), "w", encoding="utf-8") as f:
            json.dump(natural_vqa_results, f, indent=2)

        ckpt_csv = os.path.join(config.paths.checkpoint_dir, "training_progress.csv")
        out_csv = os.path.join(config.paths.output_dir, "training_progress.csv")
        if os.path.exists(ckpt_csv) and not os.path.exists(out_csv):
            import shutil
            shutil.copy2(ckpt_csv, out_csv)

        trigger_modal_stage_commit("evaluate")

    # 6. Synthesis & Consolidated Reporting
    if run_all or "synthesize" in args.stages:
        print("\n" + "=" * 80)
        print("STEP 7: SYNTHESIS & REPORT GENERATION")
        print("=" * 80)

        metrics_summary = {
            "eval_results": {k: {cond: v["conditions"][cond] for cond in v["conditions"]} for k, v in eval_results.items()},
            "advantages": {k: v.get("advantages", {}) for k, v in eval_results.items()},
            "connector_calibration": {
                "rule": "Mean_L2_Norm(text_embeddings) / Mean_L2_Norm(unscaled_connector_output)",
                "scale_factor_gamma": scale_factor,
            },
            "parameters": {
                "total": total_p,
                "connector_trainable_s1": train_p,
                "connector_and_lora_trainable_s2_s3": train_p + 10092544,
            },
            "total_pipeline_time_sec": time.time() - t0_pipeline,
        }

        print("\n--- FINAL GROUNDING EVALUATION RESULTS (PERCENT) ---")
        for suite, res in eval_results.items():
            print(f"\nSuite: {suite.upper()}")
            for cond, vals in res["conditions"].items():
                acc = vals["accuracy"]
                pair_acc = vals["pair_accuracy"]
                nll = vals["mean_nll"]
                print(f"  [{cond:20s}] Accuracy: {acc:5.2f}% | Pair Acc: {pair_acc:5.2f}% | NLL: {nll:.4f}")
            adv = res.get("advantages", {})
            d_acc = adv.get("delta_accuracy_pp", 0.0)
            d_nll = adv.get("delta_nll", 0.0)
            print(f"  >> Correct-Image Advantage: delta_Acc = +{d_acc:.2f} pp | delta_NLL = +{d_nll:.4f}")

        sum_path = os.path.join(config.paths.output_dir, "metrics_summary.json")
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\n[Synthesis] Saved consolidated metrics summary to {sum_path}")

        anno_path = os.path.join(config.paths.output_dir, "qualitative_annotations.md")
        with open(anno_path, "w", encoding="utf-8") as f:
            f.write("# Task 2: Qualitative Complementary Pair Grounding Annotations\n\n")
            f.write("| Suite | Pair ID | Question | Correct Image Pred | Partner Swap Pred | Mismatched Pred | No Image Pred |\n")
            f.write("|---|---|---|---|---|---|---|\n")

            for suite, res in eval_results.items():
                preds_corr = {p["pair_id"]: p for p in res["predictions"].get("correct", [])}
                preds_swap = {p["pair_id"]: p for p in res["predictions"].get("complementary_swap", [])}
                preds_mism = {p["pair_id"]: p for p in res["predictions"].get("mismatched", [])}
                preds_noim = {p["pair_id"]: p for p in res["predictions"].get("no_image", [])}

                sample_pair_ids = list(preds_corr.keys())[:10]
                for pid in sample_pair_ids:
                    q = preds_corr[pid]["question"]
                    c_p = f"{preds_corr[pid]['pred1']} / {preds_corr[pid]['pred2']}"
                    s_p = f"{preds_swap.get(pid, {}).get('pred1', '')} / {preds_swap.get(pid, {}).get('pred2', '')}"
                    m_p = f"{preds_mism.get(pid, {}).get('pred1', '')} / {preds_mism.get(pid, {}).get('pred2', '')}"
                    n_p = f"{preds_noim.get(pid, {}).get('pred1', '')} / {preds_noim.get(pid, {}).get('pred2', '')}"
                    f.write(f"| {suite} | {pid} | {q} | {c_p} | {s_p} | {m_p} | {n_p} |\n")

        print(f"[Synthesis] Saved qualitative annotations table to {anno_path}")
        trigger_modal_stage_commit("synthesize")

    print("\n" + "=" * 80)
    print("TASK 2 EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
