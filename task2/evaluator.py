"""
Grounding and Natural VQA evaluation protocol for Task 2.
Implements the 4 image conditions:
1. Correct image (I_correct)
2. Complementary swap (I_partner)
3. Mismatched (unrelated image from another pair)
4. No image (visual tokens omitted)
Computes:
- Normalized Exact Match (Grounding)
- Local VQA Soft Consensus Accuracy (10-human answer consensus)
- Pair accuracy, identical-answer rate, empty-answer rate, reference NLL
- Correct-image advantage (delta_Acc and delta_NLL)
- 4,000-resample complete-pair bootstrap 95% confidence intervals.
"""

import copy
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from transformers import AutoTokenizer

from task2.models import VLMModel
from task2.loss import compute_reference_answer_nll
from task2.data import (
    VLMExample,
    DynamicVLMCollator,
    normalize_vqa_answer,
)


def compute_vqa_soft_consensus_score(prediction: str, human_references: List[str]) -> float:
    """
    Standard VQA soft consensus score against 10 human answers:
    s(a_hat) = (1 / 10) * sum_{i=1}^{10} min(1, (1 / 3) * sum_{j != i} 1[a_hat == a_j])
    """
    norm_pred = normalize_vqa_answer(prediction)
    norm_refs = [normalize_vqa_answer(r) for r in human_references]

    if not norm_pred:
        return 0.0

    scores = []
    for i in range(len(norm_refs)):
        match_count = sum(1 for j, r in enumerate(norm_refs) if j != i and r == norm_pred)
        scores.append(min(1.0, match_count / 3.0))

    return float(np.mean(scores)) if scores else 0.0


def compute_pair_bootstrap_ci(
    pair_metrics: List[float],
    num_resamples: int = 4000,
    confidence_level: float = 0.95,
    seed: int = 6304,
) -> Tuple[float, float, float]:
    """
    Computes 95% confidence intervals using bootstrap resamples of complete pairs.
    Returns: (mean, lower_bound, upper_bound)
    """
    if not pair_metrics:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    n = len(pair_metrics)
    data = np.array(pair_metrics, dtype=np.float64)

    boot_means = np.empty(num_resamples, dtype=np.float64)
    for b in range(num_resamples):
        indices = rng.randint(0, n, size=n)
        boot_means[b] = np.mean(data[indices])

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    mean_val = float(np.mean(data))

    return mean_val, lower, upper


def evaluate_test_suite_4_conditions(
    model: VLMModel,
    suite_name: str,
    pairs: List[Tuple[VLMExample, VLMExample]],
    tokenizer: AutoTokenizer,
    device: torch.device,
    feature_cache_dir: Optional[str] = None,
    num_patches: int = 196,
    hidden_dim: int = 768,
    batch_size: int = 8,
    num_bootstrap: int = 4000,
    seed: int = 6304,
) -> Dict[str, Any]:
    """
    Evaluates a test suite (grounding or natural VQAv2) across all 4 image conditions:
    1. Correct
    2. Complementary Swap
    3. Mismatched
    4. No Image
    """
    model.eval()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Build mismatched mapping: fixed permutation excluding both images in original pair
    all_img_ids = [p[0].image_id for p in pairs] + [p[1].image_id for p in pairs]
    rng = random.Random(seed)
    mismatched_map: Dict[int, int] = {}
    for p in pairs:
        cand_a = [img for img in all_img_ids if img not in [p[0].image_id, p[1].image_id]]
        cand_b = [img for img in all_img_ids if img not in [p[0].image_id, p[1].image_id]]
        mismatched_map[p[0].image_id] = rng.choice(cand_a) if cand_a else p[1].image_id
        mismatched_map[p[1].image_id] = rng.choice(cand_b) if cand_b else p[0].image_id

    conditions = ["correct", "complementary_swap", "mismatched", "no_image"]
    results_by_condition: Dict[str, Any] = {}

    # Store per-pair scores for matched bootstrapping
    per_condition_pair_accs: Dict[str, List[float]] = {}
    per_condition_pair_nlls: Dict[str, List[float]] = {}
    per_condition_predictions: Dict[str, List[Dict[str, Any]]] = {}

    for cond in conditions:
        is_noimg = (cond == "no_image")
        collator = DynamicVLMCollator(
            tokenizer=tokenizer,
            feature_cache_dir=feature_cache_dir,
            num_patches=num_patches,
            hidden_dim=hidden_dim,
            no_image=is_noimg,
        )

        all_pair_acc = []
        all_pair_nll = []
        all_item_acc = []
        all_item_nll = []
        identical_pair_count = 0
        empty_answer_count = 0
        total_answers = 0
        cond_preds = []

        t_gen_start = time.perf_counter()

        for p in pairs:
            # Prepare pair examples for the condition
            ex1 = copy.copy(p[0])
            ex2 = copy.copy(p[1])

            if cond == "complementary_swap":
                ex1.image_id = p[0].partner_image_id
                ex2.image_id = p[1].partner_image_id
            elif cond == "mismatched":
                ex1.image_id = mismatched_map.get(p[0].image_id, p[1].image_id)
                ex2.image_id = mismatched_map.get(p[1].image_id, p[0].image_id)

            pair_batch = [ex1, ex2]
            batch = collator(pair_batch)

            prefix_ids = batch["prefix_ids"].to(device)
            vis_feats = batch["visual_features"].to(device) if batch["visual_features"] is not None else None
            suffix_ids = batch["suffix_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 1. Reference NLL
            with torch.no_grad():
                out = model.forward_multimodal(prefix_ids, vis_feats, suffix_ids, attention_mask)
                nll_val, _ = compute_reference_answer_nll(out.logits, labels, im_end_id=im_end_id)
                all_pair_nll.append(nll_val)
                all_item_nll.extend([nll_val, nll_val])

            # 2. Greedy Answer Generation
            question_only_ids = []
            for ex in pair_batch:
                q_toks = tokenizer.encode(f"\n{ex.question}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
                question_only_ids.append(q_toks)
            q_max = max(len(q) for q in question_only_ids)
            q_padded = [q + [tokenizer.pad_token_id or 0] * (q_max - len(q)) for q in question_only_ids]
            q_tensor = torch.tensor(q_padded, device=device, dtype=torch.long)

            gen_answers = model.generate_answer(
                prefix_ids=prefix_ids,
                visual_features=vis_feats,
                question_ids=q_tensor,
                tokenizer=tokenizer,
                max_new_tokens=16,
            )

            pred1, pred2 = gen_answers[0], gen_answers[1]
            total_answers += 2
            if not pred1.strip():
                empty_answer_count += 1
            if not pred2.strip():
                empty_answer_count += 1

            norm_p1 = normalize_vqa_answer(pred1)
            norm_p2 = normalize_vqa_answer(pred2)

            if norm_p1 == norm_p2:
                identical_pair_count += 1

            # Scoring: Grounding Exact Match vs Natural VQA Soft Consensus
            if suite_name == "grounding":
                score1 = 1.0 if norm_p1 == p[0].normalized_answer else 0.0
                score2 = 1.0 if norm_p2 == p[1].normalized_answer else 0.0
                pair_correct = 1.0 if (score1 == 1.0 and score2 == 1.0) else 0.0
            else:
                score1 = compute_vqa_soft_consensus_score(pred1, p[0].all_human_answers)
                score2 = compute_vqa_soft_consensus_score(pred2, p[1].all_human_answers)
                # Pair accuracy matches canonical consensus answers
                pair_correct = 1.0 if (norm_p1 == p[0].normalized_answer and norm_p2 == p[1].normalized_answer) else 0.0

            all_item_acc.extend([score1, score2])
            all_pair_acc.append(pair_correct)

            cond_preds.append({
                "pair_id": p[0].pair_id,
                "question": p[0].question,
                "pred1": pred1,
                "target1": p[0].answer,
                "score1": score1,
                "pred2": pred2,
                "target2": p[1].answer,
                "score2": score2,
                "pair_correct": pair_correct,
            })

        t_gen_elapsed = time.perf_counter() - t_gen_start
        gen_time_per_answer = t_gen_elapsed / max(1, total_answers)

        mean_acc, acc_lo, acc_hi = compute_pair_bootstrap_ci(all_item_acc, num_resamples=num_bootstrap, seed=seed)
        mean_pair_acc, pair_lo, pair_hi = compute_pair_bootstrap_ci(all_pair_acc, num_resamples=num_bootstrap, seed=seed)
        mean_nll, nll_lo, nll_hi = compute_pair_bootstrap_ci(all_pair_nll, num_resamples=num_bootstrap, seed=seed)

        per_condition_pair_accs[cond] = all_pair_acc
        per_condition_pair_nlls[cond] = all_pair_nll
        per_condition_predictions[cond] = cond_preds

        results_by_condition[cond] = {
            "accuracy": mean_acc * 100.0,
            "accuracy_ci": (acc_lo * 100.0, acc_hi * 100.0),
            "pair_accuracy": mean_pair_acc * 100.0,
            "pair_accuracy_ci": (pair_lo * 100.0, pair_hi * 100.0),
            "mean_nll": mean_nll,
            "mean_nll_ci": (nll_lo, nll_hi),
            "identical_answer_rate": (identical_pair_count / max(1, len(pairs))) * 100.0,
            "empty_answer_rate": (empty_answer_count / max(1, total_answers)) * 100.0,
            "gen_time_per_answer_sec": gen_time_per_answer,
        }

    # Compute Correct-Image Advantages over Complementary Swaps with matched bootstrapping
    corr_accs = np.array(per_condition_pair_accs["correct"])
    swap_accs = np.array(per_condition_pair_accs["complementary_swap"])
    delta_acc_pairs = list(corr_accs - swap_accs)
    mean_d_acc, d_acc_lo, d_acc_hi = compute_pair_bootstrap_ci(delta_acc_pairs, num_resamples=num_bootstrap, seed=seed + 10)

    corr_nlls = np.array(per_condition_pair_nlls["correct"])
    swap_nlls = np.array(per_condition_pair_nlls["complementary_swap"])
    delta_nll_pairs = list(swap_nlls - corr_nlls)  # Positive = correct image makes reference more likely
    mean_d_nll, d_nll_lo, d_nll_hi = compute_pair_bootstrap_ci(delta_nll_pairs, num_resamples=num_bootstrap, seed=seed + 11)

    advantages = {
        "delta_accuracy_pp": mean_d_acc * 100.0,
        "delta_accuracy_ci": (d_acc_lo * 100.0, d_acc_hi * 100.0),
        "delta_nll": mean_d_nll,
        "delta_nll_ci": (d_nll_lo, d_nll_hi),
    }

    return {
        "suite": suite_name,
        "num_pairs": len(pairs),
        "num_questions": len(pairs) * 2,
        "conditions": results_by_condition,
        "advantages": advantages,
        "predictions": per_condition_predictions,
    }
