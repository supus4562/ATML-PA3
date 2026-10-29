"""
Signal analysis, within-question quartile contrasts, length correlations,
and tokenwise fusion ablations for Task 3 (Step 3).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.stats import spearmanr, pearsonr
from task3.data import RolloutItem
from task3.metrics import compute_visconf_scores, compute_visual_engagement


def compute_within_question_quartiles(
    values: List[float],
    correctness: List[bool],
) -> Tuple[float, float, float]:
    """
    Computes correctness in top quartile vs bottom quartile of a score within a question.
    Returns: (top_quartile_accuracy, bottom_quartile_accuracy, difference)
    """
    arr_v = np.array(values, dtype=np.float64)
    arr_c = np.array(correctness, dtype=np.float64)

    q25 = np.percentile(arr_v, 25)
    q75 = np.percentile(arr_v, 75)

    top_mask = arr_v >= q75
    bottom_mask = arr_v <= q25

    top_acc = float(np.mean(arr_c[top_mask])) if np.any(top_mask) else 0.0
    bot_acc = float(np.mean(arr_c[bottom_mask])) if np.any(bottom_mask) else 0.0
    diff = top_acc - bot_acc

    return top_acc, bot_acc, diff


def analyze_signals_at_n32(
    question_rollouts: Dict[str, List[RolloutItem]],
    dataset_name: str = "MATH",
) -> Dict[str, Any]:
    """
    Performs Step 3 Signal Analysis across all questions in the evaluation split at N = 32:
    1. Within-question correlations (rollout length, scores, correctness).
    2. Top vs bottom quartile correctness contrast.
    3. MathVista confidence-vision correlations within rollout r(c_{i,t}, v_{i,t}).
    4. Tokenwise vs trajectory-level fusion comparison Z_token vs Z.
    """
    signal_keys = ["mean_logp", "mean_self_certainty"]
    if dataset_name == "MathVista":
        signal_keys.extend(["visual_engagement", "visconf", "visconf_tokenwise"])

    quartile_diffs: Dict[str, List[float]] = {k: [] for k in signal_keys}
    quartile_top: Dict[str, List[float]] = {k: [] for k in signal_keys}
    quartile_bot: Dict[str, List[float]] = {k: [] for k in signal_keys}

    # Length associations
    length_corr_list: List[float] = []
    length_quartile_diffs: List[float] = []

    # Tokenwise fusion covariance
    covariances: List[float] = []
    conf_vision_correlations: List[float] = []

    for s_id, rollouts in question_rollouts.items():
        if len(rollouts) < 32:
            continue
        cands = rollouts[:32]
        metrics_dict = compute_visconf_scores(cands)

        lengths = [float(c.length) for c in cands]
        corrects = [c.is_correct for c in cands]

        # Length correlation with correctness
        if len(set(corrects)) > 1:
            r_len, _ = spearmanr(lengths, [1.0 if c else 0.0 for c in corrects])
            if not np.isnan(r_len):
                length_corr_list.append(float(r_len))

        # Length quartile contrast
        t_acc, b_acc, d_acc = compute_within_question_quartiles(lengths, corrects)
        length_quartile_diffs.append(d_acc)

        # Score quartile contrasts
        for k in signal_keys:
            vals = [metrics_dict[c.rollout_id][k] for c in cands]
            top_a, bot_a, diff_a = compute_within_question_quartiles(vals, corrects)
            quartile_top[k].append(top_a)
            quartile_bot[k].append(bot_a)
            quartile_diffs[k].append(diff_a)

        # MathVista specific tokenwise correlations
        if dataset_name == "MathVista":
            for c in cands:
                covariances.append(metrics_dict[c.rollout_id]["token_covariance"])
                _, v_tokens = compute_visual_engagement(c)
                k_tokens = c.kl_u_p
                if len(v_tokens) == len(k_tokens) and len(v_tokens) > 2:
                    if np.std(v_tokens) > 1e-6 and np.std(k_tokens) > 1e-6:
                        r_cv, _ = pearsonr(v_tokens, k_tokens)
                        if not np.isnan(r_cv):
                            conf_vision_correlations.append(float(r_cv))

    summary = {
        "dataset": dataset_name,
        "num_questions": len(question_rollouts),
        "length_analysis": {
            "mean_spearman_with_correctness": float(np.mean(length_corr_list)) if length_corr_list else 0.0,
            "mean_quartile_diff": float(np.mean(length_quartile_diffs)) if length_quartile_diffs else 0.0,
        },
        "score_quartile_contrasts": {
            k: {
                "top_quartile_acc": float(np.mean(quartile_top[k])) * 100.0 if quartile_top[k] else 0.0,
                "bottom_quartile_acc": float(np.mean(quartile_bot[k])) * 100.0 if quartile_bot[k] else 0.0,
                "advantage_pp": float(np.mean(quartile_diffs[k])) * 100.0 if quartile_diffs[k] else 0.0,
            }
            for k in signal_keys
        },
    }

    if dataset_name == "MathVista":
        summary["visconf_fusion_analysis"] = {
            "mean_covariance_token_vs_rollout": float(np.mean(covariances)) if covariances else 0.0,
            "mean_confidence_vision_pearson_r": float(np.mean(conf_vision_correlations)) if conf_vision_correlations else 0.0,
        }

    return summary
