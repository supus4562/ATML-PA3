"""
Adaptive sampling controllers (agreement-based and quality-aware),
development set threshold optimization, and held-out evaluation (Step 5).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from task3.data import RolloutItem, check_answer_correctness
from task3.metrics import compute_visconf_scores
from task3.selection import compute_rwc_weights, select_answer


def check_adaptive_stop(
    candidates_n: List[RolloutItem],
    controller_type: str,
    tau: float,
    gamma: float = 0.0,
    dataset_name: str = "MATH",
    precomputed_metrics: Optional[Dict[int, Dict[str, float]]] = None,
) -> Tuple[bool, str]:
    """
    Checks if adaptive stopping condition is met at checkpoint n.
    Returns: (should_stop, current_self_consistency_answer)
    """
    n = len(candidates_n)
    if n == 0:
        return False, "NO_ANSWER"

    # Count votes for self-consistency
    votes: Dict[str, int] = {}
    first_idx: Dict[str, int] = {}
    for idx, c in enumerate(candidates_n):
        ans = c.normalized_answer
        votes[ans] = votes.get(ans, 0) + 1
        if ans not in first_idx:
            first_idx[ans] = idx

    sorted_votes = sorted(votes.items(), key=lambda kv: (-kv[1], first_idx[kv[0]]))
    top_ans, top_count = sorted_votes[0]
    runner_up_count = sorted_votes[1][1] if len(sorted_votes) > 1 else 0

    # Agreement Condition:
    # 1. Non-NO_ANSWER
    # 2. Unique leading answer
    # 3. Fraction >= tau
    is_unique = (top_count > runner_up_count) or (len(sorted_votes) == 1)
    if top_ans == "NO_ANSWER" or not is_unique:
        return False, top_ans

    share = top_count / float(n)
    if share < tau:
        return False, top_ans

    if controller_type == "agreement_based":
        return True, top_ans

    # Quality-Aware Condition:
    # Requires agreement condition PLUS:
    # 1. SC winner equals RWC winner
    # 2. RWC vote margin >= gamma
    if precomputed_metrics is None:
        metrics_dict = compute_visconf_scores(candidates_n)
    else:
        metrics_dict = precomputed_metrics

    metric_key = "mean_self_certainty" if dataset_name == "MATH" else "visconf"
    scores = [metrics_dict[c.rollout_id][metric_key] for c in candidates_n]
    weights = compute_rwc_weights(scores)

    rwc_votes: Dict[str, float] = {}
    rwc_first_idx: Dict[str, int] = {}
    for idx, (c, w) in enumerate(zip(candidates_n, weights)):
        ans = c.normalized_answer
        rwc_votes[ans] = rwc_votes.get(ans, 0.0) + w
        if ans not in rwc_first_idx:
            rwc_first_idx[ans] = idx

    sorted_rwc = sorted(rwc_votes.items(), key=lambda kv: (-kv[1], rwc_first_idx[kv[0]]))
    rwc_top_ans, rwc_top_weight = sorted_rwc[0]
    rwc_runner_up_weight = sorted_rwc[1][1] if len(sorted_rwc) > 1 else 0.0

    # SC winner must match RWC winner
    if top_ans != rwc_top_ans:
        return False, top_ans

    # RWC vote margin >= gamma
    total_weight = sum(rwc_votes.values())
    if total_weight <= 1e-8:
        return False, top_ans

    margin = (rwc_top_weight - rwc_runner_up_weight) / total_weight
    if margin < gamma:
        return False, top_ans

    return True, top_ans


def simulate_adaptive_sampling_policy(
    question_rollouts: Dict[str, List[RolloutItem]],
    permutations: List[List[int]],
    controller_type: str,
    tau: float,
    gamma: float = 0.0,
    checkpoints: Optional[List[int]] = None,
    dataset_name: str = "MATH",
) -> Dict[str, Any]:
    """
    Simulates adaptive sampling across a set of questions (dev or eval) over the 10 permutations.
    Checkpoints: n in {4, 8, 16}; fallback: 32.
    Returns:
    - mean_accuracy
    - mean_token_cost
    - mean_rollouts
    - stopping_distribution (histogram over 4, 8, 16, 32)
    - incorrect_early_stops
    - recoverable_by_fixed32
    """
    if checkpoints is None:
        checkpoints = [4, 8, 16]

    num_perms = len(permutations)
    num_questions = len(question_rollouts)

    total_correct = 0.0
    total_tokens = 0.0
    total_rollouts = 0.0

    stop_hist = {4: 0, 8: 0, 16: 0, 32: 0}
    incorrect_early_stops = 0
    recoverable_by_fixed32 = 0

    for s_id, rollouts in question_rollouts.items():
        gold_ans = rollouts[0].gold_answer
        q_meta = rollouts[0].question_metadata

        for perm in permutations:
            stopped = False
            chosen_ans = "NO_ANSWER"
            stopped_n = 32

            for n in checkpoints:
                prefix_cands = [rollouts[i] for i in perm[:n]]
                should_stop, ans = check_adaptive_stop(
                    candidates_n=prefix_cands,
                    controller_type=controller_type,
                    tau=tau,
                    gamma=gamma,
                    dataset_name=dataset_name,
                )
                if should_stop:
                    stopped = True
                    chosen_ans = ans
                    stopped_n = n
                    break

            if not stopped:
                # Fallback to fixed-32 self-consistency
                prefix_32 = [rollouts[i] for i in perm[:32]]
                chosen_ans, _, _ = select_answer(prefix_32, "self_consistency", dataset_name)
                stopped_n = 32

            cost = sum(rollouts[i].length for i in perm[:stopped_n])
            is_corr = check_answer_correctness(chosen_ans, gold_ans, dataset_name, q_meta)

            total_correct += 1.0 if is_corr else 0.0
            total_tokens += cost
            total_rollouts += stopped_n
            stop_hist[stopped_n] += 1

            # Failure analysis on early stops
            if stopped_n < 32 and not is_corr:
                incorrect_early_stops += 1
                # Check if fixed-32 would have gotten it right
                fixed32_ans, _, _ = select_answer([rollouts[i] for i in perm[:32]], "self_consistency", dataset_name)
                if check_answer_correctness(fixed32_ans, gold_ans, dataset_name, q_meta):
                    recoverable_by_fixed32 += 1

    total_evals = num_questions * num_perms
    mean_acc = (total_correct / total_evals) * 100.0
    mean_cost = total_tokens / total_evals
    mean_rollouts = total_rollouts / total_evals

    hist_percent = {k: (v / total_evals) * 100.0 for k, v in stop_hist.items()}

    return {
        "controller": controller_type,
        "tau": tau,
        "gamma": gamma,
        "accuracy": mean_acc,
        "mean_token_cost": mean_cost,
        "mean_rollouts": mean_rollouts,
        "stopping_distribution_pct": hist_percent,
        "incorrect_early_stops": incorrect_early_stops,
        "recoverable_by_fixed32": recoverable_by_fixed32,
    }


def find_optimal_adaptive_thresholds(
    dev_rollouts: Dict[str, List[RolloutItem]],
    permutations: List[List[int]],
    dataset_name: str = "MATH",
    tau_grid: Optional[List[float]] = None,
    gamma_grid: Optional[List[float]] = None,
    max_accuracy_drop_pp: float = 1.0,
) -> Dict[str, Any]:
    """
    Grid search on the 200 Development questions per PA Section 3:
    - tau in {0.5, 0.625, 0.75, 0.875, 1.0}
    - gamma in {0.0, 0.1, 0.25, 0.5}
    - Select setting with lowest mean token cost whose dev accuracy is no more than 1.0 pp below fixed-32 SC.
    - Tie-breaking: Higher accuracy, then smaller tau and gamma.
    """
    if tau_grid is None:
        tau_grid = [0.50, 0.625, 0.75, 0.875, 1.00]
    if gamma_grid is None:
        gamma_grid = [0.00, 0.10, 0.25, 0.50]

    # Baseline: Fixed-32 Self-Consistency on Dev
    baseline_sim = simulate_adaptive_sampling_policy(
        question_rollouts=dev_rollouts,
        permutations=permutations,
        controller_type="agreement_based",
        tau=999.0,  # Never triggers, always consumes 32
        dataset_name=dataset_name,
    )
    baseline_acc = baseline_sim["accuracy"]
    target_min_acc = baseline_acc - max_accuracy_drop_pp
    print(f"[Adaptive Grid] {dataset_name} Dev Fixed-32 SC Accuracy: {baseline_acc:.2f}% | Target Min Acc: {target_min_acc:.2f}%")

    # 1. Agreement-Based Search
    agreement_candidates = []
    for tau in tau_grid:
        sim = simulate_adaptive_sampling_policy(
            dev_rollouts, permutations, "agreement_based", tau, 0.0, dataset_name=dataset_name
        )
        if sim["accuracy"] >= target_min_acc:
            agreement_candidates.append(sim)

    if agreement_candidates:
        # Sort: lowest cost, highest accuracy, smallest tau
        agreement_candidates.sort(key=lambda s: (s["mean_token_cost"], -s["accuracy"], s["tau"]))
        best_agreement = agreement_candidates[0]
    else:
        # Fallback to closest accuracy
        all_agr = [simulate_adaptive_sampling_policy(dev_rollouts, permutations, "agreement_based", tau, 0.0, dataset_name=dataset_name) for tau in tau_grid]
        best_agreement = max(all_agr, key=lambda s: s["accuracy"])

    # 2. Quality-Aware Search
    quality_candidates = []
    for tau in tau_grid:
        for gamma in gamma_grid:
            sim = simulate_adaptive_sampling_policy(
                dev_rollouts, permutations, "quality_aware", tau, gamma, dataset_name=dataset_name
            )
            if sim["accuracy"] >= target_min_acc:
                quality_candidates.append(sim)

    if quality_candidates:
        quality_candidates.sort(key=lambda s: (s["mean_token_cost"], -s["accuracy"], s["tau"], s["gamma"]))
        best_quality = quality_candidates[0]
    else:
        all_qua = [
            simulate_adaptive_sampling_policy(dev_rollouts, permutations, "quality_aware", tau, gamma, dataset_name=dataset_name)
            for tau in tau_grid for gamma in gamma_grid
        ]
        best_quality = max(all_qua, key=lambda s: s["accuracy"])

    return {
        "dataset": dataset_name,
        "dev_fixed32_sc_acc": baseline_acc,
        "best_agreement_thresholds": {
            "tau": best_agreement["tau"],
            "dev_acc": best_agreement["accuracy"],
            "dev_cost": best_agreement["mean_token_cost"],
            "dev_rollouts": best_agreement["mean_rollouts"],
        },
        "best_quality_thresholds": {
            "tau": best_quality["tau"],
            "gamma": best_quality["gamma"],
            "dev_acc": best_quality["accuracy"],
            "dev_cost": best_quality["mean_token_cost"],
            "dev_rollouts": best_quality["mean_rollouts"],
        },
    }
