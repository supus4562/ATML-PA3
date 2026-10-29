"""
Candidate selection algorithms, Rank-Weighted Consensus (RWC),
permutation management, failure decomposition, and bootstrap confidence intervals.
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
from scipy.stats import rankdata
from task3.data import RolloutItem, check_answer_correctness
from task3.metrics import compute_visconf_scores, compute_mean_logp, compute_mean_self_certainty, compute_visual_engagement


def generate_candidate_permutations(
    num_rollouts: int = 32,
    num_permutations: int = 10,
    seed: int = 6304,
) -> List[List[int]]:
    """
    Generates 10 seeded random permutations of the 32 rollout IDs per question.
    Reused across all methods, budgets, and stopping policies.
    """
    rng = np.random.default_rng(seed)
    perms = []
    for _ in range(num_permutations):
        perms.append(list(rng.permutation(num_rollouts)))
    return perms


def compute_rwc_weights(scores: List[float]) -> List[float]:
    """
    Computes Rank-Weighted Consensus weights w_i = (N - r_i) / (N - 1)
    where r_i is descending average rank (1 is best).
    Uses average ranks for ties.
    """
    n = len(scores)
    if n <= 1:
        return [1.0] * n

    # rankdata assigns 1 to lowest value. To rank descending (1 is highest), negate scores.
    ranks = rankdata([-s for s in scores], method="average")
    weights = [(n - r) / (n - 1.0) for r in ranks]
    return weights


def select_answer(
    candidates: List[RolloutItem],
    rule_name: str,
    dataset_name: str = "MATH",
    precomputed_metrics: Optional[Dict[int, Dict[str, float]]] = None,
) -> Tuple[str, int, float]:
    """
    Executes a selection rule on a candidate prefix list.
    Returns: (selected_normalized_answer, selected_rollout_id, selection_score)
    Ties are broken by the earliest supporting candidate in candidate order.
    """
    n = len(candidates)
    if n == 0:
        return "NO_ANSWER", -1, 0.0
    if n == 1 or rule_name == "first_ans":
        return candidates[0].normalized_answer, candidates[0].rollout_id, 1.0

    # Retrieve quality metrics for the candidates
    if precomputed_metrics is None:
        metrics_dict = compute_visconf_scores(candidates)
    else:
        metrics_dict = precomputed_metrics

    # 1. Self-Consistency (Plurality Voting)
    if rule_name == "self_consistency":
        votes: Dict[str, float] = {}
        first_idx: Dict[str, int] = {}
        rollout_map: Dict[str, int] = {}

        for idx, c in enumerate(candidates):
            ans = c.normalized_answer
            votes[ans] = votes.get(ans, 0.0) + 1.0
            if ans not in first_idx:
                first_idx[ans] = idx
                rollout_map[ans] = c.rollout_id

        # Sort by vote count descending, then by earliest appearance ascending
        sorted_answers = sorted(votes.keys(), key=lambda a: (-votes[a], first_idx[a]))
        best_ans = sorted_answers[0]
        return best_ans, rollout_map[best_ans], votes[best_ans]

    # 2. Score-based Highest Selection
    score_keys = {
        "highest_logp": "mean_logp",
        "highest_self_certainty": "mean_self_certainty",
        "highest_vision": "visual_engagement",
        "highest_visconf": "visconf",
        "highest_visconf_tokenwise": "visconf_tokenwise",
    }
    if rule_name in score_keys:
        k = score_keys[rule_name]
        best_cand = candidates[0]
        best_score = metrics_dict[best_cand.rollout_id][k]

        for c in candidates[1:]:
            s = metrics_dict[c.rollout_id][k]
            # Strict inequality preserves earliest supporting candidate on ties
            if s > best_score:
                best_score = s
                best_cand = c
        return best_cand.normalized_answer, best_cand.rollout_id, best_score

    # 3. Rank-Weighted Consensus (RWC)
    rwc_score_keys = {
        "rwc_self_certainty": "mean_self_certainty",
        "rwc_vision": "visual_engagement",
        "rwc_visconf": "visconf",
        "rwc_visconf_tokenwise": "visconf_tokenwise",
    }
    if rule_name in rwc_score_keys:
        k = rwc_score_keys[rule_name]
        scores = [metrics_dict[c.rollout_id][k] for c in candidates]
        weights = compute_rwc_weights(scores)

        ans_weights: Dict[str, float] = {}
        first_idx: Dict[str, int] = {}
        rollout_map: Dict[str, int] = {}

        for idx, (c, w) in enumerate(zip(candidates, weights)):
            ans = c.normalized_answer
            ans_weights[ans] = ans_weights.get(ans, 0.0) + w
            if ans not in first_idx:
                first_idx[ans] = idx
                rollout_map[ans] = c.rollout_id

        sorted_answers = sorted(ans_weights.keys(), key=lambda a: (-ans_weights[a], first_idx[a]))
        best_ans = sorted_answers[0]
        return best_ans, rollout_map[best_ans], ans_weights[best_ans]

    # Length-only control: shortest rollout
    if rule_name == "shortest_rollout":
        best_cand = min(candidates, key=lambda c: c.length)
        return best_cand.normalized_answer, best_cand.rollout_id, float(best_cand.length)

    raise ValueError(f"Unknown selection rule: {rule_name}")


def evaluate_question_budget(
    rollouts: List[RolloutItem],
    budget: int,
    permutations: List[List[int]],
    rule_names: List[str],
    dataset_name: str = "MATH",
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates selection rules for a single question at candidate budget N.
    Averages over the 10 seeded random orders.
    Returns per-rule dictionary containing:
    - accuracy
    - token_cost
    - oracle_coverage
    - distinct_groups
    - sampling_failure_rate
    - selection_failure_rate
    """
    num_perms = len(permutations)
    rule_results: Dict[str, Dict[str, Any]] = {
        r: {
            "correct_count": 0.0,
            "token_cost": 0.0,
            "oracle_hit": 0.0,
            "num_distinct_groups": 0.0,
            "sampling_failures": 0.0,
            "selection_failures": 0.0,
        }
        for r in rule_names
    }

    gold_ans = rollouts[0].gold_answer
    q_meta = rollouts[0].question_metadata
    ans_corr_cache: Dict[str, bool] = {c.normalized_answer: c.is_correct for c in rollouts}

    for perm in permutations:
        prefix_indices = perm[:budget]
        candidates = [rollouts[i] for i in prefix_indices]

        # Oracle coverage for this prefix
        oracle_correct = any(c.is_correct for c in candidates)
        distinct_answers = len(set(c.normalized_answer for c in candidates))

        # VisConf candidate-calibrated scores for this specific prefix
        metrics_dict = compute_visconf_scores(candidates)

        total_tokens_consumed = sum(c.length for c in candidates)

        for rule in rule_names:
            sel_ans, sel_id, _ = select_answer(
                candidates=candidates,
                rule_name=rule,
                dataset_name=dataset_name,
                precomputed_metrics=metrics_dict,
            )

            if sel_ans in ans_corr_cache:
                is_corr = ans_corr_cache[sel_ans]
            else:
                is_corr = check_answer_correctness(sel_ans, gold_ans, dataset_name, q_meta)
                ans_corr_cache[sel_ans] = is_corr

            cost = candidates[0].length if rule == "first_ans" else total_tokens_consumed

            res = rule_results[rule]
            res["correct_count"] += 1.0 if is_corr else 0.0
            res["token_cost"] += cost
            res["oracle_hit"] += 1.0 if oracle_correct else 0.0
            res["num_distinct_groups"] += distinct_answers

            if not is_corr:
                if not oracle_correct:
                    res["sampling_failures"] += 1.0
                else:
                    res["selection_failures"] += 1.0

    # Normalize by number of permutations
    normalized: Dict[str, Dict[str, Any]] = {}
    for r, vals in rule_results.items():
        normalized[r] = {
            "accuracy": vals["correct_count"] / num_perms,
            "token_cost": vals["token_cost"] / num_perms,
            "oracle_coverage": vals["oracle_hit"] / num_perms,
            "distinct_groups": vals["num_distinct_groups"] / num_perms,
            "sampling_failure": vals["sampling_failures"] / num_perms,
            "selection_failure": vals["selection_failures"] / num_perms,
        }

    return normalized


def compute_paired_bootstrap_ci(
    diffs: List[float],
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 6304,
) -> Tuple[float, float, float]:
    """
    Computes 95% confidence intervals using question-level paired bootstrap resamples.
    Returns: (mean_diff, lower_bound, upper_bound)
    """
    if not diffs:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    n = len(diffs)
    arr = np.array(diffs, dtype=np.float64)

    boot_means = np.empty(num_resamples, dtype=np.float64)
    for b in range(num_resamples):
        indices = rng.integers(0, n, size=n)
        boot_means[b] = np.mean(arr[indices])

    alpha = 1.0 - confidence_level
    try:
        lower = float(np.percentile(boot_means, 100 * (alpha / 2.0), method="linear"))
        upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0), method="linear"))
    except TypeError:
        lower = float(np.percentile(boot_means, 100 * (alpha / 2.0), interpolation="linear"))
        upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0), interpolation="linear"))
    mean_val = float(np.mean(arr))

    return mean_val, lower, upper
