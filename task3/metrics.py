"""
Quality scoring, self-certainty, visual engagement, and candidate-calibrated
VisConf metrics for Task 3.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from task3.data import RolloutItem


def compute_mean_logp(rollout: RolloutItem) -> float:
    """
    Mean log-probability:
    L_i = (1 / T_i) * sum_{t=1}^{T_i} log p_{i,t}(y_{i,t})
    """
    if not rollout.logp:
        return -999.0
    return float(np.mean(rollout.logp))


def compute_mean_self_certainty(rollout: RolloutItem) -> float:
    """
    Mean self-certainty (KL divergence from uniform):
    K_i = (1 / T_i) * sum_{t=1}^{T_i} D_KL(U || p_{i,t})
    """
    if not rollout.kl_u_p:
        return 0.0
    return float(np.mean(rollout.kl_u_p))


def compute_visual_engagement(rollout: RolloutItem) -> Tuple[float, List[float]]:
    """
    Visual engagement for VLM rollouts (MathVista):
    v_{i,t} = A_{i,t}^{image} / (A_{i,t}^{image} + A_{i,t}^{prompt} + A_{i,t}^{generated})
    V_i = (1 / T_i) * sum_{t=1}^{T_i} v_{i,t}
    Returns: (V_i, tokenwise_v)
    """
    if not rollout.a_image or not rollout.a_prompt or not rollout.a_generated:
        return 0.0, []

    a_img = np.array(rollout.a_image, dtype=np.float64)
    a_prm = np.array(rollout.a_prompt, dtype=np.float64)
    a_gen = np.array(rollout.a_generated, dtype=np.float64)

    total_attn = a_img + a_prm + a_gen
    denom = np.maximum(total_attn, 1e-8)
    tokenwise_v = a_img / denom
    mean_v = float(np.mean(tokenwise_v)) if len(tokenwise_v) > 0 else 0.0

    return mean_v, tokenwise_v.tolist()


def compute_visconf_scores(
    candidates: List[RolloutItem],
) -> Dict[int, Dict[str, float]]:
    """
    Computes candidate-calibrated VisConf scores for a set of revealed rollouts G.
    Per PA Section 2:
    1. Pool all token KL values across the current candidate set G.
    2. Calibrate using 5th and 95th percentiles (linear interpolation):
       c_{i,t} = clip_{[0, 1]}( (k_{i,t} - Q_{.05}(G)) / max(Q_{.95}(G) - Q_{.05}(G), 10^{-8}) )
    3. C_i = (1 / T_i) * sum c_{i,t}
    4. Z_i = V_i * C_i
    5. Z_token = (1 / T_i) * sum (c_{i,t} * v_{i,t})
    Returns dict mapping rollout_id to quality dictionary.
    """
    # 1. Pool token KL values
    all_kl = []
    for c in candidates:
        all_kl.extend(c.kl_u_p)

    if not all_kl:
        q05, q95 = 0.0, 1.0
    else:
        try:
            q05 = float(np.percentile(all_kl, 5, method="linear"))
            q95 = float(np.percentile(all_kl, 95, method="linear"))
        except TypeError:
            q05 = float(np.percentile(all_kl, 5, interpolation="linear"))
            q95 = float(np.percentile(all_kl, 95, interpolation="linear"))

    denom = max(q95 - q05, 1e-8)

    results: Dict[int, Dict[str, float]] = {}
    for c in candidates:
        k_arr = np.array(c.kl_u_p, dtype=np.float64)
        if len(k_arr) == 0:
            c_tokens = np.array([0.0])
        else:
            c_tokens = np.clip((k_arr - q05) / denom, 0.0, 1.0)

        mean_c = float(np.mean(c_tokens))
        mean_v, tokenwise_v = compute_visual_engagement(c)

        z_val = mean_v * mean_c

        # Tokenwise fusion ablation
        if len(tokenwise_v) == len(c_tokens) and len(c_tokens) > 0:
            v_arr = np.array(tokenwise_v, dtype=np.float64)
            z_token = float(np.mean(c_tokens * v_arr))
            covariance = z_token - z_val
        else:
            z_token = z_val
            covariance = 0.0

        results[c.rollout_id] = {
            "mean_logp": compute_mean_logp(c),
            "mean_self_certainty": compute_mean_self_certainty(c),
            "visual_engagement": mean_v,
            "calibrated_confidence": mean_c,
            "visconf": z_val,
            "visconf_tokenwise": z_token,
            "token_covariance": covariance,
        }

    return results
