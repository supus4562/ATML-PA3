"""
Diagnostics module for Task 1:
1. Weight-space diagnostics:
   - Subspace-interference ratio: rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2
   - Relative update magnitude: eta = ||dW||_F / ||W_0||_F
   Examined for down_proj in layers 0, 13, 27.
2. Activation diagnostics:
   - Fixed probe set: 32 GSM8K test + 32 HellaSwag validation sampled with seed 6304.
   - Forward hooks cache baseline down_proj input z at the last non-padding prompt token.
   - Ratio: R(z) = ||dW z||_2 / (||W_0 z||_2 + 1e-8) in FP32 on math vs non-math prompts.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from task1.models import compute_layer_svd, LoRALinear, OPLoRALinear


def compute_weight_space_metrics(
    W_0: torch.Tensor,
    delta_W: torch.Tensor,
    U_k: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute:
    rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2
    eta = ||dW||_F / ||W_0||_F
    in FP32.
    """
    W0_f = W_0.float()
    dW_f = delta_W.float()
    Uk_f = U_k.float()

    norm_W0 = torch.norm(W0_f, p="fro").item()
    norm_dW = torch.norm(dW_f, p="fro").item()

    if norm_dW < 1e-12:
        return {"rho_k": 0.0, "eta": 0.0, "norm_dW": 0.0, "norm_W0": norm_W0}

    # U_k: [out_features, k]
    # U_k U_k^T dW: [out_features, in_features]
    # Efficient: U_k (U_k^T dW)
    proj_dW = Uk_f @ (Uk_f.T @ dW_f)
    norm_proj = torch.norm(proj_dW, p="fro").item()

    rho_k = (norm_proj ** 2) / (norm_dW ** 2)
    eta = norm_dW / (norm_W0 + 1e-12)

    return {
        "rho_k": float(rho_k),
        "eta": float(eta),
        "norm_dW": float(norm_dW),
        "norm_W0": float(norm_W0),
    }


def sample_activation_probe_set(
    seed: int = 6304,
    num_gsm8k: int = 32,
    num_hellaswag: int = 32,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Sample 32 GSM8K test questions and 32 HellaSwag validation contexts
    using seed 6304 before inspecting model results.
    """
    rng = random.Random(seed)
    probe_set: Dict[str, List[Dict[str, Any]]] = {"math": [], "non_math": []}

    try:
        from datasets import load_dataset
        # 1. GSM8K test
        try:
            gsm8k = load_dataset("openai/gsm8k", "main", split="test")
        except Exception:
            gsm8k = load_dataset("gsm8k", "main", split="test")
        gsm8k_indices = list(range(len(gsm8k)))
        rng.shuffle(gsm8k_indices)
        selected_gsm8k = gsm8k_indices[:num_gsm8k]
        for idx in selected_gsm8k:
            item = gsm8k[idx]
            probe_set["math"].append({
                "id": idx,
                "dataset": "gsm8k",
                "question": item["question"],
            })

        # 2. HellaSwag val
        hellaswag = load_dataset("hellaswag", split="validation")
        hs_indices = list(range(len(hellaswag)))
        rng.shuffle(hs_indices)
        selected_hs = hs_indices[:num_hellaswag]
        for idx in selected_hs:
            item = hellaswag[idx]
            # Context for completion
            probe_set["non_math"].append({
                "id": idx,
                "dataset": "hellaswag",
                "context": item["ctx"],
            })
    except Exception as e:
        print(f"[Warning] Error loading datasets for probe set: {e}")
        # Synthetic fallback for offline validation
        for i in range(num_gsm8k):
            probe_set["math"].append({"id": i, "dataset": "gsm8k", "question": f"Solve problem {i}: 2x + {i} = 10."})
        for i in range(num_hellaswag):
            probe_set["non_math"].append({"id": i, "dataset": "hellaswag", "context": f"A person walks into room {i} and looks around."})

    return probe_set


class ActivationProbeCacher:
    """
    Forward hook manager to cache input activations z at down_proj
    at the last non-padding prompt token for layers 0, 13, 27.
    """
    def __init__(self, target_layers: List[int] = [0, 13, 27]):
        self.target_layers = target_layers
        self.cached_z: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}
        self.hooks = []

    def register(self, model: nn.Module):
        self.clear()
        # Find down_proj in transformer layers
        for layer_idx in self.target_layers:
            # Qwen layers naming: model.layers[layer_idx].mlp.down_proj
            layer_module = None
            try:
                layer_module = model.model.layers[layer_idx].mlp.down_proj
            except AttributeError:
                # Search dynamically
                for name, mod in model.named_modules():
                    if f"layers.{layer_idx}." in name and "down_proj" in name:
                        layer_module = mod
                        break

            if layer_module is not None:
                hook = self._make_hook(layer_idx)
                h = layer_module.register_forward_hook(hook)
                self.hooks.append(h)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            # inputs[0] shape: [batch_size, seq_len, in_features]
            # We want the activation at the last non-padding token
            # Store detached on CPU in FP32
            inp = inputs[0].detach()
            # If batch_size == 1, take the last token: inp[0, -1, :]
            for b in range(inp.shape[0]):
                z_token = inp[b, -1, :].float().cpu()
                self.cached_z[layer_idx].append(z_token)
        return hook_fn

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.cached_z = {l: [] for l in self.target_layers}


def compute_activation_ratio(
    W_0: torch.Tensor,
    delta_W: torch.Tensor,
    z_list: List[torch.Tensor],
) -> float:
    """
    Compute R(z) = ||delta_W @ z||_2 / (||W_0 @ z||_2 + 1e-8) in FP32,
    averaged across cached activation vectors z.
    """
    W0_f = W_0.float().cpu()
    dW_f = delta_W.float().cpu()

    ratios = []
    for z in z_list:
        # z: [in_features]
        z_f = z.float().cpu()
        num = torch.norm(dW_f @ z_f, p=2).item()
        denom = torch.norm(W0_f @ z_f, p=2).item() + 1e-8
        ratios.append(num / denom)

    return float(np.mean(ratios)) if ratios else 0.0
