"""
Model architectures, custom LoRA and OPLoRA layers, SVD basis computation,
parameter grouping, and weight merging with logit parity checking.
"""

import math
import time
from typing import Dict, List, Optional, Set, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Standard Low-Rank Adaptation (LoRA) layer.
    W = W_0 + (alpha / r) * B * A
    where W_0 is frozen, A ~ N(0, 1/r), B = 0, no trainable biases.
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout_prob = dropout

        # Frozen base weight
        self.weight = base_layer.weight
        self.weight.requires_grad = False
        if base_layer.bias is not None:
            self.bias = base_layer.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        # Trainable adapter matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32))

        # Reset parameters: A random gaussian, B zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def get_delta_weight(self) -> torch.Tensor:
        """Compute Delta W = (alpha / r) * B * A in FP32."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)

        x_dropped = self.lora_dropout(x.to(dtype=self.lora_A.dtype))
        adapter_out = F.linear(F.linear(x_dropped, self.lora_A), self.lora_B) * self.scaling

        return base_out + adapter_out.to(dtype=base_out.dtype)


class OPLoRALinear(nn.Module):
    """
    Orthogonal Projection LoRA (OPLoRA) layer.
    Protects the top-k leading singular subspaces of the starting weight W_0:
    W_0 = U Sigma V^T
    B_tilde = B - U_k (U_k^T B)
    A_tilde = A - (A V_k) V_k^T
    Delta W = (alpha / r) * B_tilde * A_tilde

    Uses effective factors rather than dense projection matrices.
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        protection_rank_k: int = 128,
        dropout: float = 0.05,
        U_k: Optional[torch.Tensor] = None,
        V_k: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.protection_rank_k = min(protection_rank_k, self.in_features, self.out_features)
        self.scaling = alpha / rank
        self.dropout_prob = dropout

        # Frozen base weight
        self.weight = base_layer.weight
        self.weight.requires_grad = False
        if base_layer.bias is not None:
            self.bias = base_layer.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        # Trainable adapter matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32))

        # Reset parameters: A random gaussian, B zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Frozen singular bases U_k [out_features, k] and V_k [in_features, k]
        if U_k is not None and V_k is not None:
            self.register_buffer("U_k", U_k.float(), persistent=False)
            self.register_buffer("V_k", V_k.float(), persistent=False)
        else:
            self.register_buffer("U_k", torch.empty(self.out_features, self.protection_rank_k), persistent=False)
            self.register_buffer("V_k", torch.empty(self.in_features, self.protection_rank_k), persistent=False)

    def set_singular_bases(self, U_k: torch.Tensor, V_k: torch.Tensor) -> None:
        """Register pre-computed singular bases."""
        self.U_k.copy_(U_k.float())
        self.V_k.copy_(V_k.float())

    def get_projected_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute effective factors:
        B_tilde = B - U_k (U_k^T B)
        A_tilde = A - (A V_k) V_k^T
        """
        B_tilde = self.lora_B - (self.U_k @ (self.U_k.T @ self.lora_B))
        A_tilde = self.lora_A - ((self.lora_A @ self.V_k) @ self.V_k.T)
        return A_tilde, B_tilde

    def get_delta_weight(self) -> torch.Tensor:
        """Compute Delta W = (alpha / r) * B_tilde * A_tilde in FP32."""
        A_tilde, B_tilde = self.get_projected_factors()
        return self.scaling * (B_tilde @ A_tilde)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)

        A_tilde, B_tilde = self.get_projected_factors()
        x_dropped = self.lora_dropout(x.to(dtype=torch.float32))
        adapter_out = F.linear(F.linear(x_dropped, A_tilde), B_tilde) * self.scaling

        return base_out + adapter_out.to(dtype=base_out.dtype)


def compute_layer_svd(
    weight: torch.Tensor,
    k: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute leading singular bases U_k [out_features, k] and V_k [in_features, k]
    using GPU SVD in FP32.
    """
    orig_device = weight.device
    target_device = device if device is not None else (orig_device if orig_device.type == "cuda" else torch.device("cpu"))

    w_fp32 = weight.detach().to(device=target_device, dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(w_fp32, full_matrices=False)

    k_eff = min(k, U.shape[1], Vh.shape[0])
    U_k = U[:, :k_eff].contiguous()
    V_k = Vh[:k_eff, :].T.contiguous()

    return U_k.to(orig_device), V_k.to(orig_device)


def build_optimizer_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Exclude biases and parameters with fewer than 2 dimensions (e.g. RMSNorm/LayerNorm weights)
    from weight decay, counting tied parameters only once.
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    visited_ids: Set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in visited_ids:
            continue
        visited_ids.add(param_id)

        if param.ndim < 2 or "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Wrap target linear layers in the transformer with LoRALinear."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    for name, module in list(model.named_modules()):
        for target_name in target_modules:
            if target_name in name and isinstance(module, nn.Linear):
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = model.get_submodule(parent_name) if parent_name else model
                lora_layer = LoRALinear(
                    base_layer=module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(parent, child_name, lora_layer)

    return model


def apply_oplora_to_model(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    protection_rank_k: int = 128,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Wrap target linear layers with OPLoRALinear and pre-compute SVD bases."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    overhead_stats = {
        "preprocessing_time_sec": 0.0,
        "basis_storage_mb": 0.0,
        "num_matrices_svd": 0,
    }

    start_time = time.perf_counter()
    total_basis_elements = 0

    for name, module in list(model.named_modules()):
        for target_name in target_modules:
            if target_name in name and isinstance(module, nn.Linear):
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = model.get_submodule(parent_name) if parent_name else model

                U_k, V_k = compute_layer_svd(module.weight, k=protection_rank_k, device=device)
                total_basis_elements += (U_k.numel() + V_k.numel())
                overhead_stats["num_matrices_svd"] += 1

                oplora_layer = OPLoRALinear(
                    base_layer=module,
                    rank=rank,
                    alpha=alpha,
                    protection_rank_k=protection_rank_k,
                    dropout=dropout,
                    U_k=U_k,
                    V_k=V_k,
                )
                setattr(parent, child_name, oplora_layer)

    overhead_stats["preprocessing_time_sec"] = time.perf_counter() - start_time
    overhead_stats["basis_storage_mb"] = (total_basis_elements * 4.0) / (1024.0 * 1024.0)
    print(f"[OPLoRA] SVD Preprocessing: {overhead_stats['num_matrices_svd']} matrices in "
          f"{overhead_stats['preprocessing_time_sec']:.2f}s | Basis storage: {overhead_stats['basis_storage_mb']:.2f} MB")

    return model, overhead_stats


def merge_adapters_into_backbone(
    model: nn.Module,
    verify_logits_parity: bool = False,
    test_batch: Optional[Dict[str, torch.Tensor]] = None,
    atol: float = 1e-3,
) -> nn.Module:
    """Merge adapter corrections (Delta W) into fresh copies of backbone weights."""
    unmerged_logits = None
    if verify_logits_parity and test_batch is not None:
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=test_batch["input_ids"],
                attention_mask=test_batch.get("attention_mask", None),
            )
            unmerged_logits = outputs.logits.detach().float()

    for name, module in list(model.named_modules()):
        if isinstance(module, (LoRALinear, OPLoRALinear)):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model

            delta_w = module.get_delta_weight().to(device=module.weight.device, dtype=module.weight.dtype)
            merged_weight = module.weight.data + delta_w

            linear_layer = nn.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=(module.bias is not None),
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            linear_layer.weight.data.copy_(merged_weight)
            if module.bias is not None:
                linear_layer.bias.data.copy_(module.bias.data)

            setattr(parent, child_name, linear_layer)

    if verify_logits_parity and unmerged_logits is not None and test_batch is not None:
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=test_batch["input_ids"],
                attention_mask=test_batch.get("attention_mask", None),
            )
            merged_logits = outputs.logits.detach().float()
            max_diff = torch.max(torch.abs(unmerged_logits - merged_logits)).item()
            print(f"[Merge] Logit parity check max absolute diff: {max_diff:.6f}")
            if max_diff > atol:
                print(f"[Warning] Logit parity max diff {max_diff} exceeded tolerance {atol}")

    return model
