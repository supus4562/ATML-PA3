"""
Vision-Language Model architectures for Task 2:
- Two-layer MLP Visual Connector (768 -> 1024 -> 1024, GELU)
- Automated Connector Output Magnitude Calibration against text embeddings
- VLM Model wrapper combining Qwen/Qwen3-0.6B with continuous visual token injection
- LoRA adapter injection on q_proj, k_proj, v_proj, o_proj for Stages 2 & 3.
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class VisualConnector(nn.Module):
    """
    Two-layer MLP connecting SigLIP visual patch features to Qwen embedding space.
    Architecture: Linear(768, 1024) -> GELU -> Linear(1024, 1024)
    Maps F(I) in R^{196 x 768} to Z(I) in R^{196 x 1024}.
    """
    def __init__(
        self,
        in_features: int = 768,
        hidden_features: int = 1024,
        out_features: int = 1024,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.linear1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_features, out_features, bias=True)

        # Learned or calibrated scale factor matching connector output magnitude to text embeddings
        self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standard Xavier uniform initialization
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def set_scale(self, scale_value: float) -> None:
        """Set the calibrated output scale factor."""
        self.scale.copy_(torch.tensor(scale_value, dtype=torch.float32))

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Input: [batch_size, num_patches, 768] in FP16 or FP32
        Output: [batch_size, num_patches, 1024]
        """
        orig_dtype = visual_features.dtype
        x = visual_features.float()
        h = self.act(self.linear1(x))
        out = self.linear2(h) * self.scale
        return out.to(dtype=orig_dtype)


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation layer for language model attention projections.
    delta_W = (alpha / r) * B * A
    Frozen base weight in FP16, trainable A and B in FP32.
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

        # Trainable adapter matrices on same device as base layer
        layer_device = base_layer.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32, device=layer_device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32, device=layer_device))

        # Reset parameters: A random gaussian, B zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        x_dropped = self.lora_dropout(x.to(dtype=self.lora_A.dtype))
        adapter_out = F.linear(F.linear(x_dropped, self.lora_A), self.lora_B) * self.scaling
        return base_out + adapter_out.to(dtype=base_out.dtype)


def apply_lora_to_vlm(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Enable LoRA adapters on target projections in Qwen while keeping original backbone frozen.
    Target modules: q_proj, k_proj, v_proj, o_proj.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # 1. Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    # 2. Inject LoRA layers
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

    # 3. Mark only adapter parameters and connector as trainable
    for name, param in model.named_parameters():
        if "lora_" in name or "connector" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    try:
        model_device = next(model.parameters()).device
        model.to(model_device)
    except StopIteration:
        pass

    return model


class VLMModel(nn.Module):
    """
    Unified Vision-Language Model integrating:
    1. VisualConnector (trained throughout)
    2. Qwen/Qwen3-0.6B language model (frozen backbone + LoRA in S2/S3)
    Supports multimodal forward pass and no-image ablation condition.
    """
    def __init__(
        self,
        language_model: AutoModelForCausalLM,
        connector: VisualConnector,
    ):
        super().__init__()
        self.language_model = language_model
        self.connector = connector

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.language_model.model.embed_tokens

    def forward_multimodal(
        self,
        prefix_ids: torch.Tensor,               # [B, L_prefix]
        visual_features: Optional[torch.Tensor],# [B, 196, 768] (None for no-image)
        suffix_ids: torch.Tensor,               # [B, L_suffix]
        attention_mask: torch.Tensor,           # [B, Total_L]
        position_ids: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        Assemble continuous multimodal embeddings:
        [embed_tokens(prefix), Z(I), embed_tokens(suffix)]
        and perform forward pass through Qwen.
        """
        prefix_embeds = self.embed_tokens(prefix_ids)
        suffix_embeds = self.embed_tokens(suffix_ids)

        if visual_features is not None:
            # Map visual features through connector
            visual_embeds = self.connector(visual_features)  # [B, 196, 1024]
            # Ensure dtypes match
            visual_embeds = visual_embeds.to(dtype=prefix_embeds.dtype)
            inputs_embeds = torch.cat([prefix_embeds, visual_embeds, suffix_embeds], dim=1)
        else:
            # No-image condition: visual tokens omitted, instructions unchanged
            inputs_embeds = torch.cat([prefix_embeds, suffix_embeds], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return outputs

    @torch.no_grad()
    def generate_answer(
        self,
        prefix_ids: torch.Tensor,
        visual_features: Optional[torch.Tensor],
        question_ids: torch.Tensor,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 16,
    ) -> List[str]:
        """
        Greedy decoding (temperature=0.0) with a 16-new-token cap per PA spec.
        Visual tokens are placed before the question inside the user turn.
        """
        self.eval()
        device = prefix_ids.device
        batch_size = prefix_ids.size(0)

        prefix_embeds = self.embed_tokens(prefix_ids)
        question_embeds = self.embed_tokens(question_ids)

        if visual_features is not None:
            visual_embeds = self.connector(visual_features).to(dtype=prefix_embeds.dtype)
            inputs_embeds = torch.cat([prefix_embeds, visual_embeds, question_embeds], dim=1)
        else:
            inputs_embeds = torch.cat([prefix_embeds, question_embeds], dim=1)

        seq_len = inputs_embeds.size(1)
        attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
        position_ids = torch.arange(0, seq_len, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

        # Initial forward pass to compute past_key_values
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_tokens = [next_token]
        past_key_values = outputs.past_key_values
        cur_pos = seq_len

        eos_token_id = tokenizer.eos_token_id
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        stop_ids = {eos_token_id, im_end_id}

        for _ in range(max_new_tokens - 1):
            cur_embed = self.embed_tokens(next_token)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device, dtype=torch.long)], dim=1)
            pos_id = torch.tensor([[cur_pos]], device=device, dtype=torch.long).repeat(batch_size, 1)

            outputs = self.language_model(
                inputs_embeds=cur_embed,
                attention_mask=attention_mask,
                position_ids=pos_id,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            cur_pos += 1

            # Check if all batches reached stop token
            all_stopped = True
            for b in range(batch_size):
                tok_val = next_token[b, 0].item()
                if tok_val not in stop_ids:
                    all_stopped = False
                    break
            if all_stopped:
                break

        gen_tensor = torch.cat(generated_tokens, dim=1)  # [B, gen_len]
        answers = []
        for b in range(batch_size):
            toks = gen_tensor[b].tolist()
            # Trim at first stop token
            clean_toks = []
            for t in toks:
                if t in stop_ids:
                    break
                clean_toks.append(t)
            decoded = tokenizer.decode(clean_toks, skip_special_tokens=True).strip()
            answers.append(decoded)

        return answers


def calibrate_connector_scale(
    connector: VisualConnector,
    embed_tokens: nn.Embedding,
    sample_visual_features: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Step 1 Calibration requirement in PA PDF:
    'Calibrate connector output magnitude against text embeddings using training data only,
    and record the rule and resulting scale.'
    Rule:
    Scale = Mean_L2_Norm(text_embeddings) / Mean_L2_Norm(unscaled_connector_output)
    """
    with torch.no_grad():
        # Text embedding magnitude across vocabulary
        all_text_weights = embed_tokens.weight.detach().float()
        mean_text_norm = torch.norm(all_text_weights, p=2, dim=-1).mean().item()

        # Unscaled connector output magnitude on sample visual features
        sample_v = sample_visual_features.to(device).float()
        h = connector.act(connector.linear1(sample_v))
        raw_out = connector.linear2(h)
        mean_vis_norm = torch.norm(raw_out, p=2, dim=-1).mean().item()

        scale = mean_text_norm / max(1e-6, mean_vis_norm)
        connector.set_scale(scale)

        print(f"[Calibration] Connector Output Magnitude Calibration:")
        print(f"  -> Mean Text Embedding L2 Norm: {mean_text_norm:.4f}")
        print(f"  -> Raw Connector Output L2 Norm: {mean_vis_norm:.4f}")
        print(f"  -> Calibration Scale Factor gamma: {scale:.6f}")
        print(f"  -> Calibrated Output L2 Norm: {mean_vis_norm * scale:.4f}")

        return scale
