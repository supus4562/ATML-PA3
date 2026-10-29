"""
SigLIP Vision Encoder extraction, deterministic image preprocessing,
feature caching in FP16, and optional 2x2 spatial patch compression (196 -> 49 tokens).
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SigLIPFeatureExtractor:
    """
    Wraps google/siglip-base-patch16-224 vision encoder.
    Always frozen; extracts 196 patch features (14x14 grid) of dimension 768.
    Applies final layer normalization exactly once.
    """
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        from transformers import AutoProcessor, SiglipVisionModel

        print(f"[SigLIP] Loading frozen vision encoder: {model_name} on {self.device} ({dtype})...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SiglipVisionModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)

        # Strictly freeze all parameters per PA invariant
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_patch_features(
        self,
        images: Union[List[Image.Image], Image.Image],
    ) -> torch.Tensor:
        """
        Extract 196 patch features in 14x14 spatial order.
        Returns: [batch_size, 196, 768] in FP16.
        """
        if isinstance(images, Image.Image):
            images = [images]

        # Deterministic preprocessing (resize to 224x224 bicubic, normalize)
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)

        # Forward pass: SiglipVisionModel outputs.last_hidden_state applies post_layernorm once
        outputs = self.model(pixel_values=pixel_values)
        patch_features = outputs.last_hidden_state  # [B, 196, 768]

        return patch_features

    def unload(self) -> None:
        """
        Remove vision encoder completely from GPU memory before Qwen loads,
        as explicitly instructed in the PA specification.
        """
        print("[SigLIP] Unloading vision encoder from GPU memory to maximize headroom...")
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compress_features_2x2(features: torch.Tensor) -> torch.Tensor:
    """
    Optional Visual-Token Compression (Step 7 in PA PDF):
    Averages each spatial 2x2 block of the 14x14 feature grid before the connector.
    Input: [B, 196, D] where 196 = 14 x 14.
    Output: [B, 49, D] where 49 = 7 x 7.
    """
    B, N, D = features.shape
    assert N == 196, f"Expected 196 patch features, got {N}"
    grid = features.view(B, 14, 14, D)  # [B, 14, 14, D]
    grid = grid.permute(0, 3, 1, 2)     # [B, D, 14, 14]
    pooled = F.avg_pool2d(grid, kernel_size=2, stride=2)  # [B, D, 7, 7]
    pooled = pooled.permute(0, 2, 3, 1)  # [B, 7, 7, D]
    return pooled.reshape(B, 49, D)


def cache_image_features(
    image_dict: Dict[int, Union[str, Image.Image]],
    cache_dir: str,
    extractor: SigLIPFeatureExtractor,
    batch_size: int = 32,
    compress: bool = False,
) -> Dict[int, str]:
    """
    Pre-extracts and durably caches all image patch features to disk in FP16.
    Returns mapping: image_id -> cached_tensor_path.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cached_paths: Dict[int, str] = {}
    missing_ids: List[int] = []

    suffix = "_comp49.pt" if compress else "_196.pt"

    for img_id in image_dict:
        out_path = os.path.join(cache_dir, f"img_{img_id}{suffix}")
        if os.path.exists(out_path):
            cached_paths[img_id] = out_path
        else:
            missing_ids.append(img_id)

    if not missing_ids:
        print(f"[Vision Cache] All {len(image_dict)} image features already cached in {cache_dir}.")
        return cached_paths

    print(f"[Vision Cache] Extracting and caching features for {len(missing_ids)} images...")
    t_start = time.perf_counter()

    for i in tqdm(range(0, len(missing_ids), batch_size), desc="Caching Features", unit="batch"):
        batch_ids = missing_ids[i:i + batch_size]
        batch_imgs = []
        valid_ids = []

        for b_id in batch_ids:
            item = image_dict[b_id]
            try:
                if isinstance(item, str):
                    img = Image.open(item).convert("RGB")
                elif isinstance(item, Image.Image):
                    img = item.convert("RGB")
                else:
                    continue
                batch_imgs.append(img)
                valid_ids.append(b_id)
            except Exception as e:
                print(f"[Vision Cache] Warning: Failed to open image {b_id}: {e}")

        if not batch_imgs:
            continue

        feats = extractor.extract_patch_features(batch_imgs)  # [B, 196, 768]
        if compress:
            feats = compress_features_2x2(feats)             # [B, 49, 768]

        feats_cpu = feats.detach().cpu().to(torch.float16)

        for idx, b_id in enumerate(valid_ids):
            out_path = os.path.join(cache_dir, f"img_{b_id}{suffix}")
            torch.save(feats_cpu[idx], out_path)
            cached_paths[b_id] = out_path

    elapsed = time.perf_counter() - t_start
    print(f"[Vision Cache] Completed caching {len(cached_paths)} images in {elapsed:.1f}s.")
    return cached_paths
