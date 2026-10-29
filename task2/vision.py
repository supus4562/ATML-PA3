"""
SigLIP Vision Encoder extraction, deterministic image preprocessing,
feature caching in FP16, and optional 2x2 spatial patch compression (196 -> 49 tokens).
"""

import concurrent.futures
import os
import time
import urllib.request
from typing import Dict, List, Optional, Tuple, Union, Any, Set
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

        from transformers import AutoImageProcessor, SiglipVisionModel

        print(f"[SigLIP] Loading frozen vision encoder: {model_name} on {self.device} ({dtype})...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
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


def download_coco_image(img_id: int, split: str, temp_dir: str) -> Optional[str]:
    """Downloads a single real COCO JPEG from official CDN to temp_dir."""
    dest_path = os.path.join(temp_dir, f"COCO_{split}_{img_id:012d}.jpg")
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
        return dest_path

    url = f"http://images.cocodataset.org/{split}/COCO_{split}_{img_id:012d}.jpg"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            with open(dest_path, "wb") as f_out:
                f_out.write(data)
            return dest_path
        except Exception as e:
            if attempt == 2:
                print(f"[Vision Download] Warning: Failed to download {url} after 3 attempts: {e}")
                return None
            time.sleep(1.0)
    return None


def cache_real_coco_features(
    image_ids: Set[int],
    image_split_map: Dict[int, str],
    cache_dir: str,
    extractor: SigLIPFeatureExtractor,
    batch_size: int = 64,
    max_workers: int = 32,
    compress: bool = False,
) -> None:
    """
    Downloads missing real MS COCO 2014 JPEGs in parallel and extracts FP16 SigLIP patch features.
    Saves tensors as `img_{img_id}_196.pt` and removes temporary JPEGs immediately to save disk.
    """
    os.makedirs(cache_dir, exist_ok=True)
    temp_dir = os.path.join(cache_dir, "temp_jpegs")
    os.makedirs(temp_dir, exist_ok=True)

    suffix = "_comp49.pt" if compress else "_196.pt"

    missing_ids = [
        img_id for img_id in sorted(list(image_ids))
        if not os.path.exists(os.path.join(cache_dir, f"img_{img_id}{suffix}"))
    ]
    print(f"[Vision Cache] Total unique images required: {len(image_ids)}. Already cached: {len(image_ids) - len(missing_ids)}. Missing: {len(missing_ids)}.")

    if not missing_ids:
        print(f"[Vision Cache] All {len(image_ids)} images are already cached in {cache_dir}!")
        return

    print(f"[Vision Cache] Downloading {len(missing_ids)} missing real COCO images in parallel ({max_workers} threads)...")
    t_start = time.perf_counter()

    chunk_size = 512
    for chunk_start in range(0, len(missing_ids), chunk_size):
        chunk_ids = missing_ids[chunk_start : chunk_start + chunk_size]
        img_path_map: Dict[int, str] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(
                    download_coco_image,
                    img_id,
                    image_split_map.get(img_id, "train2014"),
                    temp_dir,
                ): img_id
                for img_id in chunk_ids
            }
            for future in concurrent.futures.as_completed(future_to_id):
                img_id = future_to_id[future]
                path = future.result()
                if path and os.path.exists(path):
                    img_path_map[img_id] = path

        print(f"[Vision Cache] Chunk {chunk_start // chunk_size + 1}/{(len(missing_ids) + chunk_size - 1) // chunk_size}: Downloaded {len(img_path_map)}/{len(chunk_ids)} images. Extracting features on {extractor.device}...")

        batch_ids = list(img_path_map.keys())
        for b in range(0, len(batch_ids), batch_size):
            cur_batch_ids = batch_ids[b : b + batch_size]
            pil_images = []
            valid_ids = []
            for b_id in cur_batch_ids:
                try:
                    with Image.open(img_path_map[b_id]) as im:
                        pil_images.append(im.convert("RGB"))
                    valid_ids.append(b_id)
                except Exception as e:
                    print(f"[Vision Cache] Corrupt image {b_id}: {e}")

            if not pil_images:
                continue

            with torch.no_grad():
                patch_features = extractor.extract_patch_features(pil_images)  # [B, 196, 768] in FP16
                if compress:
                    patch_features = compress_features_2x2(patch_features)

            for idx, b_id in enumerate(valid_ids):
                feat_tensor = patch_features[idx].cpu()
                out_p = os.path.join(cache_dir, f"img_{b_id}{suffix}")
                torch.save(feat_tensor, out_p)
                if os.path.exists(img_path_map[b_id]):
                    os.remove(img_path_map[b_id])

    try:
        os.rmdir(temp_dir)
    except Exception:
        pass

    final_cached = len([f for f in os.listdir(cache_dir) if f.endswith(suffix)])
    t_elapsed = time.perf_counter() - t_start
    print(f"[Vision Cache] Successfully completed real feature extraction in {t_elapsed:.1f}s! Total cached in {cache_dir}: {final_cached}.")
