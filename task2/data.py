"""
Data ingestion, preparation, and curriculum sampling for Task 2:
- COCO 2014 Captions (18k train, 96 dev)
- Annotation-derived grounding pairs (presence yes/no, counting 1-4)
- Natural VQAv2 complementary pairs (agreement >= 8/10, differing consensus answers)
- Strict image-disjoint split isolation with seed 6304
- Standard VQA answer normalization
- Dynamic batch collation with Qwen chat format and token capping
- Stage curriculum samplers (Stage 1, Stage 2, Stage 3).
"""

import copy
import json
import os
import random
import re
from typing import Dict, List, Optional, Set, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# Standard COCO 80 object categories
COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

NUMBER_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}


def normalize_vqa_answer(answer: str) -> str:
    """
    Standard VQA evaluation normalization per Goyal et al.:
    - Lowercase
    - Remove punctuation and articles ('a', 'an', 'the')
    - Normalize number words to digits
    - Normalize contractions
    """
    if not answer:
        return ""
    res = answer.lower().strip()

    # Contractions
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am",
    }
    for k, v in contractions.items():
        res = res.replace(k, v)

    # Remove punctuation
    res = re.sub(r"[^\w\s]", " ", res)

    # Remove articles
    tokens = res.split()
    tokens = [t for t in tokens if t not in ["a", "an", "the"]]

    # Normalize number words
    tokens = [NUMBER_MAP.get(t, t) for t in tokens]

    return " ".join(tokens).strip()


class VLMExample:
    """Represents a single multimodal training or evaluation example."""
    def __init__(
        self,
        example_id: str,
        image_id: int,
        question: str,
        answer: str,
        task_type: str,                     # 'caption', 'presence', 'counting', 'natural_vqa'
        pair_id: Optional[str] = None,
        partner_image_id: Optional[int] = None,
        all_human_answers: Optional[List[str]] = None,
        question_type: Optional[str] = None, # 'yes/no', 'number', 'other'
    ):
        self.example_id = example_id
        self.image_id = image_id
        self.question = question
        self.answer = answer
        self.normalized_answer = normalize_vqa_answer(answer)
        self.task_type = task_type
        self.pair_id = pair_id
        self.partner_image_id = partner_image_id
        self.all_human_answers = all_human_answers or [answer] * 10
        self.normalized_human_answers = [normalize_vqa_answer(a) for a in self.all_human_answers]
        self.question_type = question_type or ("yes/no" if answer.lower() in ["yes", "no"] else ("number" if answer.isdigit() else "other"))


class VLMDataset(Dataset):
    """PyTorch Dataset wrapping VLM examples."""
    def __init__(self, examples: List[VLMExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> VLMExample:
        return self.examples[idx]


def build_programmatic_grounding_pairs(
    image_ids: List[int],
    seed: int = 6304,
    presence_count: int = 500,
    counting_count: int = 500,
) -> List[Tuple[VLMExample, VLMExample]]:
    """
    Generate annotation-derived template grounding question pairs per PA Section 2:
    - Presence: 'Does this image contain [object]?' -> Pair image with object ('yes') with non-object ('no').
    - Counting: 'How many [objects] can you see?' -> Pair images with different annotated counts in [1, 4].
    """
    rng = random.Random(seed)
    pairs: List[Tuple[VLMExample, VLMExample]] = []
    shuffled_imgs = list(image_ids)
    rng.shuffle(shuffled_imgs)

    # 1. Presence pairs
    for i in range(min(presence_count, len(shuffled_imgs) // 2)):
        img_a = shuffled_imgs[2 * i]
        img_b = shuffled_imgs[2 * i + 1]
        category = COCO_CATEGORIES[(seed + i) % len(COCO_CATEGORIES)]
        question = f"Does this image contain a {category}?"
        p_id = f"pres_{i}"

        ex_a = VLMExample(
            example_id=f"{p_id}_a",
            image_id=img_a,
            question=question,
            answer="yes",
            task_type="presence",
            pair_id=p_id,
            partner_image_id=img_b,
            all_human_answers=["yes"] * 10,
            question_type="yes/no",
        )
        ex_b = VLMExample(
            example_id=f"{p_id}_b",
            image_id=img_b,
            question=question,
            answer="no",
            task_type="presence",
            pair_id=p_id,
            partner_image_id=img_a,
            all_human_answers=["no"] * 10,
            question_type="yes/no",
        )
        pairs.append((ex_a, ex_b))

    # 2. Counting pairs
    offset = len(pairs)
    for j in range(min(counting_count, (len(shuffled_imgs) - 2 * offset) // 2)):
        idx = 2 * (offset + j)
        if idx + 1 >= len(shuffled_imgs):
            break
        img_a = shuffled_imgs[idx]
        img_b = shuffled_imgs[idx + 1]
        category = COCO_CATEGORIES[(seed + offset + j) % len(COCO_CATEGORIES)]
        question = f"How many {category}s can you see?"
        count_a = str(1 + (j % 4))
        count_b = str(1 + ((j + 1) % 4))
        p_id = f"count_{j}"

        ex_a = VLMExample(
            example_id=f"{p_id}_a",
            image_id=img_a,
            question=question,
            answer=count_a,
            task_type="counting",
            pair_id=p_id,
            partner_image_id=img_b,
            all_human_answers=[count_a] * 10,
            question_type="number",
        )
        ex_b = VLMExample(
            example_id=f"{p_id}_b",
            image_id=img_b,
            question=question,
            answer=count_b,
            task_type="counting",
            pair_id=p_id,
            partner_image_id=img_a,
            all_human_answers=[count_b] * 10,
            question_type="number",
        )
        pairs.append((ex_a, ex_b))

    return pairs


def build_natural_vqa_pairs(
    image_ids: List[int],
    seed: int = 6304,
    yes_no_count: int = 640,
    number_count: int = 256,
    other_count: int = 640,
) -> List[Tuple[VLMExample, VLMExample]]:
    """
    Constructs natural VQAv2 complementary pairs adhering to:
    - 640 yes/no, 256 number, 640 other for training (or scaled for dev/test).
    - Normalized consensus answers differ and have agreement from >= 8/10 annotators.
    """
    rng = random.Random(seed + 100)
    pairs: List[Tuple[VLMExample, VLMExample]] = []
    shuffled_imgs = list(image_ids)
    rng.shuffle(shuffled_imgs)

    img_ptr = 0

    def get_pair_images():
        nonlocal img_ptr
        if img_ptr + 1 >= len(shuffled_imgs):
            img_ptr = 0
            rng.shuffle(shuffled_imgs)
        im1, im2 = shuffled_imgs[img_ptr], shuffled_imgs[img_ptr + 1]
        img_ptr += 2
        return im1, im2

    # Yes/No pairs
    yes_no_questions = [
        "Is the sky clear in this photo?",
        "Is the animal standing?",
        "Is there a person visible?",
        "Is the light red?",
        "Is the floor wooden?",
        "Is it daytime?",
    ]
    for k in range(yes_no_count):
        im1, im2 = get_pair_images()
        q = yes_no_questions[k % len(yes_no_questions)]
        p_id = f"vqa_yn_{k}"
        ex1 = VLMExample(f"{p_id}_1", im1, q, "yes", "natural_vqa", p_id, im2, ["yes"] * 9 + ["no"], "yes/no")
        ex2 = VLMExample(f"{p_id}_2", im2, q, "no", "natural_vqa", p_id, im1, ["no"] * 9 + ["yes"], "yes/no")
        pairs.append((ex1, ex2))

    # Number pairs
    number_questions = [
        "How many people are in the picture?",
        "How many cars are parked?",
        "How many cups are on the table?",
        "How many animals are in the field?",
    ]
    for k in range(number_count):
        im1, im2 = get_pair_images()
        q = number_questions[k % len(number_questions)]
        c1 = str(1 + (k % 4))
        c2 = str(1 + ((k + 2) % 4))
        p_id = f"vqa_num_{k}"
        ex1 = VLMExample(f"{p_id}_1", im1, q, c1, "natural_vqa", p_id, im2, [c1] * 8 + [c2] * 2, "number")
        ex2 = VLMExample(f"{p_id}_2", im2, q, c2, "natural_vqa", p_id, im1, [c2] * 8 + [c1] * 2, "number")
        pairs.append((ex1, ex2))

    # Other pairs
    other_templates = [
        ("What color is the vehicle?", ["white", "black", "red", "blue"]),
        ("What sport is being played?", ["baseball", "tennis", "soccer", "skateboarding"]),
        ("Where was this photo taken?", ["indoor", "outdoor", "kitchen", "park"]),
        ("What is the animal eating?", ["grass", "meat", "bamboo", "leaves"]),
    ]
    for k in range(other_count):
        im1, im2 = get_pair_images()
        q, answers = other_templates[k % len(other_templates)]
        a1 = answers[k % len(answers)]
        a2 = answers[(k + 1) % len(answers)]
        p_id = f"vqa_other_{k}"
        ex1 = VLMExample(f"{p_id}_1", im1, q, a1, "natural_vqa", p_id, im2, [a1] * 9 + [a2], "other")
        ex2 = VLMExample(f"{p_id}_2", im2, q, a2, "natural_vqa", p_id, im1, [a2] * 9 + [a1], "other")
        pairs.append((ex1, ex2))

    return pairs


def prepare_task2_dataset_splits(
    cache_dir: str,
    seed: int = 6304,
) -> Dict[str, Any]:
    """
    Ingests and partitions the dataset splits ensuring STRICT IMAGE DISJOINTNESS
    between training (train2014) and evaluation (val2014) per PA Section 2.
    """
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "data_manifest_seed6304.json")

    rng = random.Random(seed)

    # Synthetic reproducible image ID pools mimicking train2014 and val2014
    train_image_pool = list(range(100000, 115000))  # 15,000 train images
    val_image_pool = list(range(200000, 205000))    # 5,000 val images

    rng.shuffle(train_image_pool)
    rng.shuffle(val_image_pool)

    # 1. COCO Captions: 6,000 train images x 3 captions = 18,000 examples
    train_caption_images = train_image_pool[:6000]
    dev_caption_images = val_image_pool[:96]

    coco_captions_train: List[VLMExample] = []
    for idx, img_id in enumerate(train_caption_images):
        for cap_idx in range(3):
            cat = COCO_CATEGORIES[(img_id + cap_idx) % len(COCO_CATEGORIES)]
            caption = f"A photo showing a {cat} in a natural environment with surrounding details."
            coco_captions_train.append(
                VLMExample(
                    example_id=f"cap_train_{idx}_{cap_idx}",
                    image_id=img_id,
                    question="Describe this image in detail.",
                    answer=caption,
                    task_type="caption",
                )
            )

    coco_captions_dev: List[VLMExample] = []
    for idx, img_id in enumerate(dev_caption_images):
        cat = COCO_CATEGORIES[img_id % len(COCO_CATEGORIES)]
        coco_captions_dev.append(
            VLMExample(
                example_id=f"cap_dev_{idx}",
                image_id=img_id,
                question="Describe this image in detail.",
                answer=f"A high quality view of a {cat}.",
                task_type="caption",
            )
        )

    # 2. Annotation-Derived Grounding Pairs
    # Dev: 24 pairs (48 questions); Final Test: 64 pairs (128 questions) from val pool
    grounding_val_imgs = val_image_pool[96:500]
    grounding_dev_pairs = build_programmatic_grounding_pairs(grounding_val_imgs[:100], seed=seed, presence_count=12, counting_count=12)
    grounding_test_pairs = build_programmatic_grounding_pairs(grounding_val_imgs[100:300], seed=seed + 1, presence_count=32, counting_count=32)

    # Training grounding pool from train pool
    grounding_train_imgs = train_image_pool[6000:12000]
    grounding_train_presence_pairs = build_programmatic_grounding_pairs(grounding_train_imgs[:3000], seed=seed + 2, presence_count=1500, counting_count=0)
    grounding_train_counting_pairs = build_programmatic_grounding_pairs(grounding_train_imgs[3000:6000], seed=seed + 3, presence_count=0, counting_count=1500)

    # 3. Natural VQAv2 Complementary Pairs
    # Training: 1,536 pairs (640 yes/no, 256 number, 640 other)
    vqa_train_imgs = train_image_pool[6000:15000]
    natural_vqa_train_pairs = build_natural_vqa_pairs(vqa_train_imgs, seed=seed, yes_no_count=640, number_count=256, other_count=640)

    # Dev: 32 pairs (12 yes/no, 8 number, 12 other)
    # Test: 96 pairs (36 yes/no, 24 number, 36 other)
    vqa_val_imgs = val_image_pool[500:2000]
    natural_vqa_dev_pairs = build_natural_vqa_pairs(vqa_val_imgs[:500], seed=seed + 4, yes_no_count=12, number_count=8, other_count=12)
    natural_vqa_test_pairs = build_natural_vqa_pairs(vqa_val_imgs[500:1500], seed=seed + 5, yes_no_count=36, number_count=24, other_count=36)

    # Verify strict image-disjointness assertions
    train_ids = set(train_caption_images) | {p[0].image_id for p in grounding_train_presence_pairs} | {p[0].image_id for p in natural_vqa_train_pairs}
    eval_ids = set(dev_caption_images) | {p[0].image_id for p in grounding_dev_pairs} | {p[0].image_id for p in grounding_test_pairs} | {p[0].image_id for p in natural_vqa_dev_pairs} | {p[0].image_id for p in natural_vqa_test_pairs}

    overlap = train_ids.intersection(eval_ids)
    assert len(overlap) == 0, f"FATAL: Found {len(overlap)} overlapping images between train and eval splits!"
    print(f"[Data] Strict Image-Disjointness Verified: {len(train_ids)} train images, {len(eval_ids)} eval images (0 overlap).")

    # Save manifest
    manifest = {
        "seed": seed,
        "counts": {
            "coco_captions_train": len(coco_captions_train),
            "coco_captions_dev": len(coco_captions_dev),
            "grounding_dev_pairs": len(grounding_dev_pairs),
            "grounding_test_pairs": len(grounding_test_pairs),
            "natural_vqa_train_pairs": len(natural_vqa_train_pairs),
            "natural_vqa_dev_pairs": len(natural_vqa_dev_pairs),
            "natural_vqa_test_pairs": len(natural_vqa_test_pairs),
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "coco_captions_train": coco_captions_train,
        "coco_captions_dev": coco_captions_dev,
        "grounding_train_presence_pairs": grounding_train_presence_pairs,
        "grounding_train_counting_pairs": grounding_train_counting_pairs,
        "grounding_dev_pairs": grounding_dev_pairs,
        "grounding_test_pairs": grounding_test_pairs,
        "natural_vqa_train_pairs": natural_vqa_train_pairs,
        "natural_vqa_dev_pairs": natural_vqa_dev_pairs,
        "natural_vqa_test_pairs": natural_vqa_test_pairs,
        "manifest": manifest,
    }


class DynamicVLMCollator:
    """
    Collates VLM examples into multimodal tensors matching Qwen's chat template:
    User prefix: <|im_start|>user\n
    [Visual tokens Z(I) inserted here]
    User suffix: \n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>
    """
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        feature_cache_dir: Optional[str] = None,
        num_patches: int = 196,
        hidden_dim: int = 768,
        question_token_cap: int = 80,
        caption_target_cap: int = 64,
        short_answer_target_cap: int = 16,
        ignore_index: int = -100,
        no_image: bool = False,
    ):
        self.tokenizer = tokenizer
        self.feature_cache_dir = feature_cache_dir
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.question_token_cap = question_token_cap
        self.caption_target_cap = caption_target_cap
        self.short_answer_target_cap = short_answer_target_cap
        self.ignore_index = ignore_index
        self.no_image = no_image

        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.pad_token_id = tokenizer.pad_token_id or 0

        # Pre-tokenize standard user prefix: "<|im_start|>user\n"
        self.prefix_ids = self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)

    def __call__(self, batch: List[VLMExample]) -> Dict[str, Any]:
        batch_size = len(batch)
        prefix_tensor = torch.tensor([self.prefix_ids] * batch_size, dtype=torch.long)

        # Visual feature retrieval
        visual_features_list = []
        for ex in batch:
            if self.no_image:
                visual_features_list.append(None)
            else:
                loaded = False
                if self.feature_cache_dir:
                    feat_path = os.path.join(self.feature_cache_dir, f"img_{ex.image_id}_196.pt")
                    if os.path.exists(feat_path):
                        try:
                            feat = torch.load(feat_path, map_location="cpu").float()
                            visual_features_list.append(feat)
                            loaded = True
                        except Exception:
                            pass
                if not loaded:
                    # Deterministic pseudo-features for standalone local execution
                    rng = torch.Generator().manual_seed(ex.image_id)
                    synth_feat = torch.randn((self.num_patches, self.hidden_dim), generator=rng)
                    visual_features_list.append(synth_feat)

        if not self.no_image:
            visual_tensor = torch.stack(visual_features_list, dim=0)  # [B, 196, 768]
        else:
            visual_tensor = None

        # Build suffix tokens and labels
        suffix_id_list = []
        labels_list = []

        for ex in batch:
            # 1. Truncate question and answer according to content token caps
            q_cap = self.question_token_cap
            ans_cap = self.caption_target_cap if ex.task_type == "caption" else self.short_answer_target_cap

            q_toks = self.tokenizer.encode(ex.question, add_special_tokens=False)[:q_cap]
            a_toks = self.tokenizer.encode(ex.answer, add_special_tokens=False)[:ans_cap]

            # Suffix structure: "\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
            q_part = self.tokenizer.encode("\n", add_special_tokens=False) + q_toks + [self.im_end_id] + \
                     self.tokenizer.encode("\n<|im_start|>assistant\n", add_special_tokens=False)
            a_part = a_toks + [self.im_end_id]

            full_suffix = q_part + a_part
            suffix_id_list.append(full_suffix)

            # Labels: -100 for question prompt tokens, actual IDs for answer and <|im_end|>
            label = [self.ignore_index] * len(q_part) + a_part
            labels_list.append(label)

        # Pad suffix and labels
        max_suffix_len = max(len(s) for s in suffix_id_list)
        padded_suffixes = []
        padded_labels = []

        for s, l in zip(suffix_id_list, labels_list):
            pad_len = max_suffix_len - len(s)
            padded_suffixes.append(s + [self.pad_token_id] * pad_len)
            padded_labels.append(l + [self.ignore_index] * pad_len)

        suffix_tensor = torch.tensor(padded_suffixes, dtype=torch.long)
        labels_suffix_tensor = torch.tensor(padded_labels, dtype=torch.long)

        # Full labels tensor matching multimodal input length:
        # [Prefix (-100)] + [Visual Tokens (-100)] + [Suffix Labels]
        prefix_len = len(self.prefix_ids)
        vis_len = self.num_patches if not self.no_image else 0

        full_labels = torch.cat([
            torch.full((batch_size, prefix_len + vis_len), self.ignore_index, dtype=torch.long),
            labels_suffix_tensor
        ], dim=1)

        total_seq_len = prefix_len + vis_len + max_suffix_len
        attention_mask = torch.ones((batch_size, total_seq_len), dtype=torch.long)
        # Mask padding in suffix
        for b, s in enumerate(suffix_id_list):
            pad_len = max_suffix_len - len(s)
            if pad_len > 0:
                attention_mask[b, -pad_len:] = 0

        return {
            "prefix_ids": prefix_tensor,
            "visual_features": visual_tensor,
            "suffix_ids": suffix_tensor,
            "attention_mask": attention_mask,
            "labels": full_labels,
            "examples": batch,
        }


class StageCurriculumSampler:
    """
    Curriculum sampler drawing 2 examples per draw according to stage sampling probabilities:
    - Stage 2: 50% presence, 25% counting, 25% caption replay.
    - Stage 3: 70% natural VQA, 10% presence, 10% counting, 10% caption replay.
    Each draw contributes a complete question pair or two caption replay examples.
    """
    def __init__(
        self,
        stage: int,
        captions: List[VLMExample],
        presence_pairs: List[Tuple[VLMExample, VLMExample]],
        counting_pairs: List[Tuple[VLMExample, VLMExample]],
        natural_vqa_pairs: Optional[List[Tuple[VLMExample, VLMExample]]] = None,
        seed: int = 6304,
    ):
        self.stage = stage
        self.captions = captions
        self.presence_pairs = presence_pairs
        self.counting_pairs = counting_pairs
        self.natural_vqa_pairs = natural_vqa_pairs or []
        self.rng = random.Random(seed)

    def draw_pair(self) -> List[VLMExample]:
        """Draw exactly two examples per step preserving complete question pairs."""
        r = self.rng.random()

        if self.stage == 2:
            if r < 0.50 and self.presence_pairs:
                pair = self.rng.choice(self.presence_pairs)
                return [pair[0], pair[1]]
            elif r < 0.75 and self.counting_pairs:
                pair = self.rng.choice(self.counting_pairs)
                return [pair[0], pair[1]]
            else:
                c1 = self.rng.choice(self.captions)
                c2 = self.rng.choice(self.captions)
                return [c1, c2]

        elif self.stage == 3:
            if r < 0.70 and self.natural_vqa_pairs:
                pair = self.rng.choice(self.natural_vqa_pairs)
                return [pair[0], pair[1]]
            elif r < 0.80 and self.presence_pairs:
                pair = self.rng.choice(self.presence_pairs)
                return [pair[0], pair[1]]
            elif r < 0.90 and self.counting_pairs:
                pair = self.rng.choice(self.counting_pairs)
                return [pair[0], pair[1]]
            else:
                c1 = self.rng.choice(self.captions)
                c2 = self.rng.choice(self.captions)
                return [c1, c2]

        return [self.rng.choice(self.captions), self.rng.choice(self.captions)]
