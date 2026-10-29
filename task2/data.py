"""
Real Data Ingestion, Extraction, and Curriculum Sampling for Task 2:
- Real MS COCO 2014 Captions (18,000 train across 6,000 images, 96 dev)
- Real MS COCO 2014 Annotation-Derived Grounding Pairs (Presence yes/no, Counting 1-4)
- Real Natural VQAv2 Complementary Pairs (agreement >= 8/10, differing consensus answers)
- Automatic download of official MS COCO 2014 and VQAv2 annotations
- Strict image-disjoint split isolation with seed 6304
- Standard VQA answer normalization
- Dynamic batch collation with Qwen chat format and token capping
- Stage curriculum samplers (Stage 1, Stage 2, Stage 3).
"""

import concurrent.futures
import copy
import io
import json
import os
import random
import re
import time
import urllib.request
import zipfile
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


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

    res = re.sub(r"[^\w\s]", " ", res)
    tokens = res.split()
    tokens = [t for t in tokens if t not in ["a", "an", "the"]]
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
        task_type: str,
        pair_id: Optional[str] = None,
        partner_image_id: Optional[int] = None,
        all_human_answers: Optional[List[str]] = None,
        question_type: Optional[str] = None,
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


def download_file_streaming(url: str, dest_path: str, chunk_size: int = 2 * 1024 * 1024, max_retries: int = 3):
    """Downloads a file over HTTP/HTTPS with streaming, progress reporting, and automatic retries."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        return
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    temp_dest = dest_path + ".tmp"

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            print(f"[Download] Fetching {url} -> {dest_path} (Attempt {attempt}/{max_retries})...")
            with urllib.request.urlopen(req, timeout=300) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(temp_dest, "wb") as f_out:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        downloaded += len(chunk)
                        if total > 0 and downloaded % (20 * 1024 * 1024) < chunk_size:
                            print(f"  Downloaded {downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB ({(downloaded/total)*100:.1f}%)")
            os.replace(temp_dest, dest_path)
            print(f"[Download] Successfully completed {os.path.basename(dest_path)} ({os.path.getsize(dest_path)/(1024*1024):.1f} MB).")
            return
        except Exception as e:
            if os.path.exists(temp_dest):
                os.remove(temp_dest)
            print(f"[Download] Attempt {attempt} failed for {url}: {e}")
            if attempt == max_retries:
                raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {e}")
            time.sleep(2.0 * attempt)


def ensure_real_annotations(cache_dir: str) -> Tuple[str, str]:
    """Downloads and extracts official MS COCO 2014 and VQAv2 annotations if missing."""
    os.makedirs(cache_dir, exist_ok=True)
    coco_dir = os.path.join(cache_dir, "coco_annotations")
    vqa_dir = os.path.join(cache_dir, "vqa_annotations")
    os.makedirs(coco_dir, exist_ok=True)
    os.makedirs(vqa_dir, exist_ok=True)

    # 1. COCO 2014 Captions and Instances
    required_coco = [
        "captions_train2014.json",
        "captions_val2014.json",
        "instances_train2014.json",
        "instances_val2014.json",
    ]
    missing_coco = [f for f in required_coco if not os.path.exists(os.path.join(coco_dir, f))]
    if missing_coco:
        coco_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
        coco_zip = os.path.join(cache_dir, "annotations_trainval2014.zip")
        download_file_streaming(coco_url, coco_zip)
        print("[Data Loader] Extracting MS COCO 2014 annotations...")
        with zipfile.ZipFile(coco_zip, "r") as zf:
            for member in zf.namelist():
                base_name = os.path.basename(member)
                if base_name in required_coco:
                    with open(os.path.join(coco_dir, base_name), "wb") as f_out:
                        f_out.write(zf.read(member))
        if os.path.exists(coco_zip):
            os.remove(coco_zip)

    # 2. VQAv2 Annotations & Questions
    vqa_files = [
        ("v2_OpenEnded_mscoco_train2014_questions.json", "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"),
        ("v2_mscoco_train2014_annotations.json", "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"),
        ("v2_OpenEnded_mscoco_val2014_questions.json", "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"),
        ("v2_mscoco_val2014_annotations.json", "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"),
    ]
    for target_json, url in vqa_files:
        dest_p = os.path.join(vqa_dir, target_json)
        if not os.path.exists(dest_p):
            zip_p = os.path.join(cache_dir, os.path.basename(url))
            download_file_streaming(url, zip_p)
            print(f"[Data Loader] Extracting VQAv2 {target_json}...")
            with zipfile.ZipFile(zip_p, "r") as zf:
                for member in zf.namelist():
                    if member.endswith(".json"):
                        with open(dest_p, "wb") as f_out:
                            f_out.write(zf.read(member))
            if os.path.exists(zip_p):
                os.remove(zip_p)

    return coco_dir, vqa_dir


def prepare_task2_dataset_splits(
    cache_dir: str,
    seed: int = 6304,
) -> Dict[str, Any]:
    """
    Ingests official MS COCO 2014 and VQAv2 datasets and partitions into:
    - COCO Captions (18k train, 96 dev)
    - Programmatic Grounding (Presence & Counting)
    - Natural VQAv2 Complementary Pairs (agreement >= 8/10, consensus differing)
    Ensures STRICT IMAGE DISJOINTNESS between train and eval splits.
    """
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "data_manifest_seed6304.json")

    # Download real annotations if missing
    coco_dir, vqa_dir = ensure_real_annotations(cache_dir)

    rng = random.Random(seed)
    image_split_map: Dict[int, str] = {}

    # =========================================================================
    # 1. COCO Captions
    # =========================================================================
    print("[Data Loader] Processing real MS COCO 2014 Captions...")
    with open(os.path.join(coco_dir, "captions_train2014.json"), "r", encoding="utf-8") as f:
        train_cap_data = json.load(f)["annotations"]
    with open(os.path.join(coco_dir, "captions_val2014.json"), "r", encoding="utf-8") as f:
        val_cap_data = json.load(f)["annotations"]

    train_cap_map: Dict[int, List[str]] = {}
    for ann in train_cap_data:
        train_cap_map.setdefault(ann["image_id"], []).append(ann["caption"].strip())

    val_cap_map: Dict[int, List[str]] = {}
    for ann in val_cap_data:
        val_cap_map.setdefault(ann["image_id"], []).append(ann["caption"].strip())

    # Filter to images with >= 3 captions
    eligible_train_cap_imgs = [img_id for img_id, caps in train_cap_map.items() if len(caps) >= 3]
    rng.shuffle(eligible_train_cap_imgs)
    selected_train_cap_imgs = eligible_train_cap_imgs[:6000]

    coco_captions_train: List[VLMExample] = []
    for img_id in selected_train_cap_imgs:
        image_split_map[img_id] = "train2014"
        for cap_idx, c in enumerate(train_cap_map[img_id][:3]):
            coco_captions_train.append(
                VLMExample(
                    example_id=f"cap_train_{img_id}_{cap_idx}",
                    image_id=img_id,
                    question="Describe this image in detail.",
                    answer=c,
                    task_type="caption",
                )
            )

    # 96 images for dev, 1 caption each
    eligible_val_cap_imgs = list(val_cap_map.keys())
    rng.shuffle(eligible_val_cap_imgs)
    selected_dev_cap_imgs = eligible_val_cap_imgs[:96]

    coco_captions_dev: List[VLMExample] = []
    for img_id in selected_dev_cap_imgs:
        image_split_map[img_id] = "val2014"
        c = val_cap_map[img_id][0]
        coco_captions_dev.append(
            VLMExample(
                example_id=f"cap_dev_{img_id}",
                image_id=img_id,
                question="Describe this image in detail.",
                answer=c,
                task_type="caption",
            )
        )

    # =========================================================================
    # 2. Annotation-Derived Grounding Pairs (Presence & Counting)
    # =========================================================================
    print("[Data Loader] Processing real MS COCO 2014 Instances for Grounding Pairs...")
    with open(os.path.join(coco_dir, "instances_train2014.json"), "r", encoding="utf-8") as f:
        train_inst_raw = json.load(f)
    with open(os.path.join(coco_dir, "instances_val2014.json"), "r", encoding="utf-8") as f:
        val_inst_raw = json.load(f)

    cat_map = {c["id"]: c["name"] for c in train_inst_raw["categories"]}
    all_categories = sorted(list(cat_map.values()))

    def build_grounding_structures(inst_data, excluded_imgs):
        img_cat_counts: Dict[int, Dict[str, int]] = {}
        all_imgs = set()
        for ann in inst_data["annotations"]:
            img_id = ann["image_id"]
            if img_id in excluded_imgs:
                continue
            all_imgs.add(img_id)
            cname = cat_map.get(ann["category_id"])
            if cname:
                img_cat_counts.setdefault(img_id, {}).setdefault(cname, 0)
                img_cat_counts[img_id][cname] += 1
        return img_cat_counts, sorted(list(all_imgs))

    val_cat_counts, val_imgs = build_grounding_structures(val_inst_raw, excluded_imgs=set(selected_dev_cap_imgs))
    train_cat_counts, train_imgs = build_grounding_structures(train_inst_raw, excluded_imgs=set(selected_train_cap_imgs))

    def create_grounding_pairs(cat_counts, img_list, presence_target, counting_target, split_name, seed_offset):
        local_rng = random.Random(seed + seed_offset)
        # Pre-index by category
        cat_to_with: Dict[str, List[int]] = {}
        cat_to_without: Dict[str, List[int]] = {}
        cat_and_count: Dict[Tuple[str, int], List[int]] = {}

        for im in img_list:
            image_split_map[im] = split_name
            counts = cat_counts.get(im, {})
            for c in all_categories:
                cnt = counts.get(c, 0)
                if cnt > 0:
                    cat_to_with.setdefault(c, []).append(im)
                    if cnt in [1, 2, 3, 4]:
                        cat_and_count.setdefault((c, cnt), []).append(im)
                else:
                    cat_to_without.setdefault(c, []).append(im)

        pairs = []
        # Presence pairs
        shuffled_cats = list(all_categories)
        local_rng.shuffle(shuffled_cats)
        p_idx = 0
        cat_ptr = 0
        while len(pairs) < presence_target and cat_ptr < len(shuffled_cats) * 10:
            c = shuffled_cats[cat_ptr % len(shuffled_cats)]
            cat_ptr += 1
            w_list = cat_to_with.get(c, [])
            wo_list = cat_to_without.get(c, [])
            if w_list and wo_list:
                im_a = local_rng.choice(w_list)
                im_b = local_rng.choice(wo_list)
                if im_a != im_b:
                    p_id = f"pres_{split_name}_{p_idx}"
                    q = f"Does this image contain a {c}?"
                    ex_a = VLMExample(f"{p_id}_a", im_a, q, "yes", "presence", p_id, im_b, ["yes"] * 10, "yes/no")
                    ex_b = VLMExample(f"{p_id}_b", im_b, q, "no", "presence", p_id, im_a, ["no"] * 10, "yes/no")
                    pairs.append((ex_a, ex_b))
                    p_idx += 1

        # Counting pairs
        c_idx = 0
        cat_ptr = 0
        while len(pairs) < (presence_target + counting_target) and cat_ptr < len(shuffled_cats) * 10:
            c = shuffled_cats[cat_ptr % len(shuffled_cats)]
            cat_ptr += 1
            n1 = local_rng.randint(1, 4)
            n2 = local_rng.randint(1, 4)
            if n1 == n2:
                n2 = 1 if n1 != 1 else 2
            l1 = cat_and_count.get((c, n1), [])
            l2 = cat_and_count.get((c, n2), [])
            if l1 and l2:
                im_a = local_rng.choice(l1)
                im_b = local_rng.choice(l2)
                if im_a != im_b:
                    p_id = f"count_{split_name}_{c_idx}"
                    q = f"How many {c}s can you see?"
                    ex_a = VLMExample(f"{p_id}_a", im_a, q, str(n1), "counting", p_id, im_b, [str(n1)] * 10, "number")
                    ex_b = VLMExample(f"{p_id}_b", im_b, q, str(n2), "counting", p_id, im_a, [str(n2)] * 10, "number")
                    pairs.append((ex_a, ex_b))
                    c_idx += 1

        return pairs

    # Dev: 24 pairs (12 presence, 12 counting); Test: 64 pairs (32 presence, 32 counting)
    rng.shuffle(val_imgs)
    val_grounding_dev_pool = val_imgs[: len(val_imgs) // 2]
    val_grounding_test_pool = val_imgs[len(val_imgs) // 2 :]

    grounding_dev_pairs = create_grounding_pairs(val_cat_counts, val_grounding_dev_pool, 12, 12, "val2014", seed_offset=1)
    grounding_test_pairs = create_grounding_pairs(val_cat_counts, val_grounding_test_pool, 32, 32, "val2014", seed_offset=2)

    # Train grounding pools
    grounding_train_presence_pairs = create_grounding_pairs(train_cat_counts, train_imgs, 1500, 0, "train2014", seed_offset=3)
    grounding_train_counting_pairs = create_grounding_pairs(train_cat_counts, train_imgs, 0, 1500, "train2014", seed_offset=4)

    # =========================================================================
    # 3. Real Natural VQAv2 Complementary Pairs
    # =========================================================================
    print("[Data Loader] Processing real VQAv2 complementary question pairs (agreement >= 8/10)...")

    def extract_real_vqa_pairs(q_json_path, a_json_path, split_name):
        with open(q_json_path, "r", encoding="utf-8") as f:
            q_raw = json.load(f)["questions"]
        with open(a_json_path, "r", encoding="utf-8") as f:
            a_raw = json.load(f)["annotations"]

        q_dict = {q["question_id"]: q for q in q_raw}
        norm_q_groups: Dict[str, List[Dict[str, Any]]] = {}

        for a in a_raw:
            qid = a["question_id"]
            if qid not in q_dict:
                continue
            q_info = q_dict[qid]
            raw_answers = [ans["answer"] for ans in a["answers"]]
            norm_answers = [normalize_vqa_answer(ans) for ans in raw_answers]
            counts = Counter(norm_answers)
            if not counts:
                continue
            consensus, votes = counts.most_common(1)[0]
            # Strict agreement invariant >= 8/10 per PA Section 2
            if votes < 8:
                continue

            # Original consensus answer string
            canonical_answer = raw_answers[0]
            for r_a, n_a in zip(raw_answers, norm_answers):
                if n_a == consensus:
                    canonical_answer = r_a
                    break

            raw_q = q_info["question"].strip()
            norm_q = raw_q.lower().rstrip("?").strip()
            item = {
                "question_id": qid,
                "image_id": q_info["image_id"],
                "question": raw_q,
                "canonical_answer": canonical_answer,
                "all_human_answers": raw_answers,
                "consensus": consensus,
                "answer_type": a.get("answer_type", "other"),
            }
            norm_q_groups.setdefault(norm_q, []).append(item)

        yn_pairs = []
        num_pairs = []
        oth_pairs = []

        for norm_q, items in norm_q_groups.items():
            if len(items) < 2:
                continue
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    it1, it2 = items[i], items[j]
                    if it1["image_id"] != it2["image_id"] and it1["consensus"] != it2["consensus"]:
                        image_split_map[it1["image_id"]] = split_name
                        image_split_map[it2["image_id"]] = split_name
                        p_id = f"vqa_{split_name}_{it1['question_id']}_{it2['question_id']}"
                        atype = it1["answer_type"]
                        ex1 = VLMExample(
                            example_id=f"vqa_{it1['question_id']}",
                            image_id=it1["image_id"],
                            question=it1["question"],
                            answer=it1["canonical_answer"],
                            task_type="natural_vqa",
                            pair_id=p_id,
                            partner_image_id=it2["image_id"],
                            all_human_answers=it1["all_human_answers"],
                            question_type=atype,
                        )
                        ex2 = VLMExample(
                            example_id=f"vqa_{it2['question_id']}",
                            image_id=it2["image_id"],
                            question=it2["question"],
                            answer=it2["canonical_answer"],
                            task_type="natural_vqa",
                            pair_id=p_id,
                            partner_image_id=it1["image_id"],
                            all_human_answers=it2["all_human_answers"],
                            question_type=atype,
                        )
                        pair = (ex1, ex2)
                        if atype == "yes/no":
                            yn_pairs.append(pair)
                        elif atype == "number":
                            num_pairs.append(pair)
                        else:
                            oth_pairs.append(pair)
                        break

        return yn_pairs, num_pairs, oth_pairs

    train_yn, train_num, train_oth = extract_real_vqa_pairs(
        os.path.join(vqa_dir, "v2_OpenEnded_mscoco_train2014_questions.json"),
        os.path.join(vqa_dir, "v2_mscoco_train2014_annotations.json"),
        "train2014",
    )
    val_yn, val_num, val_oth = extract_real_vqa_pairs(
        os.path.join(vqa_dir, "v2_OpenEnded_mscoco_val2014_questions.json"),
        os.path.join(vqa_dir, "v2_mscoco_val2014_annotations.json"),
        "val2014",
    )

    rng.shuffle(train_yn)
    rng.shuffle(train_num)
    rng.shuffle(train_oth)
    rng.shuffle(val_yn)
    rng.shuffle(val_num)
    rng.shuffle(val_oth)

    # 1. Train Natural VQAv2 pairs: 640 yes/no, 256 number, 640 other = 1536 pairs
    natural_vqa_train_pairs = train_yn[:640] + train_num[:256] + train_oth[:640]

    # 2. Strict disjoint partitioning for Val splits (dev vs test vs captions vs grounding)
    dev_excluded = set(selected_dev_cap_imgs)
    for p in grounding_dev_pairs:
        dev_excluded.add(p[0].image_id)
        dev_excluded.add(p[1].image_id)

    test_excluded = set()
    for p in grounding_test_pairs:
        test_excluded.add(p[0].image_id)
        test_excluded.add(p[1].image_id)

    def select_disjoint_pairs(pool: List[Tuple[VLMExample, VLMExample]], count: int, exclude: Set[int]) -> Tuple[List[Tuple[VLMExample, VLMExample]], Set[int]]:
        selected = []
        used = set()
        for p in pool:
            im1, im2 = p[0].image_id, p[1].image_id
            if im1 in exclude or im2 in exclude:
                continue
            if im1 in used or im2 in used:
                continue
            selected.append(p)
            used.add(im1)
            used.add(im2)
            if len(selected) == count:
                break
        return selected, used

    # Dev VQA: 12 yes/no, 8 number, 12 other = 32 pairs
    dev_yn, dev_used_yn = select_disjoint_pairs(val_yn, 12, dev_excluded | test_excluded)
    dev_num, dev_used_num = select_disjoint_pairs(val_num, 8, dev_excluded | test_excluded | dev_used_yn)
    dev_oth, dev_used_oth = select_disjoint_pairs(val_oth, 12, dev_excluded | test_excluded | dev_used_yn | dev_used_num)
    natural_vqa_dev_pairs = dev_yn + dev_num + dev_oth

    dev_all_vqa_used = dev_used_yn | dev_used_num | dev_used_oth

    # Test VQA: 36 yes/no, 24 number, 36 other = 96 pairs
    test_yn, test_used_yn = select_disjoint_pairs(val_yn, 36, dev_excluded | test_excluded | dev_all_vqa_used)
    test_num, test_used_num = select_disjoint_pairs(val_num, 24, dev_excluded | test_excluded | dev_all_vqa_used | test_used_yn)
    test_oth, test_used_oth = select_disjoint_pairs(val_oth, 36, dev_excluded | test_excluded | dev_all_vqa_used | test_used_yn | test_used_num)
    natural_vqa_test_pairs = test_yn + test_num + test_oth

    # Verify and collect all image IDs
    train_ids = set(selected_train_cap_imgs)
    for p in grounding_train_presence_pairs:
        train_ids.add(p[0].image_id)
        train_ids.add(p[1].image_id)
    for p in grounding_train_counting_pairs:
        train_ids.add(p[0].image_id)
        train_ids.add(p[1].image_id)
    for p in natural_vqa_train_pairs:
        train_ids.add(p[0].image_id)
        train_ids.add(p[1].image_id)

    eval_dev_ids = set(selected_dev_cap_imgs)
    for p in grounding_dev_pairs:
        eval_dev_ids.add(p[0].image_id)
        eval_dev_ids.add(p[1].image_id)
    for p in natural_vqa_dev_pairs:
        eval_dev_ids.add(p[0].image_id)
        eval_dev_ids.add(p[1].image_id)

    eval_test_ids = set()
    for p in grounding_test_pairs:
        eval_test_ids.add(p[0].image_id)
        eval_test_ids.add(p[1].image_id)
    for p in natural_vqa_test_pairs:
        eval_test_ids.add(p[0].image_id)
        eval_test_ids.add(p[1].image_id)

    eval_ids = eval_dev_ids | eval_test_ids

    # Enforce strict image disjointness assertions
    assert len(train_ids.intersection(eval_ids)) == 0, f"FATAL: Found {len(train_ids.intersection(eval_ids))} overlapping images between train and eval!"
    assert len(eval_dev_ids.intersection(eval_test_ids)) == 0, f"FATAL: Found {len(eval_dev_ids.intersection(eval_test_ids))} overlapping images between dev and test!"

    for im in train_ids:
        image_split_map[im] = "train2014"
    for im in eval_ids:
        image_split_map[im] = "val2014"

    all_image_ids = train_ids | eval_ids
    print(f"[Data] Strict Image-Disjointness Verified: {len(train_ids)} train images, {len(eval_dev_ids)} dev images, {len(eval_test_ids)} test images (0 overlap). Total unique images: {len(all_image_ids)}.")

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
            "train_unique_images": len(train_ids),
            "dev_unique_images": len(eval_dev_ids),
            "test_unique_images": len(eval_test_ids),
            "total_unique_images": len(all_image_ids),
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
        "all_image_ids": all_image_ids,
        "image_split_map": image_split_map,
    }


_FEATURE_RAM_CACHE: Dict[Tuple[int, bool], torch.Tensor] = {}
_TOKEN_CACHE: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = {}


def preload_features_into_ram(
    image_ids: Set[int],
    feature_cache_dir: str,
    max_workers: int = 32,
    compress: bool = False,
) -> None:
    """Preloads all cached FP16 SigLIP feature tensors into CPU RAM to eliminate disk I/O during training."""
    suffix = "_comp49.pt" if compress else "_196.pt"
    to_load = [img_id for img_id in sorted(list(image_ids)) if (img_id, compress) not in _FEATURE_RAM_CACHE]
    if not to_load:
        print(f"[RAM Cache] All {len(image_ids)} image features already in RAM.")
        return

    print(f"[RAM Cache] Preloading {len(to_load)} feature tensors directly into CPU RAM ({max_workers} threads)...")
    t0 = time.perf_counter()

    def _load_single(img_id: int):
        p = os.path.join(feature_cache_dir, f"img_{img_id}{suffix}")
        if os.path.exists(p):
            feat = torch.load(p, map_location="cpu").float()
            return img_id, feat
        return img_id, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for img_id, feat in executor.map(_load_single, to_load):
            if feat is not None:
                _FEATURE_RAM_CACHE[(img_id, compress)] = feat

    elapsed = time.perf_counter() - t0
    ram_mb = sum(t.element_size() * t.nelement() for t in _FEATURE_RAM_CACHE.values()) / (1024 * 1024)
    print(f"[RAM Cache] Successfully cached {len(_FEATURE_RAM_CACHE)} tensors ({ram_mb:.1f} MB) in RAM in {elapsed:.2f}s!")


class DynamicVLMCollator:
    """
    Collates VLM examples into multimodal tensors matching Qwen's chat template:
    User prefix: <|im_start|>user\n
    [Visual tokens Z(I) inserted here]
    User suffix: \n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>
    Utilizes in-memory RAM caches for features and tokenized text to maximize throughput.
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

        self.prefix_ids = self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)

    def __call__(self, batch: List[VLMExample]) -> Dict[str, Any]:
        batch_size = len(batch)
        prefix_tensor = torch.tensor([self.prefix_ids] * batch_size, dtype=torch.long)

        # Real visual feature retrieval from in-RAM cache or disk
        visual_features_list = []
        compress_flag = (self.num_patches == 49)
        for ex in batch:
            if self.no_image:
                visual_features_list.append(None)
            else:
                key = (ex.image_id, compress_flag)
                if key in _FEATURE_RAM_CACHE:
                    feat = _FEATURE_RAM_CACHE[key]
                else:
                    suffix = "_comp49.pt" if compress_flag else "_196.pt"
                    feat_path = os.path.join(self.feature_cache_dir, f"img_{ex.image_id}{suffix}") if self.feature_cache_dir else None
                    if feat_path and os.path.exists(feat_path):
                        feat = torch.load(feat_path, map_location="cpu").float()
                        _FEATURE_RAM_CACHE[key] = feat
                    else:
                        raise FileNotFoundError(f"Missing cached real SigLIP feature for image {ex.image_id} at {feat_path}!")
                visual_features_list.append(feat)

        if not self.no_image:
            visual_tensor = torch.stack(visual_features_list, dim=0)
        else:
            visual_tensor = None

        suffix_id_list = []
        labels_list = []

        for ex in batch:
            cache_key = (ex.question, ex.answer, ex.task_type)
            if cache_key in _TOKEN_CACHE:
                q_tokens, ans_tokens = _TOKEN_CACHE[cache_key]
            else:
                q_cap = self.question_token_cap
                ans_cap = self.caption_target_cap if ex.task_type == "caption" else self.short_answer_target_cap

                q_tokens = self.tokenizer.encode(f"\n{ex.question}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
                if len(q_tokens) > q_cap:
                    q_tokens = q_tokens[:q_cap]

                ans_tokens = self.tokenizer.encode(f"{ex.answer}<|im_end|>", add_special_tokens=False)
                if len(ans_tokens) > ans_cap:
                    ans_tokens = ans_tokens[:ans_cap]

                _TOKEN_CACHE[cache_key] = (q_tokens, ans_tokens)

            full_suffix = q_tokens + ans_tokens
            suffix_id_list.append(full_suffix)

            labels = [self.ignore_index] * len(q_tokens) + ans_tokens
            labels_list.append(labels)

        max_suffix_len = max(len(s) for s in suffix_id_list)
        padded_suffix = []
        padded_labels = []
        attention_mask_list = []

        for s_ids, l_ids in zip(suffix_id_list, labels_list):
            pad_len = max_suffix_len - len(s_ids)
            padded_s = s_ids + [self.pad_token_id] * pad_len

            vis_len = 0 if self.no_image else self.num_patches
            prefix_vis_labels = [self.ignore_index] * (len(self.prefix_ids) + vis_len)
            padded_l = prefix_vis_labels + l_ids + [self.ignore_index] * pad_len

            total_active_len = len(self.prefix_ids) + vis_len + len(s_ids)
            total_pad_len = len(self.prefix_ids) + vis_len + max_suffix_len
            mask = [1] * total_active_len + [0] * (total_pad_len - total_active_len)

            padded_suffix.append(padded_s)
            padded_labels.append(padded_l)
            attention_mask_list.append(mask)

        suffix_tensor = torch.tensor(padded_suffix, dtype=torch.long)
        labels_tensor = torch.tensor(padded_labels, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)

        return {
            "prefix_ids": prefix_tensor,
            "visual_features": visual_tensor,
            "suffix_ids": suffix_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask,
        }


class StageCurriculumSampler:
    """
    Mixture sampler for Stages 2 and 3 adhering strictly to PA Section 2:
    - Stage 2: 50% presence pairs, 25% counting pairs, 25% caption replay
    - Stage 3: 70% natural VQAv2 pairs, 10% presence replay, 10% counting replay, 10% caption replay
    Each draw contributes two examples (a complete question pair or two caption examples).
    """
    def __init__(
        self,
        presence_pool: Optional[List[Tuple[VLMExample, VLMExample]]] = None,
        counting_pool: Optional[List[Tuple[VLMExample, VLMExample]]] = None,
        caption_pool: Optional[List[VLMExample]] = None,
        natural_vqa_pool: Optional[List[Tuple[VLMExample, VLMExample]]] = None,
        presence_pairs: Optional[List[Tuple[VLMExample, VLMExample]]] = None,
        counting_pairs: Optional[List[Tuple[VLMExample, VLMExample]]] = None,
        captions: Optional[List[VLMExample]] = None,
        natural_vqa_pairs: Optional[List[Tuple[VLMExample, VLMExample]]] = None,
        stage: int = 2,
        seed: int = 6304,
    ):
        self.stage = stage
        self.rng = random.Random(seed)

        pres = presence_pairs if presence_pairs is not None else presence_pool
        cnt = counting_pairs if counting_pairs is not None else counting_pool
        cap = captions if captions is not None else caption_pool
        vqa = natural_vqa_pairs if natural_vqa_pairs is not None else natural_vqa_pool

        self.presence_pool = list(pres) if pres else []
        self.counting_pool = list(cnt) if cnt else []
        self.caption_pool = list(cap) if cap else []
        self.natural_vqa_pool = list(vqa) if vqa else []

        if stage == 2:
            self.choices = ["presence", "counting", "caption"]
            self.weights = [0.50, 0.25, 0.25]
        elif stage == 3:
            self.choices = ["natural_vqa", "presence", "counting", "caption"]
            self.weights = [0.70, 0.10, 0.10, 0.10]
        else:
            raise ValueError(f"Invalid curriculum mixture stage: {stage}")

    def draw_pair(self) -> List[VLMExample]:
        pick = self.rng.choices(self.choices, weights=self.weights, k=1)[0]
        if pick == "presence":
            pair = self.rng.choice(self.presence_pool)
            return [pair[0], pair[1]]
        elif pick == "counting":
            pair = self.rng.choice(self.counting_pool)
            return [pair[0], pair[1]]
        elif pick == "natural_vqa":
            pair = self.rng.choice(self.natural_vqa_pool)
            return [pair[0], pair[1]]
        else:
            c1 = self.rng.choice(self.caption_pool)
            c2 = self.rng.choice(self.caption_pool)
            return [c1, c2]

    def draw_pair_or_two_examples(self) -> List[VLMExample]:
        return self.draw_pair()
