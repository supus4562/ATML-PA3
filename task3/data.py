"""
Dataset ingestion, split partitioning (seed 6304), answer normalization,
and equivalence grouping for Task 3 (MATH & MathVista rollouts).
"""

import os
import re
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import pandas as pd


@dataclass
class RolloutItem:
    dataset: str
    sample_id: str
    rollout_id: int
    text: str
    token_ids: List[int]
    length: int
    extracted_answer: str
    gold_answer: str
    logp: List[float]
    kl_u_p: List[float]
    # VLM attention totals (per token)
    a_image: Optional[List[float]] = None
    a_prompt: Optional[List[float]] = None
    a_generated: Optional[List[float]] = None
    # Computed metrics
    normalized_answer: str = ""
    is_correct: bool = False
    question_text: str = ""
    question_metadata: Optional[Dict[str, Any]] = None


def normalize_math_answer(answer_str: Optional[str]) -> str:
    """
    Normalizes mathematical answers using standard LaTeX / SymPy canonicalization rules.
    Falls back cleanly if math-verify is not installed.
    """
    if answer_str is None or not str(answer_str).strip():
        return "NO_ANSWER"

    s = str(answer_str).strip()
    if s == "NO_ANSWER":
        return "NO_ANSWER"

    # Remove outer LaTeX wrappers: \text{}, \mathrm{}, \mathbf{}, $, \left, \right
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "").strip()

    # Normalize fractions \frac{a}{b} -> a/b
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\dfrac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)

    # Normalize degrees and units
    s = s.replace("^{\\circ}", "").replace("^{\\circ", "").replace("^{\\prime}", "'")
    s = s.replace("\\%", "%")

    # Remove commas from numbers: 1,000 -> 1000
    s = re.sub(r"(?<=\d),(?=\d)", "", s)

    # Clean whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Try numerical normalization if simple float/int
    try:
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return f"{val:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        pass

    return s.lower() if s else "NO_ANSWER"


def normalize_mathvista_answer(
    answer_str: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Normalizes MathVista answers following MathVista's answer-type, choice, and precision rules.
    Represent multiple-choice answers as option letters.
    """
    if answer_str is None or not str(answer_str).strip():
        return "NO_ANSWER"

    s = str(answer_str).strip()
    if s == "NO_ANSWER":
        return "NO_ANSWER"

    q_type = metadata.get("question_type", "") if metadata else ""
    choices = metadata.get("choices", []) if metadata else []

    # If multiple-choice, extract standard option letter A, B, C, D, ...
    if q_type == "multi_choice" or choices:
        m = re.search(r"\b([A-Z])\b", s.upper())
        if m:
            return m.group(1)
        # Check if response matches one of the choices directly
        for idx, choice in enumerate(choices):
            opt_letter = chr(ord("A") + idx)
            if s.lower() == str(choice).lower():
                return opt_letter

    # Precision tolerance for floating point numbers
    try:
        val = float(s.replace(",", ""))
        if val.is_integer():
            return str(int(val))
        precision = metadata.get("precision", 2) if metadata else 2
        return f"{val:.{precision}f}".rstrip("0").rstrip(".")
    except ValueError:
        pass

    # Clean punctuation & whitespace
    s = re.sub(r"[^\w\s\-\.]", "", s).strip().lower()
    return s if s else "NO_ANSWER"


def check_answer_correctness(
    predicted: str,
    gold: str,
    dataset: str = "MATH",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Grades predicted answer against gold answer using equivalence rules.
    Gold answer is NEVER leaked to selection or stopping rules.
    """
    if predicted == "NO_ANSWER" or not predicted:
        return False

    if dataset == "MATH":
        norm_pred = normalize_math_answer(predicted)
        norm_gold = normalize_math_answer(gold)
        if norm_pred == "NO_ANSWER" or norm_gold == "NO_ANSWER":
            return False
        if norm_pred == norm_gold:
            return True
        # Try float comparison if both are numerical
        try:
            return abs(float(norm_pred) - float(norm_gold)) < 1e-6
        except ValueError:
            return False
    else:
        norm_pred = normalize_mathvista_answer(predicted, metadata)
        norm_gold = normalize_mathvista_answer(gold, metadata)
        if norm_pred == "NO_ANSWER" or norm_gold == "NO_ANSWER":
            return False
        if norm_pred == norm_gold:
            return True
        try:
            return abs(float(norm_pred) - float(norm_gold)) < 1e-3
        except ValueError:
            return False


def partition_dataset_splits(
    question_ids: List[str],
    seed: int = 6304,
    dev_count: int = 200,
    eval_count: int = 800,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Strict split partitioning per PA Section 3:
    1. Sort the 1,000 question IDs lexicographically.
    2. Permute them with NumPy's default_rng(6304).
    3. First 200 questions = Development split.
    4. Remaining 800 questions = Held-out Evaluation split.
    5. First 50 MATH evaluation questions = Live Generation subtask.
    """
    sorted_ids = sorted(list(set(question_ids)))
    assert len(sorted_ids) == (dev_count + eval_count), f"Expected {dev_count + eval_count} unique questions, got {len(sorted_ids)}"

    rng = np.random.default_rng(seed)
    permuted_ids = [sorted_ids[i] for i in rng.permutation(len(sorted_ids))]

    dev_ids = permuted_ids[:dev_count]
    eval_ids = permuted_ids[dev_count : dev_count + eval_count]
    live_ids = eval_ids[:50]

    assert len(set(dev_ids).intersection(set(eval_ids))) == 0, "FATAL: Data leakage between dev and eval splits!"
    return dev_ids, eval_ids, live_ids


def load_task3_dataset(
    data_dir: str,
    dataset_name: str,
    rollout_folder: str,
) -> Dict[str, List[RolloutItem]]:
    """
    Loads and joins generations.parquet, probability_metrics.parquet,
    and attention_metrics.parquet (for VLM).
    Returns mapping from sample_id to its 32 RolloutItem objects.
    """
    folder_path = os.path.join(data_dir, rollout_folder)
    if not os.path.exists(folder_path):
        # Check alternate locations
        alt_path = os.path.join(data_dir, dataset_name)
        if os.path.exists(alt_path):
            folder_path = alt_path
        else:
            raise FileNotFoundError(
                f"Missing rollout data directory for {dataset_name} at '{folder_path}'. "
                f"Please ensure ATML-PA3-Data is located in '{data_dir}'."
            )

    gen_path = os.path.join(folder_path, "generations.parquet")
    prob_path = os.path.join(folder_path, "probability_metrics.parquet")
    attn_path = os.path.join(folder_path, "attention_metrics.parquet")

    if not os.path.exists(gen_path) or not os.path.exists(prob_path):
        raise FileNotFoundError(f"Missing required parquet files in {folder_path} (expected generations.parquet and probability_metrics.parquet)")

    print(f"[Data] Reading {gen_path}...")
    df_gen = pd.read_parquet(gen_path)
    print(f"[Data] Reading {prob_path}...")
    df_prob = pd.read_parquet(prob_path)

    join_cols = ["dataset", "sample_id", "rollout_id"]
    for col in join_cols:
        if col not in df_gen.columns:
            # Fallback if sample_id / rollout_id named slightly differently
            if col == "sample_id" and "question_id" in df_gen.columns:
                df_gen["sample_id"] = df_gen["question_id"].astype(str)
            if col == "dataset":
                df_gen["dataset"] = dataset_name

    # Merge generations and probabilities
    df_merged = pd.merge(df_gen, df_prob, on=join_cols, how="inner")

    has_attention = os.path.exists(attn_path)
    if has_attention:
        print(f"[Data] Reading {attn_path}...")
        df_attn = pd.read_parquet(attn_path)
        df_merged = pd.merge(df_merged, df_attn, on=join_cols, how="left")

    print(f"[Data] Loaded {len(df_merged)} merged rollouts for {dataset_name}.")

    grouped: Dict[str, List[RolloutItem]] = {}
    for _, row in df_merged.iterrows():
        s_id = str(row["sample_id"])
        r_id = int(row["rollout_id"])

        text = str(row.get("text", row.get("generation", "")))
        token_ids = list(row.get("token_ids", []))
        length = int(row.get("length", len(token_ids)))
        extracted_ans = str(row.get("extracted_answer", ""))
        gold_ans = str(row.get("gold_answer", ""))

        logp_vals = [float(x) for x in row.get("logp", [])]
        kl_vals = [float(x) for x in row.get("kl_u_p", [])]

        a_img = [float(x) for x in row.get("Aimage", [])] if "Aimage" in row and row["Aimage"] is not None else None
        a_prm = [float(x) for x in row.get("Aprompt", [])] if "Aprompt" in row and row["Aprompt"] is not None else None
        a_gen = [float(x) for x in row.get("Agenerated", [])] if "Agenerated" in row and row["Agenerated"] is not None else None

        q_meta = row.get("question_metadata", None)
        if isinstance(q_meta, str):
            try:
                q_meta = json.loads(q_meta)
            except Exception:
                q_meta = None

        # Precompute normalized answer and correctness
        if dataset_name == "MATH":
            norm_ans = normalize_math_answer(extracted_ans)
        else:
            norm_ans = normalize_mathvista_answer(extracted_ans, q_meta)

        correct = check_answer_correctness(extracted_ans, gold_ans, dataset_name, q_meta)

        item = RolloutItem(
            dataset=dataset_name,
            sample_id=s_id,
            rollout_id=r_id,
            text=text,
            token_ids=token_ids,
            length=length,
            extracted_answer=extracted_ans,
            gold_answer=gold_ans,
            logp=logp_vals,
            kl_u_p=kl_vals,
            a_image=a_img,
            a_prompt=a_prm,
            a_generated=a_gen,
            normalized_answer=norm_ans,
            is_correct=correct,
            question_text=str(row.get("question", "")),
            question_metadata=q_meta,
        )

        if s_id not in grouped:
            grouped[s_id] = []
        grouped[s_id].append(item)

    # Sort rollouts within each question by rollout_id
    for s_id in grouped:
        grouped[s_id].sort(key=lambda x: x.rollout_id)

    print(f"[Data] Partitioned {len(grouped)} questions ({sum(len(v) for v in grouped.values())} total rollouts) for {dataset_name}.")
    return grouped
