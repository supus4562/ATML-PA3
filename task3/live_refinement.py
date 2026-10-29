"""
Live generation and sequential refinement experiment on 50 MATH evaluation questions
with Qwen2.5-3B-Instruct on GPU (Step 4).
Compares:
1. Single longer response (3,072 max new tokens)
2. Parallel sampling (3 x 1,024 max new tokens, self-consistency)
3. Sequential refinement (1,024 initial + 2,048 critique-and-revision)
"""

import os
import re
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task3.data import normalize_math_answer, check_answer_correctness


def extract_boxed_answer(text: str) -> str:
    """Extracts the last \\boxed{...} content using balanced brace parsing."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return "NO_ANSWER"
    start = idx + len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return "NO_ANSWER"


def run_live_refinement_experiment(
    questions: List[Dict[str, Any]],
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
    seed: int = 6304,
) -> Dict[str, Any]:
    """
    Executes Step 4 Live Generation on the 50 fixed MATH evaluation questions.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Live Refinement] Loading {model_name} on {dev} in FP16...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if dev.type == "cuda":
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    ).to(dev)
    model.eval()

    gen_kwargs_base = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "do_sample": True,
    }

    results = {
        "single_longer": {
            "correct": 0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_time_sec": 0.0,
            "truncations": 0,
            "extractions": 0,
            "per_question": [],
        },
        "parallel_sampling": {
            "correct": 0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_time_sec": 0.0,
            "truncations": 0,
            "extractions": 0,
            "per_question": [],
        },
        "sequential_refinement": {
            "correct": 0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_time_sec": 0.0,
            "truncations": 0,
            "extractions": 0,
            "wrong_to_correct": 0,
            "correct_to_wrong": 0,
            "retained_correct": 0,
            "retained_wrong": 0,
            "per_question": [],
        },
        "peak_vram_mb": 0.0,
    }

    print(f"[Live Refinement] Commencing 3-strategy evaluation across {len(questions)} MATH questions...")

    for q_idx, q_item in enumerate(questions):
        q_id = q_item["sample_id"]
        q_text = q_item["question"]
        gold_ans = q_item["gold_answer"]

        # Prompt formatting per PA Section 3
        system_msg = "You are a helpful assistant."
        user_instruction = f"{q_text}\n\nThink step by step inside <think> </think> tags, then give your final answer inside \\boxed{{}}."

        # Base conversation
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_instruction},
        ]
        base_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        base_inputs = tokenizer(base_prompt, return_tensors="pt").to(dev)
        base_in_len = int(base_inputs["input_ids"].shape[1])

        # -------------------------------------------------------------
        # Strategy 1: Single Longer Response (3,072 max new tokens)
        # -------------------------------------------------------------
        torch.manual_seed(seed + q_idx * 10 + 1)
        t0 = time.perf_counter()
        with torch.no_grad():
            out1 = model.generate(
                **base_inputs,
                max_new_tokens=3072,
                **gen_kwargs_base,
            )
        dt1 = time.perf_counter() - t0

        gen1_ids = out1[0, base_in_len:]
        out1_len = len(gen1_ids)
        gen1_text = tokenizer.decode(gen1_ids, skip_special_tokens=True)
        ans1_raw = extract_boxed_answer(gen1_text)
        ans1_norm = normalize_math_answer(ans1_raw)
        corr1 = check_answer_correctness(ans1_norm, gold_ans, dataset="MATH")

        trunc1 = bool(out1_len >= 3072)
        extr1 = bool(ans1_raw != "NO_ANSWER")

        s1_res = results["single_longer"]
        s1_res["correct"] += 1 if corr1 else 0
        s1_res["total_tokens_in"] += base_in_len
        s1_res["total_tokens_out"] += out1_len
        s1_res["total_time_sec"] += dt1
        s1_res["truncations"] += 1 if trunc1 else 0
        s1_res["extractions"] += 1 if extr1 else 0
        s1_res["per_question"].append({
            "sample_id": q_id,
            "raw_answer": ans1_raw,
            "normalized_answer": ans1_norm,
            "is_correct": corr1,
            "tokens_in": base_in_len,
            "tokens_out": out1_len,
            "time_sec": dt1,
            "truncated": trunc1,
        })

        # -------------------------------------------------------------
        # Strategy 2: Parallel Sampling (3 x 1,024 max new tokens)
        # -------------------------------------------------------------
        parallel_outs = []
        parallel_lens = []
        parallel_times = []
        parallel_answers = []

        for p_i in range(3):
            torch.manual_seed(seed + q_idx * 100 + p_i + 2)
            t_p0 = time.perf_counter()
            with torch.no_grad():
                out_p = model.generate(
                    **base_inputs,
                    max_new_tokens=1024,
                    **gen_kwargs_base,
                )
            dt_p = time.perf_counter() - t_p0

            gen_p_ids = out_p[0, base_in_len:]
            gen_p_len = len(gen_p_ids)
            gen_p_text = tokenizer.decode(gen_p_ids, skip_special_tokens=True)
            p_ans_raw = extract_boxed_answer(gen_p_text)
            p_ans_norm = normalize_math_answer(p_ans_raw)

            parallel_outs.append(gen_p_text)
            parallel_lens.append(gen_p_len)
            parallel_times.append(dt_p)
            parallel_answers.append(p_ans_norm)

        # Self-consistency majority vote over the 3 parallel responses
        vote_counts: Dict[str, int] = {}
        first_p_idx: Dict[str, int] = {}
        for idx, a in enumerate(parallel_answers):
            vote_counts[a] = vote_counts.get(a, 0) + 1
            if a not in first_p_idx:
                first_p_idx[a] = idx

        sorted_sc = sorted(vote_counts.keys(), key=lambda a: (-vote_counts[a], first_p_idx[a]))
        sc_ans = sorted_sc[0]
        corr_sc = check_answer_correctness(sc_ans, gold_ans, dataset="MATH")

        trunc_sc = sum(1 for l in parallel_lens if l >= 1024)
        extr_sc = sum(1 for a in parallel_answers if a != "NO_ANSWER")

        s2_res = results["parallel_sampling"]
        s2_res["correct"] += 1 if corr_sc else 0
        s2_res["total_tokens_in"] += base_in_len * 3
        s2_res["total_tokens_out"] += sum(parallel_lens)
        s2_res["total_time_sec"] += sum(parallel_times)
        s2_res["truncations"] += trunc_sc
        s2_res["extractions"] += extr_sc
        s2_res["per_question"].append({
            "sample_id": q_id,
            "parallel_answers": parallel_answers,
            "sc_selected_answer": sc_ans,
            "is_correct": corr_sc,
            "tokens_in": base_in_len * 3,
            "tokens_out": sum(parallel_lens),
            "time_sec": sum(parallel_times),
        })

        # -------------------------------------------------------------
        # Strategy 3: Sequential Refinement
        # Reuses the first parallel response as the initial response
        # -------------------------------------------------------------
        init_resp_text = parallel_outs[0]
        init_ans_norm = parallel_answers[0]
        corr_init = check_answer_correctness(init_ans_norm, gold_ans, dataset="MATH")

        refine_instruction = "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."
        refine_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_instruction},
            {"role": "assistant", "content": init_resp_text},
            {"role": "user", "content": refine_instruction},
        ]
        refine_prompt = tokenizer.apply_chat_template(refine_messages, tokenize=False, add_generation_prompt=True)
        refine_inputs = tokenizer(refine_prompt, return_tensors="pt").to(dev)
        refine_in_len = int(refine_inputs["input_ids"].shape[1])

        torch.manual_seed(seed + q_idx * 100 + 99)
        t_ref0 = time.perf_counter()
        with torch.no_grad():
            out_ref = model.generate(
                **refine_inputs,
                max_new_tokens=2048,
                **gen_kwargs_base,
            )
        dt_ref = time.perf_counter() - t_ref0

        gen_ref_ids = out_ref[0, refine_in_len:]
        ref_out_len = len(gen_ref_ids)
        ref_text = tokenizer.decode(gen_ref_ids, skip_special_tokens=True)
        ref_ans_raw = extract_boxed_answer(ref_text)
        ref_ans_norm = normalize_math_answer(ref_ans_raw)
        corr_ref = check_answer_correctness(ref_ans_norm, gold_ans, dataset="MATH")

        # Accounting charges: initial response + critique/revision
        total_ref_in = base_in_len + refine_in_len
        total_ref_out = parallel_lens[0] + ref_out_len
        total_ref_time = parallel_times[0] + dt_ref

        s3_res = results["sequential_refinement"]
        s3_res["correct"] += 1 if corr_ref else 0
        s3_res["total_tokens_in"] += total_ref_in
        s3_res["total_tokens_out"] += total_ref_out
        s3_res["total_time_sec"] += total_ref_time
        s3_res["truncations"] += (1 if parallel_lens[0] >= 1024 else 0) + (1 if ref_out_len >= 2048 else 0)
        s3_res["extractions"] += (1 if init_ans_norm != "NO_ANSWER" else 0) + (1 if ref_ans_norm != "NO_ANSWER" else 0)

        # Transition matrix relative to shared initial answer
        if not corr_init and corr_ref:
            s3_res["wrong_to_correct"] += 1
        elif corr_init and not corr_ref:
            s3_res["correct_to_wrong"] += 1
        elif corr_init and corr_ref:
            s3_res["retained_correct"] += 1
        else:
            s3_res["retained_wrong"] += 1

        s3_res["per_question"].append({
            "sample_id": q_id,
            "initial_answer": init_ans_norm,
            "initial_correct": corr_init,
            "revised_answer": ref_ans_norm,
            "revised_correct": corr_ref,
            "tokens_in": total_ref_in,
            "tokens_out": total_ref_out,
            "time_sec": total_ref_time,
        })

        if (q_idx + 1) % 10 == 0 or (q_idx + 1) == len(questions):
            print(f"  [Live Refinement] Completed {q_idx + 1:2d}/{len(questions)} questions | "
                  f"Single: {s1_res['correct']}/{q_idx+1} | "
                  f"Parallel: {s2_res['correct']}/{q_idx+1} | "
                  f"Refined: {s3_res['correct']}/{q_idx+1}")

    if torch.cuda.is_available():
        results["peak_vram_mb"] = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    # Calculate summary rates
    n_q = len(questions)
    for strat in ["single_longer", "parallel_sampling", "sequential_refinement"]:
        st = results[strat]
        st["accuracy_percent"] = (st["correct"] / n_q) * 100.0
        st["mean_tokens_in"] = st["total_tokens_in"] / n_q
        st["mean_tokens_out"] = st["total_tokens_out"] / n_q
        st["mean_time_sec"] = st["total_time_sec"] / n_q

    # Save artifact
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "live_refinement_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Live Refinement] Saved results to {save_path}")

    # Unload model from GPU
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results
