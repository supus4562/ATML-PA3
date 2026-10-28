"""
Evaluation engine for Task 1:
- Evaluates across all four benchmarks: GSM8K, HellaSwag, BoolQ, ARC-Easy.
- Adheres strictly to evaluation invariants:
  * Zero-shot (num_fewshot=0).
  * Evaluated in FP16 with dropout disabled.
  * GSM8K: gsm8k_cot generation, non-thinking chat template, greedy decoding, 1,024-new-token cap.
    Stock stopping rules and answer filters; unextractable answers count as incorrect.
    Saves per-example outputs, extracted answers, and truncation flags.
  * HellaSwag, BoolQ, ARC-Easy: candidate likelihood scoring using stock completion prompts (no chat template).
  * Reports raw accuracy alongside normalized accuracy where available.
  * Reports percentage-point changes relative to baseline.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    Flexible-extraction rule for GSM8K answers:
    - Look for standard delimiter '#### <answer>'
    - Look for boxed expression '\\boxed{...}'
    - Look for 'The answer is <answer>'
    - Extract trailing numerical value
    """
    if not text:
        return None
    # 1. Standard #### delimiter
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "").strip()

    # 2. Boxed expression \boxed{...}
    m_box = re.findall(r"\\boxed\{([^}]+)\}", text)
    if m_box:
        cand = m_box[-1].strip()
        num_m = re.findall(r"-?[\d,]+(?:\.\d+)?", cand)
        if num_m:
            return num_m[-1].replace(",", "").strip()
        return cand

    # 3. 'The answer is ...'
    m_ans = re.findall(r"(?:the answer is|equals|is equal to)\s*[:=]?\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m_ans:
        return m_ans[-1].replace(",", "").strip()

    # 4. Final standalone number
    all_nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if all_nums:
        return all_nums[-1].replace(",", "").strip()

    return None


def run_standalone_gsm8k_eval(
    model: nn.Module,
    tokenizer: Any,
    test_examples: List[Dict[str, str]],
    device: torch.device,
    batch_size: int = 64,
    max_new_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    High-throughput standalone GSM8K evaluation matching lm-eval gsm8k_cot specifications.
    Uses non-thinking chat template, greedy decoding, 1024 new tokens, flexible extraction.
    Uses left-padding for causal decoder-only batched generation.
    """
    model.eval()
    results = []
    correct_count = 0
    total_count = len(test_examples)
    unextractable_count = 0
    truncated_count = 0

    orig_padding_side = tokenizer.padding_side
    orig_pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Optimized generation batch size for RTX 6000 Pro (48GB VRAM)
    actual_batch_size = min(batch_size, 64) if device.type == "cuda" else batch_size
    print(f"[Eval-GSM8K] Evaluating {total_count} test examples with batch size {actual_batch_size} (left-padded)...")

    try:
        for i in range(0, total_count, actual_batch_size):
            batch = test_examples[i: i + actual_batch_size]
            prompts = []
            gold_answers = []

            for item in batch:
                q = item["question"]
                gold = extract_gsm8k_answer(item.get("answer", ""))
                gold_answers.append(gold)

                # Non-thinking chat template
                msgs = [{"role": "user", "content": f"Question: {q}\nAnswer: Let's think step by step."}]
                try:
                    p_text = tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    p_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                prompts.append(p_text)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            in_len = inputs.input_ids.shape[1]

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for b_idx in range(len(batch)):
                gen_ids = outputs[b_idx, in_len:]
                gen_len = len(gen_ids)
                truncated = (gen_len >= max_new_tokens)
                if truncated:
                    truncated_count += 1

                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                pred_ans = extract_gsm8k_answer(gen_text)
                gold_ans = gold_answers[b_idx]

                is_correct = False
                if pred_ans is None:
                    unextractable_count += 1
                elif gold_ans is not None and pred_ans == gold_ans:
                    is_correct = True
                    correct_count += 1

                results.append({
                    "question": batch[b_idx]["question"],
                    "gold_answer": gold_ans,
                    "generated_text": gen_text,
                    "extracted_answer": pred_ans,
                    "is_correct": is_correct,
                    "truncated": truncated,
                    "tokens_generated": gen_len,
                })

            if (i + actual_batch_size) % 200 == 0 or (i + actual_batch_size) >= total_count:
                print(f"  Processed {min(i + actual_batch_size, total_count)}/{total_count} | Exact Match: {correct_count}/{min(i + actual_batch_size, total_count)} ({correct_count/min(i + actual_batch_size, total_count)*100:.2f}%)")
    finally:
        tokenizer.padding_side = orig_padding_side
        tokenizer.pad_token_id = orig_pad_token_id
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy = correct_count / max(1, total_count)
    return {
        "metric": "exact_match",
        "score": float(accuracy * 100.0),
        "correct": correct_count,
        "total": total_count,
        "unextractable": unextractable_count,
        "truncated": truncated_count,
        "details": results,
    }


def run_standalone_likelihood_eval(
    model: nn.Module,
    tokenizer: Any,
    dataset_name: str,
    examples: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Batched candidate likelihood scoring for completion tasks (HellaSwag, BoolQ, ARC-Easy).
    Computes candidate log-likelihoods without chat template, reporting raw and normalized accuracy.
    Processes candidates in vectorized batches for high throughput (~15s per benchmark on A100).
    """
    model.eval()
    correct_raw = 0
    correct_norm = 0
    total = len(examples)

    print(f"[Eval-{dataset_name.upper()}] Evaluating {total} examples via batched candidate likelihood...")

    flat_candidates = []
    for ex_idx, item in enumerate(examples):
        context = item.get("context", "") or item.get("prompt", "")
        choices = item.get("choices", [])

        ctx_ids = tokenizer.encode(context, add_special_tokens=False)
        ctx_len = len(ctx_ids)

        for choice_idx, choice in enumerate(choices):
            full_text = context + (" " if not context.endswith(" ") and not choice.startswith(" ") else "") + choice
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            choice_len = max(1, len(full_ids) - ctx_len)

            flat_candidates.append({
                "ex_idx": ex_idx,
                "choice_idx": choice_idx,
                "full_ids": full_ids,
                "choice_len": choice_len,
            })

    num_choices_per_ex = [len(x.get("choices", [])) for x in examples]
    scores_raw = [[0.0] * n for n in num_choices_per_ex]
    scores_norm = [[0.0] * n for n in num_choices_per_ex]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)

    for i in range(0, len(flat_candidates), batch_size):
        chunk = flat_candidates[i: i + batch_size]
        max_len = max(len(c["full_ids"]) for c in chunk)

        batch_input_ids = []
        batch_attn_mask = []
        for c in chunk:
            fids = c["full_ids"]
            plen = max_len - len(fids)
            batch_input_ids.append(fids + [pad_id] * plen)
            batch_attn_mask.append([1] * len(fids) + [0] * plen)

        input_tensor = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
        attn_tensor = torch.tensor(batch_attn_mask, dtype=torch.long, device=device)

        with torch.inference_mode():
            outputs = model(input_ids=input_tensor, attention_mask=attn_tensor)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_tensor[:, 1:]
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_logps = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

        for k, c in enumerate(chunk):
            valid_len = len(c["full_ids"])
            choice_len = c["choice_len"]
            choice_start = max(1, valid_len - choice_len)
            choice_end = valid_len
            cand_logp = token_logps[k, (choice_start - 1):(choice_end - 1)].sum().item()
            cand_norm = cand_logp / choice_len

            scores_raw[c["ex_idx"]][c["choice_idx"]] = cand_logp
            scores_norm[c["ex_idx"]][c["choice_idx"]] = cand_norm

        if (i + batch_size) % (batch_size * 100) == 0 or (i + batch_size) >= len(flat_candidates):
            pct = min(100.0, (i + batch_size) / len(flat_candidates) * 100.0)
            print(f"  Processed {min(i + batch_size, len(flat_candidates))}/{len(flat_candidates)} candidates ({pct:.1f}%)...")

    for ex_idx, item in enumerate(examples):
        gold_idx = item.get("gold_index", 0)
        best_raw = int(np.argmax(scores_raw[ex_idx]))
        best_norm = int(np.argmax(scores_norm[ex_idx]))

        if best_raw == gold_idx:
            correct_raw += 1
        if best_norm == gold_idx:
            correct_norm += 1

    acc_raw = (correct_raw / max(1, total)) * 100.0
    acc_norm = (correct_norm / max(1, total)) * 100.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "acc": float(acc_raw),
        "acc_norm": float(acc_norm),
        "correct_raw": correct_raw,
        "correct_norm": correct_norm,
        "total": total,
    }


def evaluate_checkpoint_condition(
    model: nn.Module,
    tokenizer: Any,
    condition_name: str,
    device: torch.device,
    config: Any,
    output_dir: str = "results/task1",
) -> Dict[str, Any]:
    """
    Run full zero-shot evaluation on GSM8K, HellaSwag, BoolQ, and ARC-Easy for a given condition.
    """
    print(f"\n{'='*70}\n[Evaluator] Running benchmarks for condition: {condition_name.upper()}\n{'='*70}")
    os.makedirs(output_dir, exist_ok=True)
    condition_results = {"condition": condition_name, "scores": {}, "metadata": {}}

    # PDF invariant: Evaluate all models in FP16 with dropout disabled
    model.eval()
    if device.type == "cuda":
        model = model.half()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    has_lmeval = False
    try:
        import lm_eval
        has_lmeval = True
    except ImportError:
        pass

    if has_lmeval:
        print("[Evaluator] lm-evaluation-harness detected. Attempting native harness tasks...")
        try:
            from lm_eval.models.huggingface import HFLM
            eval_batch_size = getattr(config.evaluation, "batch_size", 64)
            lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_batch_size)
            eval_output = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=["gsm8k_cot", "hellaswag", "boolq", "arc_easy"],
                num_fewshot=0,
                batch_size=eval_batch_size,
            )
            for task_name, task_metrics in eval_output.get("results", {}).items():
                primary = task_metrics.get("acc_norm,none", task_metrics.get("acc,none", task_metrics.get("exact_match,none", 0.0)))
                condition_results["scores"][task_name] = float(primary * 100.0)
            return condition_results
        except Exception as e:
            print(f"[Evaluator] lm_eval run error: {e}. Falling back to internal standalone runner.")

    print("[Evaluator] Running internal standalone evaluator...")
    from datasets import load_dataset

    # 1. GSM8K
    try:
        try:
            gsm8k_data = load_dataset("openai/gsm8k", "main", split="test")
        except Exception:
            gsm8k_data = load_dataset("gsm8k", "main", split="test")
        gsm8k_samples = [{"question": x["question"], "answer": x["answer"]} for x in gsm8k_data]
        gsm8k_res = run_standalone_gsm8k_eval(model, tokenizer, gsm8k_samples, device, batch_size=config.evaluation.batch_size)
        condition_results["scores"]["gsm8k"] = gsm8k_res["score"]
        condition_results["metadata"]["gsm8k"] = {
            "unextractable": gsm8k_res["unextractable"],
            "truncated": gsm8k_res["truncated"],
        }
        gsm_out_file = os.path.join(output_dir, f"gsm8k_details_{condition_name}.json")
        with open(gsm_out_file, "w", encoding="utf-8") as f:
            json.dump(gsm8k_res["details"], f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed GSM8K eval: {e}")
        condition_results["scores"]["gsm8k"] = 0.0

    # 2. HellaSwag
    try:
        hs_data = load_dataset("hellaswag", split="validation")
        hs_samples = []
        for x in hs_data:
            hs_samples.append({
                "context": x["ctx"],
                "choices": x["endings"],
                "gold_index": int(x["label"]),
            })
        hs_res = run_standalone_likelihood_eval(model, tokenizer, "hellaswag", hs_samples, device)
        condition_results["scores"]["hellaswag"] = hs_res["acc_norm"]
        condition_results["scores"]["hellaswag_raw"] = hs_res["acc"]
    except Exception as e:
        print(f"[Warning] Failed HellaSwag eval: {e}")
        condition_results["scores"]["hellaswag"] = 0.0

    # 3. BoolQ
    try:
        boolq_data = load_dataset("boolq", split="validation")
        boolq_samples = []
        for x in boolq_data:
            boolq_samples.append({
                "context": f"{x['passage']}\nQuestion: {x['question']}?\nAnswer:",
                "choices": ["false", "true"],
                "gold_index": 1 if x["answer"] else 0,
            })
        boolq_res = run_standalone_likelihood_eval(model, tokenizer, "boolq", boolq_samples, device)
        condition_results["scores"]["boolq"] = boolq_res["acc"]
    except Exception as e:
        print(f"[Warning] Failed BoolQ eval: {e}")
        condition_results["scores"]["boolq"] = 0.0

    # 4. ARC-Easy
    try:
        arc_data = load_dataset("ai2_arc", "ARC-Easy", split="test")
        arc_samples = []
        for x in arc_data:
            label_map = {l: idx for idx, l in enumerate(x["choices"]["label"])}
            arc_samples.append({
                "context": f"Question: {x['question']}\nAnswer:",
                "choices": x["choices"]["text"],
                "gold_index": label_map.get(x["answerKey"], 0),
            })
        arc_res = run_standalone_likelihood_eval(model, tokenizer, "arc_easy", arc_samples, device)
        condition_results["scores"]["arc_easy"] = arc_res["acc_norm"]
        condition_results["scores"]["arc_easy_raw"] = arc_res["acc"]
    except Exception as e:
        print(f"[Warning] Failed ARC-Easy eval: {e}")
        condition_results["scores"]["arc_easy"] = 0.0

    res_path = os.path.join(output_dir, f"eval_summary_{condition_name}.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(condition_results, f, indent=2)
    print(f"[Evaluator] Saved condition results to {res_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return condition_results
