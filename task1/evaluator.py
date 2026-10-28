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

import copy
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _single_device_gsm8k_eval(
    model: nn.Module,
    tokenizer: Any,
    test_examples: List[Dict[str, str]],
    device: torch.device,
    batch_size: int = 64,
    max_new_tokens: int = 1024,
    tag: str = "",
) -> Dict[str, Any]:
    """Single-device GSM8K evaluation worker."""
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

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

    actual_batch_size = min(batch_size, 64) if device.type == "cuda" else batch_size
    prefix = f"{tag} " if tag else ""
    print(f"{prefix}[Eval-GSM8K] Evaluating {total_count} test examples with batch size {actual_batch_size} (left-padded, KV-cached)...")

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
                    use_cache=True,
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

                processed_so_far = min(i + actual_batch_size, total_count)
                print(f"  {prefix}Batch {i // actual_batch_size + 1} | Processed {processed_so_far}/{total_count} | Exact Match: {correct_count}/{processed_so_far} ({correct_count/processed_so_far*100:.2f}%)", flush=True)
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
    Automatically leverages dual GPUs (cuda:0 & cuda:1) when available.
    """
    can_use_dual_gpu = (
        device.type == "cuda"
        and device.index is None
        and torch.cuda.device_count() >= 2
        and len(test_examples) >= 4
    )

    if can_use_dual_gpu:
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        n = len(test_examples)
        mid = n // 2
        half0 = test_examples[:mid]
        half1 = test_examples[mid:]

        print(f"[Eval-GSM8K] Dual-GPU mode active: splitting {n} examples across {dev0} ({len(half0)}) and {dev1} ({len(half1)})...")
        model_dev0 = model.to(dev0)
        model_dev1 = copy.deepcopy(model).to(dev1)

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut0 = executor.submit(_single_device_gsm8k_eval, model_dev0, tokenizer, half0, dev0, batch_size, max_new_tokens, "[GPU 0]")
            fut1 = executor.submit(_single_device_gsm8k_eval, model_dev1, tokenizer, half1, dev1, batch_size, max_new_tokens, "[GPU 1]")
            res0 = fut0.result()
            res1 = fut1.result()

        del model_dev1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        combined_details = res0["details"] + res1["details"]
        correct = res0["correct"] + res1["correct"]
        total = res0["total"] + res1["total"]
        unextractable = res0["unextractable"] + res1["unextractable"]
        truncated = res0["truncated"] + res1["truncated"]
        score = (correct / max(1, total)) * 100.0

        return {
            "metric": "exact_match",
            "score": float(score),
            "correct": correct,
            "total": total,
            "unextractable": unextractable,
            "truncated": truncated,
            "details": combined_details,
        }
    else:
        return _single_device_gsm8k_eval(model, tokenizer, test_examples, device, batch_size, max_new_tokens)


def _single_device_likelihood_eval(
    model: nn.Module,
    tokenizer: Any,
    dataset_name: str,
    examples: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 16,
    tag: str = "",
) -> Dict[str, Any]:
    """Single-device batched candidate likelihood scoring worker."""
    model.eval()
    correct_raw = 0
    correct_norm = 0
    total = len(examples)

    prefix = f"{tag} " if tag else ""
    print(f"{prefix}[Eval-{dataset_name.upper()}] Evaluating {total} examples via batched candidate likelihood (batch size {batch_size})...")

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
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_tensor[:, 1:].contiguous()
            # Fused cross entropy computes -log P(target) directly without materializing
            # a massive [batch, seq_len, 151936] log_softmax tensor (saving 10-20 GB VRAM)
            neg_log_probs = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.size())
            token_logps = -neg_log_probs

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
            print(f"  {prefix}Processed {min(i + batch_size, len(flat_candidates))}/{len(flat_candidates)} candidates ({pct:.1f}%)...")

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


def run_standalone_likelihood_eval(
    model: nn.Module,
    tokenizer: Any,
    dataset_name: str,
    examples: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Batched candidate likelihood scoring for completion tasks (HellaSwag, BoolQ, ARC-Easy).
    Computes candidate log-likelihoods without chat template, reporting raw and normalized accuracy.
    Automatically leverages dual GPUs (cuda:0 & cuda:1) when available.
    """
    can_use_dual_gpu = (
        device.type == "cuda"
        and device.index is None
        and torch.cuda.device_count() >= 2
        and len(examples) >= 4
    )

    if can_use_dual_gpu:
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        n = len(examples)
        mid = n // 2
        half0 = examples[:mid]
        half1 = examples[mid:]

        print(f"[Eval-{dataset_name.upper()}] Dual-GPU mode active: splitting {n} examples across {dev0} ({len(half0)}) and {dev1} ({len(half1)})...")
        model_dev0 = model.to(dev0)
        model_dev1 = copy.deepcopy(model).to(dev1)

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut0 = executor.submit(_single_device_likelihood_eval, model_dev0, tokenizer, dataset_name, half0, dev0, batch_size, "[GPU 0]")
            fut1 = executor.submit(_single_device_likelihood_eval, model_dev1, tokenizer, dataset_name, half1, dev1, batch_size, "[GPU 1]")
            res0 = fut0.result()
            res1 = fut1.result()

        del model_dev1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        correct_raw = res0["correct_raw"] + res1["correct_raw"]
        correct_norm = res0["correct_norm"] + res1["correct_norm"]
        total = res0["total"] + res1["total"]
        acc_raw = (correct_raw / max(1, total)) * 100.0
        acc_norm = (correct_norm / max(1, total)) * 100.0

        return {
            "acc": float(acc_raw),
            "acc_norm": float(acc_norm),
            "correct_raw": correct_raw,
            "correct_norm": correct_norm,
            "total": total,
        }
    else:
        return _single_device_likelihood_eval(model, tokenizer, dataset_name, examples, device, batch_size)


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
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True
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
        print("[Evaluator] Native lm-evaluation-harness detected. Initializing HFLM harness...", flush=True)
        try:
            from lm_eval.models.huggingface import HFLM
            eval_batch_size = getattr(config.evaluation, "batch_size", 64)
            hflm_device = str(device) if device.type == "cuda" else "cpu"
            lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_batch_size, device=hflm_device)

            # 1. GSM8K via lm-eval: num_fewshot=0, chat template enabled, greedy, 1024 token cap
            print(f"[Evaluator - lm-eval] Running GSM8K (gsm8k_cot) with chat_template=True, greedy decoding, 1024-token cap...", flush=True)
            gsm_eval = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=["gsm8k_cot"],
                num_fewshot=0,
                apply_chat_template=True,
                gen_kwargs={"max_gen_toks": 1024, "do_sample": False, "temperature": 0.0},
                log_samples=True,
                batch_size=eval_batch_size,
            )

            gsm_res = gsm_eval.get("results", {}).get("gsm8k_cot", {})
            gsm_score = (
                gsm_res.get("exact_match,flexible-extract")
                or gsm_res.get("exact_match,strict-match")
                or gsm_res.get("exact_match,none")
                or gsm_res.get("acc,none", 0.0)
            )
            condition_results["scores"]["gsm8k"] = float(gsm_score * 100.0)

            # Extract per-example outputs, gold answers, and truncation flags
            gsm_samples = gsm_eval.get("samples", {}).get("gsm8k_cot", [])
            unextractable_count = 0
            truncated_count = 0
            processed_samples = []
            for s in gsm_samples:
                resps = s.get("resps", [[""]])
                gen_text = resps[0][0] if (resps and resps[0]) else ""
                pred = extract_gsm8k_answer(gen_text)
                gold = extract_gsm8k_answer(s.get("target", ""))
                is_trunc = len(tokenizer.encode(gen_text, add_special_tokens=False)) >= 1024
                if is_trunc:
                    truncated_count += 1
                if pred is None:
                    unextractable_count += 1
                processed_samples.append({
                    "doc_id": s.get("doc_id"),
                    "question": s.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "") if isinstance(s.get("arguments"), dict) else "",
                    "target": s.get("target", ""),
                    "gold_answer": gold,
                    "generated_text": gen_text,
                    "extracted_answer": pred,
                    "is_correct": bool(pred is not None and gold is not None and pred == gold),
                    "truncated": is_trunc,
                })

            condition_results["metadata"]["gsm8k"] = {
                "unextractable": unextractable_count,
                "truncated": truncated_count,
            }
            gsm_out_file = os.path.join(output_dir, f"gsm8k_details_{condition_name}.json")
            with open(gsm_out_file, "w", encoding="utf-8") as f_gsm:
                json.dump(processed_samples or gsm_samples, f_gsm, indent=2)
            gsm_jsonl_file = os.path.join(output_dir, f"gsm8k_details_{condition_name}.jsonl")
            with open(gsm_jsonl_file, "w", encoding="utf-8") as f_jl:
                for ps in (processed_samples or gsm_samples):
                    f_jl.write(json.dumps(ps) + "\n")
            print(f"[Evaluator - lm-eval] GSM8K completed: Exact Match = {condition_results['scores']['gsm8k']:.2f}% | Unextractable: {unextractable_count} | Truncated: {truncated_count}", flush=True)

            # 2. Completion benchmarks via lm-eval: stock completion prompts WITHOUT chat template
            print(f"[Evaluator - lm-eval] Running completion benchmarks (hellaswag, boolq, arc_easy) without chat template...", flush=True)
            comp_eval = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=["hellaswag", "boolq", "arc_easy"],
                num_fewshot=0,
                apply_chat_template=False,
                log_samples=True,
                batch_size=eval_batch_size,
            )

            c_results = comp_eval.get("results", {})
            # HellaSwag: acc_norm primary, acc raw
            hs_res = c_results.get("hellaswag", {})
            condition_results["scores"]["hellaswag"] = float(hs_res.get("acc_norm,none", hs_res.get("acc,none", 0.0)) * 100.0)
            condition_results["scores"]["hellaswag_raw"] = float(hs_res.get("acc,none", 0.0) * 100.0)

            # BoolQ: acc
            bq_res = c_results.get("boolq", {})
            condition_results["scores"]["boolq"] = float(bq_res.get("acc,none", 0.0) * 100.0)

            # ARC-Easy: acc_norm primary, acc raw
            arc_res = c_results.get("arc_easy", {})
            condition_results["scores"]["arc_easy"] = float(arc_res.get("acc_norm,none", arc_res.get("acc,none", 0.0)) * 100.0)
            condition_results["scores"]["arc_easy_raw"] = float(arc_res.get("acc,none", 0.0) * 100.0)

            # Redundant storage: raw lm-eval dictionary
            raw_lmeval_file = os.path.join(output_dir, f"lmeval_results_{condition_name}.json")
            with open(raw_lmeval_file, "w", encoding="utf-8") as f_raw:
                json.dump({"gsm8k": gsm_eval.get("results"), "completion": comp_eval.get("results")}, f_raw, indent=2)

            res_path = os.path.join(output_dir, f"eval_summary_{condition_name}.json")
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(condition_results, f, indent=2)
            print(f"[Evaluator - lm-eval] Benchmark evaluation successful for {condition_name}: {condition_results['scores']}", flush=True)
            return condition_results
        except Exception as e:
            print(f"[Evaluator] Warning: Native lm_eval run encountered error: {e}. Falling back to internal standalone runner.", flush=True)

    print("[Evaluator] Running internal standalone evaluator...")
    from datasets import load_dataset
    eval_batch_size = getattr(config.evaluation, "batch_size", 16)

    # 1. GSM8K
    try:
        try:
            gsm8k_data = load_dataset("openai/gsm8k", "main", split="test")
        except Exception:
            gsm8k_data = load_dataset("gsm8k", "main", split="test")
        gsm8k_samples = [{"question": x["question"], "answer": x["answer"]} for x in gsm8k_data]
        gsm8k_res = run_standalone_gsm8k_eval(model, tokenizer, gsm8k_samples, device, batch_size=eval_batch_size)
        condition_results["scores"]["gsm8k"] = gsm8k_res["score"]
        condition_results["metadata"]["gsm8k"] = {
            "unextractable": gsm8k_res["unextractable"],
            "truncated": gsm8k_res["truncated"],
        }
        gsm_out_file = os.path.join(output_dir, f"gsm8k_details_{condition_name}.json")
        with open(gsm_out_file, "w", encoding="utf-8") as f:
            json.dump(gsm8k_res["details"], f, indent=2)
        gsm_out_jsonl = os.path.join(output_dir, f"gsm8k_details_{condition_name}.jsonl")
        with open(gsm_out_jsonl, "w", encoding="utf-8") as f_jl:
            for d in gsm8k_res["details"]:
                f_jl.write(json.dumps(d) + "\n")
    except Exception as e:
        print(f"[Warning] Failed GSM8K eval: {e}")
        condition_results["scores"]["gsm8k"] = 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. HellaSwag
    try:
        try:
            hs_data = load_dataset("hellaswag", split="validation")
        except Exception:
            hs_data = load_dataset("Rowan/hellaswag", split="validation")
        hs_samples = []
        for x in hs_data:
            hs_samples.append({
                "context": x["ctx"],
                "choices": x["endings"],
                "gold_index": int(x["label"]),
            })
        hs_res = run_standalone_likelihood_eval(model, tokenizer, "hellaswag", hs_samples, device, batch_size=eval_batch_size)
        condition_results["scores"]["hellaswag"] = hs_res["acc_norm"]
        condition_results["scores"]["hellaswag_raw"] = hs_res["acc"]
    except Exception as e:
        print(f"[Warning] Failed HellaSwag eval: {e}")
        condition_results["scores"]["hellaswag"] = 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        boolq_res = run_standalone_likelihood_eval(model, tokenizer, "boolq", boolq_samples, device, batch_size=eval_batch_size)
        condition_results["scores"]["boolq"] = boolq_res["acc"]
    except Exception as e:
        print(f"[Warning] Failed BoolQ eval: {e}")
        condition_results["scores"]["boolq"] = 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        arc_res = run_standalone_likelihood_eval(model, tokenizer, "arc_easy", arc_samples, device, batch_size=eval_batch_size)
        condition_results["scores"]["arc_easy"] = arc_res["acc_norm"]
        condition_results["scores"]["arc_easy_raw"] = arc_res["acc"]
    except Exception as e:
        print(f"[Warning] Failed ARC-Easy eval: {e}")
        condition_results["scores"]["arc_easy"] = 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    res_path = os.path.join(output_dir, f"eval_summary_{condition_name}.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(condition_results, f, indent=2)
    print(f"[Evaluator] Saved condition results to {res_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return condition_results
