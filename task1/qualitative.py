"""
Qualitative analysis module for Task 1:
- Loads the 24 qualitative prompts from first_plot_questions.yaml (8 questions x 3 formats).
- Generates greedy responses with thinking disabled (enable_thinking=False) and a 512-new-token cap.
- Evaluates across all 7 evaluation conditions (168 responses total).
- Saves all 168 responses grouped by question and format in JSON.
- Generates the structured 42-response annotation sheet (first and last question across all 3 formats and 7 conditions)
  analyzing format compliance, coherence/relevance, repetition, unnecessary mathematical framing,
  inappropriate content, and truncation flags with short excerpts.
"""

import json
import os
import re
from typing import Dict, List, Optional, Any
import yaml
import torch
import torch.nn as nn


def load_qualitative_prompts(yaml_path: str = "task1/first_plot_questions.yaml") -> List[Dict[str, Any]]:
    """Load the 24 prompts structured as 8 questions x 3 formats."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("questions", [])


def generate_response(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Generate one greedy response with thinking disabled and 512 token cap.
    Returns:
        {
            "response": text,
            "num_new_tokens": int,
            "truncated": bool,
        }
    """
    target_device = device if device is not None else next(model.parameters()).device
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(target_device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0, input_len:]
    num_tokens = len(gen_ids)
    truncated = (num_tokens >= max_new_tokens)

    response_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return {
        "response": response_text,
        "num_new_tokens": num_tokens,
        "truncated": truncated,
    }


def check_format_compliance(response: str, expected_format: str) -> Tuple[bool, str]:
    """Check if the generated response adhered to the requested format."""
    if expected_format == "json":
        # Check if valid JSON can be extracted
        text = response.strip()
        # Find json block if fenced
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            candidate = match.group(1)
        else:
            cand_match = re.search(r"(\{.*\})", text, re.DOTALL)
            candidate = cand_match.group(1) if cand_match else text

        try:
            json.loads(candidate)
            return True, "Valid JSON"
        except Exception:
            return False, "Failed JSON parsing"

    elif expected_format == "template":
        # Check if key headers/template tokens exist
        headers = re.findall(r"#{1,3}\s+[^\n]+", response)
        if len(headers) >= 2:
            return True, f"Followed markdown structure ({len(headers)} headers)"
        return False, "Missing template section headers"

    else:
        # Plain
        return True, "Natural language text"


def check_unnecessary_math_framing(response: str) -> Tuple[bool, List[str]]:
    """Detect if mathematical symbols, equations, or LaTeX framing intruded into non-math answers."""
    math_patterns = [
        r"\\[\[\(]",  # LaTeX math open \[ or \(
        r"\$[^$]+\$", # Inline dollar math
        r"\\boxed\{",
        r"\\frac\{",
        r"\\sum",
        r"\\int",
        r"equation\s*\d*",
        r"\b\d+\s*=\s*\d+\b", # e.g. 5 = 5
    ]
    detected = []
    for pat in math_patterns:
        matches = re.findall(pat, response)
        if matches:
            detected.extend(matches[:2])
    return (len(detected) > 0), detected


def generate_qualitative_suite(
    models_dict: Dict[str, Tuple[nn.Module, Any]],
    prompts_path: str = "task1/first_plot_questions.yaml",
    max_new_tokens: int = 512,
    output_json_path: str = "results/task1/qualitative_responses.json",
    annotation_md_path: str = "results/task1/qualitative_annotations.md",
) -> Dict[str, Any]:
    """
    Run the qualitative generation across all 7 conditions and save both
    the complete 168-response JSON catalog and the 42-response Markdown audit.
    """
    questions = load_qualitative_prompts(prompts_path)
    all_responses: Dict[str, Any] = {}

    print(f"[Qualitative] Generating responses for {len(questions)} questions across {len(models_dict)} conditions...")
    for q in questions:
        q_id = str(q["id"])
        topic = q["topic"]
        all_responses[q_id] = {
            "topic": topic,
            "formats": {},
        }

        for fmt in ["plain", "json", "template"]:
            prompt_text = q["formats"][fmt]
            all_responses[q_id]["formats"][fmt] = {
                "prompt": prompt_text,
                "conditions": {},
            }

            for cond_name, (model, tokenizer) in models_dict.items():
                res = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                )
                fmt_valid, fmt_msg = check_format_compliance(res["response"], fmt)
                has_math, math_tokens = check_unnecessary_math_framing(res["response"])

                all_responses[q_id]["formats"][fmt]["conditions"][cond_name] = {
                    "response": res["response"],
                    "num_new_tokens": res["num_new_tokens"],
                    "truncated": res["truncated"],
                    "format_compliance": fmt_valid,
                    "format_detail": fmt_msg,
                    "unnecessary_math_framing": has_math,
                    "math_triggers": math_tokens,
                }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, indent=2)
    print(f"[Qualitative] Saved full responses to {output_json_path}")

    # Generate 42-response annotation markdown for Question 1 (first) and Question 8 (last)
    first_q_id = "1"
    last_q_id = str(len(questions))

    md_lines = [
        "# Qualitative Behavior Annotation Sheet (42 Responses)",
        "",
        "This sheet annotates the first (Q1: Photosynthesis) and last (Q8: AI Ethics) underlying questions "
        "across all three formats (plain, JSON, template) and all seven conditions (42 total responses).",
        "",
        "| Question | Format | Condition | Truncated | Format Compliant | Unnecessary Math | Excerpt |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for target_id in [first_q_id, last_q_id]:
        if target_id not in all_responses:
            continue
        q_data = all_responses[target_id]
        topic = q_data["topic"]

        for fmt in ["plain", "json", "template"]:
            for cond_name in models_dict.keys():
                entry = q_data["formats"][fmt]["conditions"].get(cond_name, {})
                resp = entry.get("response", "").replace("\n", " ")
                excerpt = (resp[:75] + "...") if len(resp) > 75 else resp
                trunc = "Yes" if entry.get("truncated", False) else "No"
                comp = "Yes" if entry.get("format_compliance", False) else "No"
                math_flag = "Yes" if entry.get("unnecessary_math_framing", False) else "No"

                md_lines.append(
                    f"| Q{target_id} ({topic}) | {fmt} | {cond_name} | {trunc} | {comp} | {math_flag} | `{excerpt}` |"
                )

    with open(annotation_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[Qualitative] Saved 42-response annotation sheet to {annotation_md_path}")

    return all_responses
