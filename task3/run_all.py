"""
Master orchestrator for Task 3: Test-Time Compute.
Coordinates Steps 1 to 6:
- Step 1: Repeated sampling opportunities and failure decomposition.
- Step 2: Quality scoring, VisConf calibration, and Rank-Weighted Consensus.
- Step 3: Within-question signal analysis and tokenwise fusion ablation.
- Step 4: Live generation & sequential refinement on 50 MATH evaluation questions.
- Step 5: Adaptive sampling threshold optimization and simulation.
- Step 6: Synthesis, bootstrap confidence intervals, and qualitative annotations.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional

import numpy as np
import torch

from task3.config import load_task3_config, Task3Config
from task3.data import (
    load_task3_dataset,
    partition_dataset_splits,
    RolloutItem,
)
from task3.selection import (
    generate_candidate_permutations,
    evaluate_question_budget,
    compute_paired_bootstrap_ci,
)
from task3.analysis import analyze_signals_at_n32
from task3.live_refinement import run_live_refinement_experiment
from task3.adaptive import (
    find_optimal_adaptive_thresholds,
    simulate_adaptive_sampling_policy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="ATML PA3 Task 3: Test-Time Compute Orchestrator")
    parser.add_argument("--config", type=str, default="configs/task3_config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        help="Stages to run: [all, step1_2, step3, step4_live, step5_adaptive, synthesize]",
    )
    parser.add_argument("--data-dir", type=str, default="", help="Override path to ATML-PA3-Data directory")
    parser.add_argument("--skip-live", action="store_true", help="Skip live GPU generation in Step 4")
    parser.add_argument("--single-gpu", action="store_true", help="Enforce strictly single GPU operation")
    return parser.parse_args()


def trigger_modal_stage_commit(stage_name: str) -> None:
    """Commit Modal cloud volume if running inside a Modal container."""
    try:
        import modal
        vol = modal.Volume.from_name("pa3-volume")
        vol.commit()
        print(f"[Modal Volume] Committed stage '{stage_name}' to cloud storage.", flush=True)
    except Exception:
        pass


def main():
    args = parse_args()
    config = load_task3_config(args.config)

    if args.data_dir:
        config.paths.data_dir = args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("ATML PA3 - TASK 3: TEST-TIME COMPUTE (SAMPLING, SELECTION & ADAPTIVE INFERENCE)")
    print("=" * 80)
    print(f"[Device] Using compute device: {device}")
    if device.type == "cuda":
        print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    os.makedirs(config.paths.output_dir, exist_ok=True)
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)

    stages = set(args.stages)
    run_all = "all" in stages

    # Check data directory
    data_dir = config.paths.data_dir
    if not os.path.exists(data_dir):
        # Check volume path fallback
        vol_path = "/root/volume/data/ATML-PA3-Data"
        if os.path.exists(vol_path):
            data_dir = vol_path
            config.paths.data_dir = vol_path

    print(f"[Config] Data directory: {data_dir}")
    print(f"[Config] Output directory: {config.paths.output_dir}")

    # 1. Load Datasets
    print("\n" + "=" * 80)
    print("STEP 0: LOADING DATASETS & PARTITIONING IMAGE-DISJOINT SPLITS (SEED 6304)")
    print("=" * 80)

    math_raw = load_task3_dataset(
        data_dir=data_dir,
        dataset_name=config.models["math_llm"].dataset,
        rollout_folder=config.models["math_llm"].rollout_folder,
    )
    mathvista_raw = load_task3_dataset(
        data_dir=data_dir,
        dataset_name=config.models["mathvista_vlm"].dataset,
        rollout_folder=config.models["mathvista_vlm"].rollout_folder,
    )

    # Split partitioning
    math_q_ids = list(math_raw.keys())
    m_dev_ids, m_eval_ids, m_live_ids = partition_dataset_splits(
        math_q_ids,
        seed=config.seed,
        dev_count=config.offline_eval.dev_split_size,
        eval_count=config.offline_eval.eval_split_size,
    )

    mv_q_ids = list(mathvista_raw.keys())
    mv_dev_ids, mv_eval_ids, _ = partition_dataset_splits(
        mv_q_ids,
        seed=config.seed,
        dev_count=config.offline_eval.dev_split_size,
        eval_count=config.offline_eval.eval_split_size,
    )

    print(f"[Splits] MATH: {len(m_dev_ids)} Dev questions, {len(m_eval_ids)} Eval questions, {len(m_live_ids)} Live questions.")
    print(f"[Splits] MathVista: {len(mv_dev_ids)} Dev questions, {len(mv_eval_ids)} Eval questions.")

    # Generate 10 seeded random candidate orders per question
    permutations = generate_candidate_permutations(
        num_rollouts=32,
        num_permutations=config.offline_eval.num_permutations,
        seed=config.seed,
    )

    # Prepare split mappings
    math_dev_pool = {qid: math_raw[qid] for qid in m_dev_ids}
    math_eval_pool = {qid: math_raw[qid] for qid in m_eval_ids}

    mv_dev_pool = {qid: mathvista_raw[qid] for qid in mv_dev_ids}
    mv_eval_pool = {qid: mathvista_raw[qid] for qid in mv_eval_ids}

    deliverables: Dict[str, Any] = {}

    # --------------------------------------------------------------------------
    # STEPS 1 & 2: REPEATED SAMPLING & METHOD COMPARISON MATRIX
    # --------------------------------------------------------------------------
    if run_all or "step1_2" in stages or "step1" in stages or "step2" in stages:
        print("\n" + "=" * 80)
        print("STEPS 1 & 2: REPEATED SAMPLING BUDGETS & METHOD COMPARISONS (N in {1, 2, 4, 8, 16, 32})")
        print("=" * 80)

        budgets = config.offline_eval.budgets
        math_rules = [
            "first_ans",
            "self_consistency",
            "highest_logp",
            "highest_self_certainty",
            "rwc_self_certainty",
        ]
        mv_rules = [
            "first_ans",
            "self_consistency",
            "highest_logp",
            "highest_self_certainty",
            "rwc_self_certainty",
            "highest_vision",
            "rwc_vision",
            "highest_visconf",
            "rwc_visconf",
            "highest_visconf_tokenwise",
            "rwc_visconf_tokenwise",
        ]

        offline_results = {"MATH": {}, "MathVista": {}}

        # Evaluate MATH on 800 evaluation questions
        print(f"[Offline Evaluation] Evaluating MATH ({len(math_eval_pool)} questions) across budgets...")
        for b in budgets:
            q_res_list = []
            for qid, rollouts in math_eval_pool.items():
                q_res = evaluate_question_budget(
                    rollouts=rollouts,
                    budget=b,
                    permutations=permutations,
                    rule_names=math_rules,
                    dataset_name="MATH",
                )
                q_res_list.append(q_res)

            # Average across questions
            budget_summary = {}
            for r in math_rules:
                accs = [q[r]["accuracy"] for q in q_res_list]
                costs = [q[r]["token_cost"] for q in q_res_list]
                oracles = [q[r]["oracle_coverage"] for q in q_res_list]
                dgroups = [q[r]["distinct_groups"] for q in q_res_list]
                samp_fails = [q[r]["sampling_failure"] for q in q_res_list]
                selec_fails = [q[r]["selection_failure"] for q in q_res_list]

                budget_summary[r] = {
                    "accuracy": float(np.mean(accs)) * 100.0,
                    "token_cost": float(np.mean(costs)),
                    "oracle_coverage": float(np.mean(oracles)) * 100.0,
                    "distinct_groups": float(np.mean(dgroups)),
                    "sampling_failure": float(np.mean(samp_fails)) * 100.0,
                    "selection_failure": float(np.mean(selec_fails)) * 100.0,
                }
            offline_results["MATH"][str(b)] = budget_summary
            sc_acc = budget_summary["self_consistency"]["accuracy"]
            or_cov = budget_summary["self_consistency"]["oracle_coverage"]
            print(f"  MATH N={b:2d} | SC Acc: {sc_acc:5.2f}% | Oracle: {or_cov:5.2f}% | Tokens: {budget_summary['self_consistency']['token_cost']:.1f}")

        # Evaluate MathVista on 800 evaluation questions
        print(f"\n[Offline Evaluation] Evaluating MathVista ({len(mv_eval_pool)} questions) across budgets...")
        for b in budgets:
            q_res_list = []
            for qid, rollouts in mv_eval_pool.items():
                q_res = evaluate_question_budget(
                    rollouts=rollouts,
                    budget=b,
                    permutations=permutations,
                    rule_names=mv_rules,
                    dataset_name="MathVista",
                )
                q_res_list.append(q_res)

            budget_summary = {}
            for r in mv_rules:
                accs = [q[r]["accuracy"] for q in q_res_list]
                costs = [q[r]["token_cost"] for q in q_res_list]
                oracles = [q[r]["oracle_coverage"] for q in q_res_list]
                dgroups = [q[r]["distinct_groups"] for q in q_res_list]
                samp_fails = [q[r]["sampling_failure"] for q in q_res_list]
                selec_fails = [q[r]["selection_failure"] for q in q_res_list]

                budget_summary[r] = {
                    "accuracy": float(np.mean(accs)) * 100.0,
                    "token_cost": float(np.mean(costs)),
                    "oracle_coverage": float(np.mean(oracles)) * 100.0,
                    "distinct_groups": float(np.mean(dgroups)),
                    "sampling_failure": float(np.mean(samp_fails)) * 100.0,
                    "selection_failure": float(np.mean(selec_fails)) * 100.0,
                }
            offline_results["MathVista"][str(b)] = budget_summary
            sc_acc = budget_summary["self_consistency"]["accuracy"]
            vc_acc = budget_summary["rwc_visconf"]["accuracy"]
            or_cov = budget_summary["self_consistency"]["oracle_coverage"]
            print(f"  MathVista N={b:2d} | SC Acc: {sc_acc:5.2f}% | RWC-VisConf: {vc_acc:5.2f}% | Oracle: {or_cov:5.2f}%")

        # 1,000 paired question-level bootstrap confidence intervals at N = 32
        print("\n[Bootstrap] Computing 1,000 paired question-level bootstrap CIs on main differences at N=32...")
        ci_summary = {}
        for dset, q_pool, rules in [("MATH", math_eval_pool, math_rules), ("MathVista", mv_eval_pool, mv_rules)]:
            sc_accs = []
            first_accs = []
            rwc_accs = []
            target_rwc = "rwc_self_certainty" if dset == "MATH" else "rwc_visconf"

            for qid, rollouts in q_pool.items():
                q_res = evaluate_question_budget(rollouts, 32, permutations, rules, dset)
                sc_accs.append(q_res["self_consistency"]["accuracy"])
                first_accs.append(q_res["first_ans"]["accuracy"])
                rwc_accs.append(q_res[target_rwc]["accuracy"])

            d_sc_first = [s - f for s, f in zip(sc_accs, first_accs)]
            d_rwc_sc = [r - s for r, s in zip(rwc_accs, sc_accs)]

            m1, lo1, hi1 = compute_paired_bootstrap_ci(d_sc_first, num_resamples=1000, seed=config.seed)
            m2, lo2, hi2 = compute_paired_bootstrap_ci(d_rwc_sc, num_resamples=1000, seed=config.seed + 1)

            ci_summary[dset] = {
                "sc_minus_firstans_pp": m1 * 100.0,
                "sc_minus_firstans_ci": [lo1 * 100.0, hi1 * 100.0],
                f"{target_rwc}_minus_sc_pp": m2 * 100.0,
                f"{target_rwc}_minus_sc_ci": [lo2 * 100.0, hi2 * 100.0],
            }

        deliverables["repeated_sampling"] = offline_results
        deliverables["bootstrap_ci_n32"] = ci_summary

        save_p = os.path.join(config.paths.output_dir, "repeated_sampling_results.json")
        with open(save_p, "w", encoding="utf-8") as f:
            json.dump({"results": offline_results, "bootstrap_ci": ci_summary}, f, indent=2)
        print(f"[Deliverable] Saved repeated sampling results to {save_p}")
        trigger_modal_stage_commit("step1_2")

    # --------------------------------------------------------------------------
    # STEP 3: SIGNAL ANALYSIS AT N = 32
    # --------------------------------------------------------------------------
    if run_all or "step3" in stages:
        print("\n" + "=" * 80)
        print("STEP 3: INVESTIGATING WHAT THE SIGNALS CAPTURE AT N = 32")
        print("=" * 80)

        math_signals = analyze_signals_at_n32(math_eval_pool, "MATH")
        mv_signals = analyze_signals_at_n32(mv_eval_pool, "MathVista")

        signal_analysis_results = {
            "MATH": math_signals,
            "MathVista": mv_signals,
        }
        deliverables["signal_analysis"] = signal_analysis_results

        p_sig = os.path.join(config.paths.output_dir, "signal_analysis.json")
        with open(p_sig, "w", encoding="utf-8") as f:
            json.dump(signal_analysis_results, f, indent=2)
        print(f"[Deliverable] Saved signal analysis results to {p_sig}")
        trigger_modal_stage_commit("step3")

    # --------------------------------------------------------------------------
    # STEP 4: LIVE GENERATION & SEQUENTIAL REFINEMENT (50 MATH QUESTIONS)
    # --------------------------------------------------------------------------
    if (run_all or "step4" in stages or "step4_live" in stages) and not args.skip_live:
        print("\n" + "=" * 80)
        print("STEP 4: LIVE GENERATION VS SEQUENTIAL REFINEMENT (50 MATH QUESTIONS)")
        print("=" * 80)

        live_questions = []
        for qid in m_live_ids:
            rollouts = math_raw[qid]
            live_questions.append({
                "sample_id": qid,
                "question": rollouts[0].question_text,
                "gold_answer": rollouts[0].gold_answer,
            })

        live_results = run_live_refinement_experiment(
            questions=live_questions,
            model_name=config.live_generation.model_name,
            device=device,
            output_dir=config.paths.output_dir,
            seed=config.seed,
        )
        deliverables["live_refinement"] = live_results
        trigger_modal_stage_commit("step4")

    # --------------------------------------------------------------------------
    # STEP 5: ADAPTIVE SAMPLING
    # --------------------------------------------------------------------------
    if run_all or "step5" in stages or "step5_adaptive" in stages:
        print("\n" + "=" * 80)
        print("STEP 5: ADAPTIVE SAMPLING CONTROLLERS & HELD-OUT SIMULATION")
        print("=" * 80)

        # 1. Dev set optimization
        print("[Adaptive Sampling] Tuning stopping thresholds on Development splits (200 questions)...")
        math_opt = find_optimal_adaptive_thresholds(
            dev_rollouts=math_dev_pool,
            permutations=permutations,
            dataset_name="MATH",
            tau_grid=config.adaptive_sampling.agreement_thresholds,
            gamma_grid=config.adaptive_sampling.quality_margins,
            max_accuracy_drop_pp=config.adaptive_sampling.dev_accuracy_margin_pp,
        )
        mv_opt = find_optimal_adaptive_thresholds(
            dev_rollouts=mv_dev_pool,
            permutations=permutations,
            dataset_name="MathVista",
            tau_grid=config.adaptive_sampling.agreement_thresholds,
            gamma_grid=config.adaptive_sampling.quality_margins,
            max_accuracy_drop_pp=config.adaptive_sampling.dev_accuracy_margin_pp,
        )

        print(f"[Adaptive Thresholds] MATH: Agreement tau={math_opt['best_agreement_thresholds']['tau']} | Quality tau={math_opt['best_quality_thresholds']['tau']}, gamma={math_opt['best_quality_thresholds']['gamma']}")
        print(f"[Adaptive Thresholds] MathVista: Agreement tau={mv_opt['best_agreement_thresholds']['tau']} | Quality tau={mv_opt['best_quality_thresholds']['tau']}, gamma={mv_opt['best_quality_thresholds']['gamma']}")

        # 2. Held-out evaluation
        print("\n[Adaptive Sampling] Running held-out evaluation on 800 Evaluation questions...")
        adaptive_eval = {"MATH": {}, "MathVista": {}}

        # MATH held-out
        m_agr_tau = math_opt["best_agreement_thresholds"]["tau"]
        m_qua_tau = math_opt["best_quality_thresholds"]["tau"]
        m_qua_gamma = math_opt["best_quality_thresholds"]["gamma"]

        adaptive_eval["MATH"]["agreement_based"] = simulate_adaptive_sampling_policy(
            math_eval_pool, permutations, "agreement_based", m_agr_tau, 0.0, checkpoints=config.adaptive_sampling.checkpoints, dataset_name="MATH"
        )
        adaptive_eval["MATH"]["quality_aware"] = simulate_adaptive_sampling_policy(
            math_eval_pool, permutations, "quality_aware", m_qua_tau, m_qua_gamma, checkpoints=config.adaptive_sampling.checkpoints, dataset_name="MATH"
        )

        # MathVista held-out
        mv_agr_tau = mv_opt["best_agreement_thresholds"]["tau"]
        mv_qua_tau = mv_opt["best_quality_thresholds"]["tau"]
        mv_qua_gamma = mv_opt["best_quality_thresholds"]["gamma"]

        adaptive_eval["MathVista"]["agreement_based"] = simulate_adaptive_sampling_policy(
            mv_eval_pool, permutations, "agreement_based", mv_agr_tau, 0.0, checkpoints=config.adaptive_sampling.checkpoints, dataset_name="MathVista"
        )
        adaptive_eval["MathVista"]["quality_aware"] = simulate_adaptive_sampling_policy(
            mv_eval_pool, permutations, "quality_aware", mv_qua_tau, mv_qua_gamma, checkpoints=config.adaptive_sampling.checkpoints, dataset_name="MathVista"
        )

        adaptive_summary = {
            "dev_threshold_selection": {
                "MATH": math_opt,
                "MathVista": mv_opt,
            },
            "eval_simulation": adaptive_eval,
        }
        deliverables["adaptive_sampling"] = adaptive_summary

        p_ad = os.path.join(config.paths.output_dir, "adaptive_sampling_results.json")
        with open(p_ad, "w", encoding="utf-8") as f:
            json.dump(adaptive_summary, f, indent=2)
        print(f"[Deliverable] Saved adaptive sampling results to {p_ad}")
        trigger_modal_stage_commit("step5")

    # --------------------------------------------------------------------------
    # STEP 6: SYNTHESIS & QUALITATIVE REPORT
    # --------------------------------------------------------------------------
    if run_all or "synthesize" in stages:
        print("\n" + "=" * 80)
        print("STEP 6: REPORT SYNTHESIS & QUALITATIVE ANNOTATIONS")
        print("=" * 80)

        # Qualitative annotations table
        p_anno = os.path.join(config.paths.output_dir, "qualitative_annotations.md")
        with open(p_anno, "w", encoding="utf-8") as f:
            f.write("# Task 3: Qualitative Test-Time Compute & Adaptive Sampling Annotations\n\n")
            f.write("Representative examples illustrating selection disagreement, sequential refinement transitions, and adaptive stopping.\n\n")
            f.write("## 1. Live Refinement Transitions (MATH Subset)\n\n")
            f.write("| Sample ID | Initial Answer | Revised Answer | Transition Type |\n")
            f.write("|---|---|---|---|\n")

            live_res = deliverables.get("live_refinement", {})
            if live_res and "sequential_refinement" in live_res:
                for item in live_res["sequential_refinement"].get("per_question", [])[:15]:
                    s_id = item["sample_id"]
                    i_ans = item["initial_answer"]
                    r_ans = item["revised_answer"]
                    i_c = item["initial_correct"]
                    r_c = item["revised_correct"]
                    t_type = "Wrong -> Correct" if not i_c and r_c else ("Correct -> Wrong" if i_c and not r_c else ("Retained Correct" if i_c else "Retained Wrong"))
                    f.write(f"| {s_id} | `{i_ans}` | `{r_ans}` | {t_type} |\n")

            f.write("\n## 2. Selection Rule Disagreements (MathVista Subset)\n\n")
            f.write("| Question ID | FirstAns | Majority SC | Highest VisConf | RWC VisConf | Gold Answer |\n")
            f.write("|---|---|---|---|---|---|\n")

            # Sample 10 MathVista evaluation questions
            sample_mv_ids = mv_eval_ids[:10]
            for qid in sample_mv_ids:
                rollouts = mv_eval_pool[qid]
                cands = [rollouts[i] for i in permutations[0][:32]]
                metrics_dict = compute_visconf_scores(cands)

                fa, _, _ = select_answer(cands, "first_ans", "MathVista", metrics_dict)
                sc, _, _ = select_answer(cands, "self_consistency", "MathVista", metrics_dict)
                h_vc, _, _ = select_answer(cands, "highest_visconf", "MathVista", metrics_dict)
                r_vc, _, _ = select_answer(cands, "rwc_visconf", "MathVista", metrics_dict)
                gold = rollouts[0].gold_answer

                f.write(f"| {qid} | `{fa}` | `{sc}` | `{h_vc}` | `{r_vc}` | `{gold}` |\n")

        print(f"[Synthesis] Saved qualitative annotations table to {p_anno}")

        # Metrics summary
        p_sum = os.path.join(config.paths.output_dir, "metrics_summary.json")
        with open(p_sum, "w", encoding="utf-8") as f:
            json.dump(deliverables, f, indent=2)
        print(f"[Synthesis] Saved consolidated metrics summary to {p_sum}")
        trigger_modal_stage_commit("synthesize")

    print("\n" + "=" * 80)
    print("TASK 3 EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
