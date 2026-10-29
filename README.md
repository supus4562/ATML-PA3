# ATML PA3 - Deep Learning Pipelines: Task 1 & Task 2

Modular, high-performance implementation for **Programming Assignment 3** of EE-5102 / CS-6304 Advanced Topics in Machine Learning:
- **Task 1**: Large Language Model Fine-Tuning (Full FT, L2-SP, LoRA-4, LoRA-16, OPLoRA-16)
- **Task 2**: Vision-Language Models and Visual Grounding (SigLIP-224, 2-Layer MLP Connector, Qwen3-0.6B LoRA, 3-Stage Curriculum, 4-Condition Evaluation)

Optimized for **NVIDIA A100 / RTX 6000 Ada / Hopper Architectures** with FP16 AMP, flash attention, dynamic batching, and serverless Modal cloud execution.

---

## 1. Directory Structure

```text
ATML-PA3/
├── configs/
│   └── task1_config.yaml         # Central configuration (hyperparameters, seeds, paths)
├── task1/
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Dataclasses & YAML schema loader
│   ├── utils.py                  # Seeding (6304), VRAM tracking, parameter displacement, timers
│   ├── data.py                   # MetaMathQA filtering, GSM8K leak detection, dynamic collator
│   ├── models.py                 # Qwen3-0.6B loader, LoRALinear, OPLoRALinear, GPU SVD, weight merger
│   ├── trainer.py                # Unified trainer: FP16 AMP, L2-SP penalty, optimizer, grad clipping
│   ├── evaluator.py              # lm-eval wrapper & standalone fast evaluator for GSM8K, HellaSwag, BoolQ, ARC
│   ├── diagnostics.py            # Weight-space (rho_k, eta) and Activation probes R(z) on layers 0, 13, 27
│   ├── qualitative.py            # 24-prompt qualitative generation (168 responses) & 42-response audit
│   ├── run_all.py                # Master orchestrator script
│   └── first_plot_questions.yaml # 24 prompts (8 questions x 3 formats: plain, json, template)
├── run_task1.sh                  # One-click execution launcher
├── requirements.txt              # Environment dependencies
└── results/task1/                # Auto-generated results & evidence
    ├── metrics_summary.json      # Consolidated benchmark scores & baseline percentage-point deltas
    ├── learning_curves.csv       # Training loss, LR, and L2-SP penalties vs update
    ├── diagnostics.json          # rho_k, eta, and R(z) for math vs non-math across layers 0, 13, 27
    ├── qualitative_responses.json# All 168 responses grouped by question and format
    ├── qualitative_annotations.md# Fixed 42-response annotation sheet
    └── gsm8k_details_*.json      # Per-example extracted GSM8K answers and truncation flags
```

---

## 2. Invariants Checklist & Exact Formulations

The codebase strictly enforces all assignment invariants:

| Category | Invariant Rule | Implementation Detail |
| :--- | :--- | :--- |
| **Model** | `Qwen/Qwen3-0.6B` | Post-trained instruction checkpoint; thinking mode explicitly disabled (`enable_thinking=False`). |
| **Reproducibility** | Seed `6304` | Fixed across Python, NumPy, PyTorch, CUDA, probe sampling, and qualitative tests. |
| **Data Selection** | Source Order (40k) | First 40,000 eligible MetaMathQA examples in strict source order. **No shuffling, no truncation**. |
| **Leakage Check** | GSM8K Leak Filter | Strips punctuation, lowercases, and removes any sample matching GSM8K test questions. |
| **Loss Masking** | Assistant Tokens Only | Next-token cross-entropy computed **only** on assistant tokens and end-of-turn token. Prompts & pads masked with `-100`. |
| **Loss Normalization** | Effective Batch Normalization | Accumulated loss normalized across all valid response tokens in the effective batch (32 samples). |
| **Budget** | 1 Epoch / 1,250 Updates | Microbatch 2, gradient accumulation 16 $\implies$ effective batch 32. 40,000 / 32 = 1,250 updates. |
| **Optimizer** | AdamW & Decay Exclusion | $\beta=(0.9, 0.999), \epsilon=10^{-8}, \text{wd}=0.01$. **Excludes biases and 1D parameters** (e.g. RMSNorm). |
| **Schedule** | Cosine with Warmup | Cosine decay with 38 warmup updates; gradient clipping at global norm 1.0. |
| **Full FT** | Free adaptation | Peak LR $2 \times 10^{-5}, \lambda = 0$. Record displacement $\|\theta - \theta_0\|_2$ at midpoint (625) and final (1250). |
| **Full FT + L2-SP** | Starting-point penalty | $\Omega(\theta, \theta_0) = \frac{1}{2} \sum (\theta_j - \theta_{0,j})^2, \lambda = 10^{-3}$. Sum over all unique parameters. |
| **LoRA-4 & 16** | Low-rank adaptation | $(r, \alpha) = (4, 8)$ and $(16, 32)$, peak LR $2 \times 10^{-4}$, dropout 0.05. Targets all 7 projections per layer. |
| **OPLoRA-16** | Orthogonal projection | $r=16, \alpha=32, k=128$. Leading singular bases $U_k, V_k$ computed in FP32 on GPU. $\tilde{B} = B - U_k(U_k^\top B), \tilde{A} = A - (AV_k)V_k^\top$. |
| **Checkpoints** | Recovery & Evaluation | Checkpoints at updates 312, 625, 937, 1250. 7 evaluation conditions (baseline, midpoint FT, 5 finals). |
| **Benchmarks** | Zero-shot evaluation | GSM8K (exact match, greedy, 1024 cap), HellaSwag (`acc_norm`), BoolQ (`acc`), ARC-Easy (`acc_norm`). |
| **Diagnostics** | Layers 0, 13, 27 | Subspace interference $\rho_k = \frac{\|U_k U_k^\top \Delta W\|_F^2}{\|\Delta W\|_F^2}$, relative magnitude $\eta$, and activation ratio $R(z)$. |
| **Qualitative** | 24 Prompts / 168 Outputs | 8 questions $\times$ 3 formats (plain, JSON, template). Fixed 42-response annotation sheet comparing compliance and unnecessary math. |

---

## 3. Optimizations for NVIDIA RTX 6000 Pro Blackwell

The implementation leverages the following high-throughput techniques:

1. **PyTorch SDPA & Blackwell Tensor Cores**: Dispatches to native FlashAttention-2 kernels for $O(L)$ memory and zero attention matrix materialization.
2. **Dynamic Microbatch Padding**: Rather than static-padding to 2,048 tokens (which wastes 85% of FLOPs on pad tokens), each microbatch dynamically pads only to its local maximum length. This speeds up training by **~4.5x** while preserving exact sequence contents and loss values.
3. **Pre-tokenized Arrow Memory Mapping**: Filters and tokenizes the 40k dataset once, caching to disk so all 5 training runs load instantly without tokenization overhead.
4. **Fused AdamW (`fused=True`)**: Coalesces parameter updates into single GPU kernel calls.
5. **GPU-Accelerated Truncated SVD**: Computes $W_0 = U \Sigma V^\top$ in FP32 on CUDA in under 2 seconds for all 196 weight matrices.
6. **Effective-Factor OPLoRA**: Formulates projection using $(A V_k) V_k^\top$ and $U_k (U_k^\top B)$ without constructing dense $d_{out} \times d_{in}$ projector matrices.
7. **Pre-Evaluation Adapter Merging**: Merges adapter weights into fresh backbone copies prior to evaluation to eliminate adapter dispatch overhead during generation.

---

## 4. Execution Guide (Remote Server with RTX 6000 Pro Blackwell)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run End-to-End Pipeline
Execute the full pipeline (data filtering, baseline eval, 5 training runs, diagnostics, qualitative analysis, and synthesis):
```bash
bash run_task1.sh
```

Or invoke the master Python module directly:
```bash
python3 -m task1.run_all --config configs/task1_config.yaml
```

### Step 3: Running Individual Stages (Optional)
You can run specific stages using the `--stages` argument:
```bash
# Run only baseline evaluation
python3 -m task1.run_all --stages baseline

# Run only LoRA and OPLoRA training
python3 -m task1.run_all --stages train_lora16 train_oplora

# Run only diagnostics and qualitative evaluation
python3 -m task1.run_all --stages diagnostics qualitative synthesize
```

Available stages:
`[all, baseline, train_full_ft, train_l2sp, train_lora4, train_lora16, train_oplora, diagnostics, qualitative, synthesize]`

---

## 5. Output Deliverables in `results/task1/`

Upon completion, all required evidence is automatically generated in `results/task1/`:
- `metrics_summary.json`: Benchmark scores across all 7 evaluation conditions and primary score changes from baseline in percentage points.
- `learning_curves.csv`: Complete history of SFT loss, learning rate, $\Omega$, $\lambda \Omega$, and gradient norm for every optimizer update.
- `diagnostics.json`: $\rho_k$, $\eta$, and $R(z)$ on math vs non-math prompts across layers 0, 13, and 27.
- `qualitative_responses.json`: Full 168-response catalog grouped by question ID and format.
- `qualitative_annotations.md`: Markdown table auditing the 42-response subset (first and last question across all 3 formats and 7 conditions) for format compliance, coherence, repetition, unnecessary mathematical framing, and truncation flags.

---

## 6. Task 2: Vision-Language Models and Visual Grounding

High-performance implementation of multimodal instruction tuning and visual grounding evaluation.

### Model Architecture & Key Invariants
- **Vision Backbone**: `google/siglip-base-patch16-224` (always frozen). Preprocesses images to 196 patch features in $14 \times 14$ spatial order with final LayerNorm. Cached in FP16 to disk; encoder is completely unloaded from GPU memory before Qwen initialization to save VRAM.
- **Visual Connector**: 2-layer MLP (`Linear(768, 1024) -> GELU -> Linear(1024, 1024)`), trainable across all 3 stages. Evaluates initial vs final activation magnitudes against text token embeddings.
- **Language Model**: `Qwen/Qwen3-0.6B` (thinking disabled). Original weights remain frozen throughout. LoRA ($r=16, \alpha=32$, dropout $0.05$) applied to projections `[q_proj, k_proj, v_proj, o_proj]` during Stages 2 and 3.

### 3-Stage Curriculum Training (4,500 Updates = 72,000 Examples)
- **Stage 1 (Caption Alignment)**:
  - 2,250 updates (2 full passes over 18,000 COCO captions)
  - Connector trained at peak LR $10^{-3}$; LLM entirely frozen.
- **Stage 2 (Visual Instruction Tuning)**:
  - 1,500 updates (24,000 examples)
  - Connector peak LR $10^{-4}$, LoRA peak LR $2 \times 10^{-5}$
  - Balanced mixture: 50% presence grounding, 25% counting grounding, 25% caption replay.
- **Stage 3 (Natural VQA Specialization)**:
  - 750 updates (12,000 examples)
  - Connector peak LR $5 \times 10^{-5}$, LoRA peak LR $10^{-5}$
  - Domain mixture: 70% VQAv2, 10% presence, 10% counting, 10% caption replay.
- **Loss Invariant**: Example-weighted causal cross-entropy (averaged over response tokens per example, then across effective batch 16 = microbatch 8 $\times$ accumulation 2).

### Evaluation Protocol
Evaluates on 3 benchmarks (Presence Grounding, Counting Grounding, Natural VQAv2) across **4 image conditions**:
1. **Correct**: Original image paired with question.
2. **Complementary Swap**: Image swapped with semantic counter-example.
3. **Mismatched**: Unrelated random image.
4. **No-Image Ablation**: Zeroed visual tokens (pure text prior test).

Metrics reported:
- Exact Match / VQA Soft Consensus
- Complete Pair Accuracy
- Identical-Answer Rate & Empty-Answer Rate
- Mean Reference NLL
- Correct-Image Advantage: $\Delta\text{Acc} = \text{Acc}_{\text{correct}} - \text{Acc}_{\text{swap}}$, $\Delta\text{NLL} = \text{NLL}_{\text{swap}} - \text{NLL}_{\text{correct}}$
- 95% Confidence Intervals via 4,000 bootstrap resamples over complete question pairs.

---

## 7. Running Task 2

### Workstation / A100 Server Execution
```bash
# Run full Task 2 end-to-end pipeline:
bash run_task2.sh

# Or invoke with custom stages:
python3 -m task2.run_all --stages stage1 stage2 stage3 evaluate synthesize
```

### Available Stages for Task 2
`[all, prepare_data, cache_features, stage1, stage2, stage3, evaluate, synthesize]`

### Output Deliverables in `results/task2/`
- `stage_metrics.json`: Final training metrics across all stages.
- `training_curves.csv`: Per-update step loss, learning rates, gradient norms, and visual-text magnitude ratios.
- `evaluation_summary.json`: Detailed 4-condition scores, pair accuracies, and 95% bootstrap confidence intervals.
- `eval_table_checkpoint_*.json`: Intermediate dev checkpoints at 500, 1000, 1500, etc.
- `task2_final_report.md`: Markdown summary tables.

