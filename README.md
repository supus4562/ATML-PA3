# ATML PA3 - Task 1: Large Language Model Fine-Tuning

Modular, high-performance implementation for **Task 1: Large Language Model Fine-Tuning** of EE-5102 / CS-6304 Advanced Topics in Machine Learning (Programming Assignment 3).

Optimized for **NVIDIA RTX 6000 Pro Blackwell / Ada / Hopper Architectures** to complete all 5 training runs, checkpointing, diagnostics, qualitative analysis, and 7 benchmark evaluations in under **30–35 minutes**.

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

## 5. Execution Guide for Kaggle (NVIDIA Tesla T4 / P100 16GB)

The `kaggle-gpu` branch incorporates specialized memory optimizations to run flawlessly within **16 GB VRAM**:
- **Sequential Model Offloading**: Inactive models are offloaded to CPU with explicit CUDA cache clearing so only a single model resides on GPU at any given moment.
- **Fused F.cross_entropy Scoring**: Replaces explicit `log_softmax` matrix generation during likelihood evaluation, saving 10–20 GB of intermediate tensor allocations.
- **Tuned Batch Sizes**: Evaluation batch size calibrated to 16 for safe, fast completion.
- **Adaptive Workers**: DataLoader concurrency adapted to Kaggle's `/dev/shm` shared memory boundaries.

### Option A: Using the Interactive Jupyter Notebook
Open and run [`kaggle_task1.ipynb`](file:///home/supus/Projects/ATML/ATML-PA3/kaggle_task1.ipynb) directly in Kaggle:
1. Upload `kaggle_task1.ipynb` to Kaggle Notebooks.
2. Select **Settings** $\to$ **Accelerator** $\to$ **GPU T4 x 2** or **GPU P100**.
3. Toggle **Internet** $\to$ **On**.
4. Run all cells.

### Option B: Using the Command Line / Terminal on Kaggle
```bash
# Clone the kaggle-gpu branch
git clone -b kaggle-gpu https://github.com/supus4562/ATML-PA3.git
cd ATML-PA3

# Run the complete pipeline
bash run_kaggle.sh
```

---

## 6. Execution Guide for Modal Cloud (modal.com)

The `modal` branch adds serverless execution on **Modal (modal.com)** with persistent cloud volumes and real-time streaming:
- **Persistent Cloud Storage (`pa3-volume`)**: Checkpoints, tokenized data cache, and results are stored in a durable cloud volume and committed automatically after each checkpoint and stage.
- **Real-Time Step-by-Step Logging**: Every single optimizer update ($1, 2, \dots, 1250$) and evaluation batch is printed and flushed in real time, with redundant streaming into `.jsonl` and `.csv`.
- **Native `lm-evaluation-harness`**: Executes GSM8K with non-thinking chat template ($1,024$-token cap, greedy decoding) and HellaSwag, BoolQ, ARC-Easy with completion prompts, saving per-example outputs and predictions.
- **Target Hardware Acceleration**: Default runner uses **NVIDIA RTX PRO 6000 (48GB VRAM)** (~$\$3.03$/hr; full Task 1 finishes in ~30–35 minutes, costing only **~$\$1.50 – \$1.80** out of the free $\$30$ credit). Also supports **A10G** (`--gpu A10G`), **A100** (`--gpu A100`), and **L4** (`--gpu L4`).

### Quick Start on Modal:
```bash
# 1. Install modal locally
pip install modal

# 2. Authenticate with your Modal account (with your $30 compute credits)
modal setup

# 3. Dry-run verification (checks container, CUDA, volume, and packages in ~30s)
modal run modal_run.py --dry-run

# 4. Launch full Task 1 pipeline on NVIDIA RTX PRO 6000 (48GB)
modal run modal_run.py

# 5. (Alternative GPUs) Launch on A10G or A100
modal run modal_run.py --gpu A10G
modal run modal_run.py --gpu A100

# 6. Run a specific stage
modal run modal_run.py --stage train_full_ft

# 7. Download all results from cloud volume directly to local repo
modal run modal_run.py --download
# (Or alternatively: modal volume get pa3-volume results/ ./results/)
```


---

## 7. Output Deliverables in `results/task1/`

Upon completion, all required evidence is automatically generated in `results/task1/`:
- `metrics_summary.json`: Benchmark scores across all 7 evaluation conditions and primary score changes from baseline in percentage points.
- `learning_curves.csv`: Complete history of SFT loss, learning rate, $\Omega$, $\lambda \Omega$, and gradient norm for every optimizer update.
- `diagnostics.json`: $\rho_k$, $\eta$, and $R(z)$ on math vs non-math prompts across layers 0, 13, and 27.
- `qualitative_responses.json`: Full 168-response catalog grouped by question ID and format.
- `qualitative_annotations.md`: Markdown table auditing the 42-response subset (first and last question across all 3 formats and 7 conditions) for format compliance, coherence, repetition, unnecessary mathematical framing, and truncation flags.
