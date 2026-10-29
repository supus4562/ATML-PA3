"""
Modal cloud execution orchestrator for ATML PA3 (Task 1 & Task 2).
Leverages Modal (modal.com) serverless GPU infrastructure with persistent volume storage.

Features:
- Persistent Volume ("pa3-volume") mounted at /root/volume for 100% durable checkpoints, cache & results.
- Task 1: LLM Fine-Tuning (Full FT, L2-SP, LoRA-4, LoRA-16, OPLoRA-16, GSM8K, HellaSwag, BoolQ, ARC-Easy).
- Task 2: VLM Visual Grounding (SigLIP-224, 2-Layer MLP Connector, Qwen3-0.6B LoRA, 3-Stage Curriculum, 4 Conditions).
- Unbuffered real-time stdout streaming of every single training update and evaluation step.
- Redundant multi-format persistence: JSON, JSONL, CSV, and automatic volume commits.
- Optimized for NVIDIA A100 (high memory bandwidth & throughput) and single RTX 6000 Ada.
- Built-in download entrypoint to sync all results to local workspace.

Usage:
  # 1. Verification / Dry-run
  modal run modal_run.py --dry-run

  # 2. Run complete Task 2 pipeline on NVIDIA A100 (Default)
  modal run modal_run.py --task task2 --gpu A100

  # 3. Run specific stages of Task 2 (e.g. stage1, stage2, stage3, evaluate)
  modal run modal_run.py --task task2 --stage "stage1 stage2 stage3 evaluate"

  # 4. Run Task 1 on NVIDIA RTX 6000 Ada
  modal run modal_run.py --task task1 --gpu RTX-PRO-6000

  # 5. Download all results from cloud volume to local repository
  modal run modal_run.py --download
"""

import os
import sys
import time
import modal

# 1. Define the Modal App
app = modal.App("atml-pa3")

# 2. Persistent Volume: durable cloud storage across runs and restarts
volume = modal.Volume.from_name("pa3-volume", create_if_missing=True)
VOLUME_MOUNT_PATH = "/root/volume"

# 3. Container Image Definition with CUDA and all required libraries
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "procps")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "pillow>=10.0.0",
        "transformers>=4.45.0",
        "tokenizers>=0.20.0",
        "datasets>=2.20.0",
        "accelerate>=0.34.0",
        "peft>=0.12.0",
        "lm-eval>=0.4.3",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "safetensors>=0.4.0",
        "scikit-learn>=1.4.0",
        "sentencepiece>=0.2.0",
    )
    .add_local_dir(
        ".",
        remote_path="/root/ATML-PA3",
        ignore=[
            "*.pt",
            "*.bin",
            "*.safetensors",
            ".git",
            "checkpoints",
            "data_cache",
            "results",
            "**/__pycache__",
            "*.pyc",
        ],
    )
)


def _execute_pipeline_in_container(
    task: str,
    stage: str,
    config_path: str,
    skip_eval: bool,
    gpu_label: str,
):
    """Core container execution function with unbuffered real-time stdout streaming."""
    import os
    import sys
    import subprocess
    import time
    import torch

    os.chdir("/root/ATML-PA3")
    sys.path.insert(0, "/root/ATML-PA3")

    # Set environment variables for unbuffered output and volume commit hooks
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["MODAL_VOLUME_COMMIT"] = "1"
    os.environ["MODAL_VOLUME_NAME"] = "pa3-volume"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    task_normalized = task.lower().strip()
    print("\n" + "=" * 80, flush=True)
    print(f"MODAL CLOUD PIPELINE: {task_normalized.upper()} on {gpu_label}", flush=True)
    print(f"Target Stage(s): {stage} | Config: {config_path}", flush=True)
    print("=" * 80, flush=True)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Hardware] GPU Active: {gpu_name} ({vram_gb:.2f} GB VRAM)", flush=True)

    # Ensure volume directory structures exist
    if task_normalized == "task1":
        os.makedirs(f"{VOLUME_MOUNT_PATH}/data_cache/task1", exist_ok=True)
        os.makedirs(f"{VOLUME_MOUNT_PATH}/checkpoints/task1", exist_ok=True)
        os.makedirs(f"{VOLUME_MOUNT_PATH}/results/task1", exist_ok=True)
        module_name = "task1.run_all"
    elif task_normalized == "task2":
        os.makedirs(f"{VOLUME_MOUNT_PATH}/data_cache/task2", exist_ok=True)
        os.makedirs(f"{VOLUME_MOUNT_PATH}/data_cache/task2/siglip_features", exist_ok=True)
        os.makedirs(f"{VOLUME_MOUNT_PATH}/checkpoints/task2", exist_ok=True)
        os.makedirs(f"{VOLUME_MOUNT_PATH}/results/task2", exist_ok=True)
        module_name = "task2.run_all"
    elif task_normalized == "task3":
        os.makedirs(f"{VOLUME_MOUNT_PATH}/data/ATML-PA3-Data", exist_ok=True)
        os.makedirs(f"{VOLUME_MOUNT_PATH}/checkpoints/task3", exist_ok=True)
        os.makedirs(f"{VOLUME_MOUNT_PATH}/results/task3", exist_ok=True)
        module_name = "task3.run_all"
    else:
        raise ValueError(f"Unknown task: '{task}'")

    volume.commit()

    stage_tokens = stage.replace(",", " ").split() if isinstance(stage, str) else list(stage)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        module_name,
        "--config",
        config_path,
        "--stages",
        *stage_tokens,
    ]
    if task_normalized == "task1":
        cmd.append("--single-gpu")
        if skip_eval:
            cmd.append("--skip-eval")

    print(f"[Command] {' '.join(cmd)}\n", flush=True)

    t_start = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream every single output line directly to Modal logs
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()

    process.stdout.close()
    return_code = process.wait()

    elapsed = time.time() - t_start
    print(
        f"\n[Modal Pipeline] Completed in {elapsed:.1f}s ({elapsed/60.0:.2f} min) with exit code {return_code}",
        flush=True,
    )

    # Final volume commit to ensure durability
    volume.commit()
    print("[Modal Pipeline] Durable cloud volume 'pa3-volume' successfully committed.", flush=True)

    if return_code != 0:
        raise RuntimeError(f"{task_normalized.upper()} execution failed with exit code {return_code}")


# ==============================================================================
# Cloud Functions for Task 2 (VLM & Grounding) - Single GPU Invariant Enforced
# ==============================================================================
@app.function(
    image=image,
    gpu="A100",
    timeout=86400,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_task2_a100(
    stage: str = "all",
    config_path: str = "configs/modal_task2_config.yaml",
    skip_eval: bool = False,
):
    """Run Task 2 on NVIDIA A100 (40GB/80GB) - optimized throughput."""
    _execute_pipeline_in_container("task2", stage, config_path, skip_eval, "NVIDIA A100")


@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=86400,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_task2_rtx6000(
    stage: str = "all",
    config_path: str = "configs/modal_task2_config.yaml",
    skip_eval: bool = False,
):
    """Run Task 2 on Single NVIDIA RTX PRO 6000 Ada (48GB VRAM)."""
    _execute_pipeline_in_container("task2", stage, config_path, skip_eval, "Single NVIDIA RTX PRO 6000 (48GB)")


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_task2_h100(
    stage: str = "all",
    config_path: str = "configs/modal_task2_config.yaml",
    skip_eval: bool = False,
):
    """Run Task 2 on NVIDIA H100 (80GB VRAM)."""
    _execute_pipeline_in_container("task2", stage, config_path, skip_eval, "NVIDIA H100 (80GB)")


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_task2_a10g(
    stage: str = "all",
    config_path: str = "configs/modal_task2_config.yaml",
    skip_eval: bool = False,
):
    """Run Task 2 on NVIDIA A10G (24GB VRAM)."""
    _execute_pipeline_in_container("task2", stage, config_path, skip_eval, "NVIDIA A10G (24GB)")


# ==============================================================================
# Cloud Functions for Task 3 (Test-Time Compute & Adaptive Sampling)
# ==============================================================================
@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_task3_h100(
    stage: str = "all",
    config_path: str = "configs/modal_task3_config.yaml",
    skip_eval: bool = False,
):
    """Run Task 3 on NVIDIA H100 (80GB VRAM)."""
    _execute_pipeline_in_container("task3", stage, config_path, skip_eval, "NVIDIA H100 (80GB)")


@app.function(
    image=image,
    gpu="A100",
    timeout=86400,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_task3_a100(
    stage: str = "all",
    config_path: str = "configs/modal_task3_config.yaml",
    skip_eval: bool = False,
):
    """Run Task 3 on NVIDIA A100 (80GB VRAM)."""
    _execute_pipeline_in_container("task3", stage, config_path, skip_eval, "NVIDIA A100")


@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=86400,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_task3_rtx6000(
    stage: str = "all",
    config_path: str = "configs/modal_task3_config.yaml",
    skip_eval: bool = False,
):
    """Run Task 3 on Single NVIDIA RTX PRO 6000 Ada (48GB VRAM)."""
    _execute_pipeline_in_container("task3", stage, config_path, skip_eval, "Single NVIDIA RTX PRO 6000 (48GB)")


# ==============================================================================
# Cloud Functions for Task 1 (LLM Fine-Tuning)
# ==============================================================================
@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=86400,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_pipeline_rtx6000(
    stage: str = "all",
    config_path: str = "configs/modal_config.yaml",
    skip_eval: bool = False,
):
    """Run Task 1 on Single NVIDIA RTX PRO 6000 Ada (48GB VRAM)."""
    _execute_pipeline_in_container("task1", stage, config_path, skip_eval, "Single NVIDIA RTX PRO 6000 (48GB)")


@app.function(
    image=image,
    gpu="A100",
    timeout=86400,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_pipeline_a100(
    stage: str = "all",
    config_path: str = "configs/modal_config.yaml",
    skip_eval: bool = False,
):
    """Run Task 1 on NVIDIA A100 (40GB/80GB VRAM)."""
    _execute_pipeline_in_container("task1", stage, config_path, skip_eval, "NVIDIA A100")


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_pipeline_h100(
    stage: str = "all",
    config_path: str = "configs/modal_config.yaml",
    skip_eval: bool = False,
):
    """Run Task 1 on NVIDIA H100 (80GB VRAM)."""
    _execute_pipeline_in_container("task1", stage, config_path, skip_eval, "NVIDIA H100 (80GB)")


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_pipeline_a10g(
    stage: str = "all",
    config_path: str = "configs/modal_config.yaml",
    skip_eval: bool = False,
):
    """Run Task 1 on NVIDIA A10G (24GB VRAM)."""
    _execute_pipeline_in_container("task1", stage, config_path, skip_eval, "NVIDIA A10G (24GB)")


@app.function(
    image=image,
    gpu="L4",
    timeout=86400,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_pipeline_l4(
    stage: str = "all",
    config_path: str = "configs/modal_config.yaml",
    skip_eval: bool = False,
):
    """Run Task 1 on NVIDIA L4 (24GB Ada Lovelace)."""
    _execute_pipeline_in_container("task1", stage, config_path, skip_eval, "NVIDIA L4 (24GB)")


# ==============================================================================
# Verification & Utility Cloud Functions
# ==============================================================================
@app.function(
    image=image,
    gpu="A10G",
    timeout=300,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def check_environment():
    """Verify Modal container build, CUDA, libraries, and volume mount."""
    import os
    import sys
    import torch
    import torchvision
    import PIL
    import transformers
    import lm_eval

    print("\n" + "=" * 60, flush=True)
    print("MODAL ENVIRONMENT & VOLUME VERIFICATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Python: {sys.version}", flush=True)
    print(f"PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}", flush=True)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU VRAM: {vram:.2f} GB", flush=True)
    print(f"Torchvision: {torchvision.__version__}", flush=True)
    print(f"Pillow: {PIL.__version__}", flush=True)
    print(f"Transformers: {transformers.__version__}", flush=True)
    print(f"LM-Eval: {lm_eval.__version__}", flush=True)

    # Test volume write & commit
    test_dir = f"{VOLUME_MOUNT_PATH}/diagnostics_test"
    os.makedirs(test_dir, exist_ok=True)
    test_file = f"{test_dir}/dry_run_check.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(f"Dry-run test verified at {time.ctime()}\n")
    volume.commit()
    print(f"Volume test file written & committed: {test_file}", flush=True)
    print("ALL ENVIRONMENT & HARDWARE CHECKS PASSED!\n", flush=True)


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume})
def list_volume_artifacts(include_checkpoints: bool = False) -> list:
    """List result files stored in persistent volume (excluding heavy model weights/caches by default)."""
    import os

    volume.reload()
    collected = []
    for root, _, files in os.walk(VOLUME_MOUNT_PATH):
        for f in files:
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, VOLUME_MOUNT_PATH)
            # Exclude massive .pt checkpoints, safetensors, and data caches by default
            if not include_checkpoints:
                if (
                    f.endswith(".pt")
                    or f.endswith(".bin")
                    or f.endswith(".safetensors")
                    or f.endswith(".npy")
                    or f.endswith(".arrow")
                    or ("checkpoints" in rel_p and not (f.endswith(".csv") or f.endswith(".json") or f.endswith(".jsonl") or f.endswith(".md")))
                    or "data_cache" in rel_p
                ):
                    continue
            sz_mb = os.path.getsize(full_p) / (1024 * 1024)
            collected.append((rel_p, sz_mb))
    return sorted(collected, key=lambda x: x[0])


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume})
def fetch_volume_file(rel_path: str) -> bytes:
    """Read a specific file from the persistent volume and return bytes."""
    import os

    volume.reload()
    full_p = os.path.join(VOLUME_MOUNT_PATH, rel_path)
    with open(full_p, "rb") as f:
        return f.read()


# ==============================================================================
# Single Local Entrypoint: CLI Interface
# ==============================================================================
@app.local_entrypoint()
def main(
    task: str = "task2",
    stage: str = "all",
    gpu: str = "H100",
    config: str = "",
    skip_eval: bool = False,
    dry_run: bool = False,
    download: bool = False,
    include_checkpoints: bool = False,
    dest_dir: str = ".",
):
    """
    Main entrypoint when running `modal run modal_run.py`.
    Parameters:
      --task: Pipeline target [task1, task2, task3] (Default: task2)
      --stage: Stage to run:
               For task1: [all, baseline, train_full_ft, train_l2sp, train_lora4, train_lora16, train_oplora, diagnostics, qualitative, synthesize]
               For task2: [all, prepare_data, cache_features, stage1, stage2, stage3, evaluate, synthesize]
               For task3: [all, step1_2, step3, step4_live, step5_adaptive, synthesize]
      --gpu: Hardware accelerator [H100, A100, RTX-PRO-6000, A10G, L4] (Default: H100, strictly 1 GPU)
      --config: Path to configuration YAML (Defaults to modal_task3_config.yaml, modal_task2_config.yaml, or modal_config.yaml)
      --skip-eval: Skip evaluation (Task 1 debug flag)
      --dry-run: Fast container & volume test without running training
      --download: Download results, metrics, and logs from cloud volume to local disk (skips heavy checkpoints by default)
      --include-checkpoints: Include heavy 6GB+ .pt checkpoints during download
      --dest-dir: Local destination directory for downloaded files (Default: .)
    """
    if download:
        filter_msg = "all files (including checkpoints)" if include_checkpoints else "results, metrics, and logs only (skipping checkpoints)"
        print(f"[Local CLI] Fetching artifact manifest from cloud volume 'pa3-volume' [{filter_msg}]...")
        artifacts = list_volume_artifacts.remote(include_checkpoints=include_checkpoints)
        if not artifacts:
            print("[Local CLI] No files found in volume yet. Run the pipeline first!")
            return
        print(f"[Local CLI] Found {len(artifacts)} files in volume. Commencing download to '{dest_dir}':")
        for rel_path, sz_mb in artifacts:
            target_path = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            print(f"  -> [{sz_mb:6.2f} MB] {rel_path} ...", end=" ", flush=True)
            data = fetch_volume_file.remote(rel_path)
            with open(target_path, "wb") as f_out:
                f_out.write(data)
            print("Done!")
        print(f"\n[Local CLI] Successfully downloaded all artifacts to '{dest_dir}'!")
        return

    if dry_run:
        print("[Local CLI] Executing fast dry-run environment check...")
        check_environment.remote()
        return

    task_normalized = task.lower().strip()
    gpu_choice = gpu.upper().strip()

    # Safety: Enforce strictly single GPU to protect compute credits
    if ":2" in gpu_choice or ":" in gpu_choice:
        base_gpu = gpu_choice.split(":")[0]
        print(f"[Local CLI Safety] Multi-GPU '{gpu_choice}' requested, but single GPU invariant is enforced to prevent accidental billing.")
        print(f"[Local CLI Safety] Normalizing to single GPU: '{base_gpu}'")
        gpu_choice = base_gpu

    # Resolve default configuration
    if not config:
        if task_normalized == "task1":
            config = "configs/modal_config.yaml"
        elif task_normalized == "task2":
            config = "configs/modal_task2_config.yaml"
        else:
            config = "configs/modal_task3_config.yaml"

    print(f"[Local CLI] Dispatching {task_normalized.upper()} to Modal Cloud GPU: {gpu_choice} | Stage(s): {stage} | Config: {config}")

    if task_normalized == "task3":
        if "H100" in gpu_choice:
            run_task3_h100.remote(stage=stage, config_path=config, skip_eval=skip_eval)
        elif "A100" in gpu_choice:
            run_task3_a100.remote(stage=stage, config_path=config, skip_eval=skip_eval)
        elif "6000" in gpu_choice or "RTX" in gpu_choice:
            run_task3_rtx6000.remote(stage=stage, config_path=config, skip_eval=skip_eval)
        else:
            run_task3_h100.remote(stage=stage, config_path=config, skip_eval=skip_eval)
    elif task_normalized == "task2":
        if "A100" in gpu_choice:
            run_task2_a100.remote(stage=stage, config_path=config, skip_eval=skip_eval)
        elif "H100" in gpu_choice:
            run_task2_h100.remote(stage=stage, config_path=config, skip_eval=skip_eval)
        elif "6000" in gpu_choice or "RTX" in gpu_choice:
            run_task2_rtx6000.remote(stage=stage, config_path=config, skip_eval=skip_eval)
        elif "A10G" in gpu_choice:
            run_task2_a10g.remote(stage=stage, config_path=config, skip_eval=skip_eval)
        else:
            run_task2_h100.remote(stage=stage, config_path=config, skip_eval=skip_eval)
    else:
        # Task 1
        if "6000" in gpu_choice or "RTX" in gpu_choice:
            run_pipeline_rtx6000.remote(stage=stage, config_path=config, skip_eval=skip_eval)
        elif "H100" in gpu_choice:
            run_pipeline_h100.remote(stage=stage, config_path=config, skip_eval=skip_eval)
        elif "A100" in gpu_choice:
            run_pipeline_a100.remote(stage=stage, config_path=config, skip_eval=skip_eval)
        elif "L4" in gpu_choice:
            run_pipeline_l4.remote(stage=stage, config_path=config, skip_eval=skip_eval)
        else:
            run_pipeline_a10g.remote(stage=stage, config_path=config, skip_eval=skip_eval)
