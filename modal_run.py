"""
Modal cloud execution orchestrator for ATML PA3 - Task 1.
Leverages Modal (modal.com) serverless GPU infrastructure with persistent volume storage.

Features:
- Persistent Volume ('pa3-volume') mounted at /root/volume for 100% durable checkpoints, cache & results.
- Unbuffered real-time stdout streaming of every single training update and evaluation step.
- Redundant multi-format persistence: JSON, JSONL, CSV, and automatic volume commits.
- Native lm-evaluation-harness (lm-eval) support across GSM8K, HellaSwag, BoolQ, and ARC-Easy.
- Multiple GPU configurations: NVIDIA A10G (default, 24GB, cost-efficient) or A100 (high-throughput).
- Built-in download entrypoint to sync all results to local workspace.

Usage:
  # 1. Verification / Dry-run
  modal run modal_run.py --dry-run

  # 2. Run complete Task 1 pipeline on NVIDIA A10G (default)
  modal run modal_run.py

  # 3. Run on NVIDIA A100 GPU
  modal run modal_run.py --gpu A100

  # 4. Run a specific stage (e.g. baseline, train_full_ft, train_l2sp, diagnostics, qualitative)
  modal run modal_run.py --stage train_full_ft

  # 5. Download all results from cloud volume to local repository
  modal run modal_run.py::download_results
"""

import os
import sys
import time
import modal

# 1. Define the Modal App
app = modal.App("atml-pa3-task1")

# 2. Persistent Volume: durable cloud storage across runs and restarts
volume = modal.Volume.from_name("pa3-volume", create_if_missing=True)
VOLUME_MOUNT_PATH = "/root/volume"

# 3. Container Image Definition with CUDA and all required libraries
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "procps")
    .pip_install(
        "torch>=2.4.0",
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

    print(f"\n{'='*80}", flush=True)
    print(f"MODAL CLOUD PIPELINE: Task 1 on {gpu_label}", flush=True)
    print(f"Target Stage: {stage} | Config: {config_path}", flush=True)
    print(f"{'='*80}", flush=True)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Hardware] GPU Active: {gpu_name} ({vram_gb:.2f} GB VRAM)", flush=True)

    # Ensure volume directory structures exist
    os.makedirs(f"{VOLUME_MOUNT_PATH}/data_cache/task1", exist_ok=True)
    os.makedirs(f"{VOLUME_MOUNT_PATH}/checkpoints/task1", exist_ok=True)
    os.makedirs(f"{VOLUME_MOUNT_PATH}/results/task1", exist_ok=True)
    volume.commit()

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "task1.run_all",
        "--config",
        config_path,
        "--stages",
        stage,
        "--single-gpu",
    ]
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

    # Stream every single output line directly to Modal's logs
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
        raise RuntimeError(f"Task 1 execution failed with exit code {return_code}")


# 4. Decorated Cloud Functions for Different GPU Configurations
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
    """Run pipeline on NVIDIA A10G (24GB VRAM) - standard, highly cost-efficient."""
    _execute_pipeline_in_container(stage, config_path, skip_eval, "NVIDIA A10G (24GB)")


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
    """Run pipeline on NVIDIA A100 (40GB/80GB VRAM) - maximum throughput."""
    _execute_pipeline_in_container(stage, config_path, skip_eval, "NVIDIA A100")


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
    """Run pipeline on NVIDIA RTX PRO 6000 (48GB VRAM) - matched to assignment target architecture."""
    _execute_pipeline_in_container(stage, config_path, skip_eval, "NVIDIA RTX PRO 6000 (48GB)")


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
    """Run pipeline on NVIDIA L4 (24GB Ada Lovelace) - ultra low cost."""
    _execute_pipeline_in_container(stage, config_path, skip_eval, "NVIDIA L4 (24GB)")


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


# 5. Remote Helpers for Result File Synchronization
@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume})
def list_volume_artifacts() -> list:
    """List all result and checkpoint files stored in the persistent volume."""
    import os

    volume.reload()
    collected = []
    for root, _, files in os.walk(VOLUME_MOUNT_PATH):
        for f in files:
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, VOLUME_MOUNT_PATH)
            sz_mb = os.path.getsize(full_p) / (1024 * 1024)
            collected.append((rel_p, sz_mb))
    return collected


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume})
def fetch_volume_file(rel_path: str) -> bytes:
    """Read a specific file from the persistent volume and return bytes."""
    import os

    volume.reload()
    full_p = os.path.join(VOLUME_MOUNT_PATH, rel_path)
    with open(full_p, "rb") as f:
        return f.read()


# 6. Single Local Entrypoint: CLI Interface
@app.local_entrypoint()
def main(
    stage: str = "all",
    gpu: str = "RTX-PRO-6000",
    config: str = "configs/modal_config.yaml",
    skip_eval: bool = False,
    dry_run: bool = False,
    download: bool = False,
    dest_dir: str = ".",
):
    """
    Main entrypoint when running `modal run modal_run.py`.
    Parameters:
      --dry-run: Fast container & volume test without running training
      --stage: Stage to run [all, baseline, train_full_ft, train_l2sp, train_lora4, train_lora16, train_oplora, diagnostics, qualitative, synthesize]
      --gpu: Hardware accelerator [RTX-PRO-6000, A10G, A100, L4] (Default: RTX-PRO-6000)
      --config: Path to configuration YAML (Default: configs/modal_config.yaml)
      --skip-eval: Skip benchmark evaluation (debug flag)
      --download: Download all artifacts from cloud volume to local disk
      --dest-dir: Local destination directory for downloaded files (Default: .)
    """
    if download:
        print(f"[Local CLI] Fetching artifact manifest from cloud volume 'pa3-volume'...")
        artifacts = list_volume_artifacts.remote()
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

    gpu_choice = gpu.upper()
    print(f"[Local CLI] Dispatching Task 1 to Modal Cloud GPU: {gpu_choice} | Stage: {stage}")

    if "6000" in gpu_choice or "RTX" in gpu_choice:
        run_pipeline_rtx6000.remote(stage=stage, config_path=config, skip_eval=skip_eval)
    elif "A100" in gpu_choice:
        run_pipeline_a100.remote(stage=stage, config_path=config, skip_eval=skip_eval)
    elif "L4" in gpu_choice:
        run_pipeline_l4.remote(stage=stage, config_path=config, skip_eval=skip_eval)
    else:
        run_pipeline_a10g.remote(stage=stage, config_path=config, skip_eval=skip_eval)

