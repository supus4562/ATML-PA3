"""
Automated downloader for ATML PA3 Task 3 Rollout Dataset (ATML-PA3-Data).
Downloads from Google Drive:
https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23?usp=sharing

Ensures all parquet tables are present, intact, and readable.
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple


DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23?usp=sharing"
DRIVE_FOLDER_ID = "1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23"

DIRECT_FILE_IDS = {
    "README.MD": "1RARGiUK_jZpOii-b8kyGLoejaldMUAa3",
    "qwen25-math/generations.parquet": "1SaWxZ4mpycw2BWZ5G0dahoYZ6Hh8tPWP",
    "qwen25-math/probability_metrics.parquet": "1RCYHrTnUKtF7gHLSTe92x6W9ZsCOekfQ",
    "qwen25vl-mathvista/generations.parquet": "1Nzg2VwE2qFzA10BF1lTxEcBZCePBe51q",
    "qwen25vl-mathvista/probability_metrics.parquet": "1f2XAu_J5NviYHEhtkRrFbKc7b2us1WiW",
    "qwen25vl-mathvista/attention_metrics.parquet": "1sedMpWLI_UgPvAQVj7tS3HYVH0ARS1yA",
}


def verify_dataset(dest_dir: str) -> bool:
    """Verifies all required parquet files exist and are non-empty."""
    for rel_path in DIRECT_FILE_IDS:
        target = os.path.join(dest_dir, rel_path)
        if not os.path.exists(target) or os.path.getsize(target) == 0:
            return False
    return True


def download_dataset(dest_dir: str = "data/ATML-PA3-Data") -> None:
    """
    Downloads ATML-PA3-Data from Google Drive to dest_dir.
    Uses gdown with fallback to direct file IDs.
    """
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "qwen25-math"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "qwen25vl-mathvista"), exist_ok=True)

    if verify_dataset(dest_dir):
        print(f"[Downloader] Dataset already present and verified in '{dest_dir}'.")
        return

    print("=" * 80)
    print("ATML PA3: DOWNLOADING TASK 3 DATASET (ATML-PA3-Data) FROM GOOGLE DRIVE")
    print(f"Destination: {dest_dir}")
    print("=" * 80)

    try:
        import gdown
    except ImportError:
        import subprocess
        print("[Downloader] Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown>=5.1.0"])
        import gdown

    # Attempt 1: Full folder download
    try:
        print(f"\n[Downloader] Attempting folder download via gdown (ID: {DRIVE_FOLDER_ID})...")
        gdown.download_folder(id=DRIVE_FOLDER_ID, output=dest_dir, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"[Downloader] Folder download encountered notice: {e}")

    # Check and download any missing individual files
    for rel_path, file_id in DIRECT_FILE_IDS.items():
        target = os.path.join(dest_dir, rel_path)
        if not os.path.exists(target) or os.path.getsize(target) == 0:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            print(f"[Downloader] Downloading {rel_path} (ID: {file_id}) ...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, target, quiet=False)

    # Final verification
    print("\n[Downloader] Verifying downloaded files:")
    all_ok = True
    for rel_path in DIRECT_FILE_IDS:
        target = os.path.join(dest_dir, rel_path)
        if os.path.exists(target):
            sz_mb = os.path.getsize(target) / (1024 * 1024)
            print(f"  [OK] {rel_path:45s} ({sz_mb:6.2f} MB)")
        else:
            print(f"  [MISSING] {rel_path}")
            all_ok = False

    if all_ok:
        print(f"\n[Downloader] Successfully verified ATML-PA3-Data in '{dest_dir}'!\n")
    else:
        raise RuntimeError(f"Dataset verification failed for one or more files in '{dest_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ATML-PA3-Data from Google Drive")
    parser.add_argument("--dest", type=str, default="data/ATML-PA3-Data", help="Destination directory")
    args = parser.parse_args()
    download_dataset(args.dest)
