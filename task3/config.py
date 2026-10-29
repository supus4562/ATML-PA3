"""
Configuration dataclasses and YAML parser for Task 3: Test-Time Compute.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml


@dataclass
class PathsConfig:
    data_dir: str = "data/ATML-PA3-Data"
    output_dir: str = "results/task3"
    checkpoint_dir: str = "checkpoints/task3"


@dataclass
class ModelEntryConfig:
    name: str = "Qwen/Qwen2.5-3B-Instruct"
    dataset: str = "MATH"
    type: str = "LLM"
    rollout_folder: str = "Qwen2.5-3B-Instruct"


@dataclass
class OfflineEvalConfig:
    budgets: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    num_permutations: int = 10
    dev_split_size: int = 200
    eval_split_size: int = 800
    bootstrap_resamples: int = 1000
    confidence_level: float = 0.95


@dataclass
class AdaptiveConfig:
    checkpoints: List[int] = field(default_factory=lambda: [4, 8, 16])
    max_candidates: int = 32
    agreement_thresholds: List[float] = field(default_factory=lambda: [0.50, 0.625, 0.75, 0.875, 1.00])
    quality_margins: List[float] = field(default_factory=lambda: [0.00, 0.10, 0.25, 0.50])
    dev_accuracy_margin_pp: float = 1.0


@dataclass
class LiveGenConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    num_questions: int = 50
    batch_size: int = 1
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.0
    single_response_max_tokens: int = 3072
    parallel_response_max_tokens: int = 1024
    refinement_revision_max_tokens: int = 2048
    system_prompt: str = "You are a helpful assistant."


@dataclass
class Task3Config:
    seed: int = 6304
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: Dict[str, ModelEntryConfig] = field(default_factory=dict)
    offline_eval: OfflineEvalConfig = field(default_factory=OfflineEvalConfig)
    adaptive_sampling: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    live_generation: LiveGenConfig = field(default_factory=LiveGenConfig)


def load_task3_config(config_path: str) -> Task3Config:
    """Load Task 3 configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Task3Config()
    cfg.seed = int(raw.get("seed", 6304))

    if "paths" in raw:
        cfg.paths = PathsConfig(**raw["paths"])

    if "models" in raw:
        cfg.models = {k: ModelEntryConfig(**v) for k, v in raw["models"].items()}
    else:
        cfg.models = {
            "math_llm": ModelEntryConfig(
                name="Qwen/Qwen2.5-3B-Instruct",
                dataset="MATH",
                type="LLM",
                rollout_folder="Qwen2.5-3B-Instruct",
            ),
            "mathvista_vlm": ModelEntryConfig(
                name="Qwen/Qwen2.5-VL-3B-Instruct",
                dataset="MathVista",
                type="VLM",
                rollout_folder="Qwen2.5-VL-3B-Instruct",
            ),
        }

    if "offline_eval" in raw:
        cfg.offline_eval = OfflineEvalConfig(**raw["offline_eval"])

    if "adaptive_sampling" in raw:
        cfg.adaptive_sampling = AdaptiveConfig(**raw["adaptive_sampling"])

    if "live_generation" in raw:
        cfg.live_generation = LiveGenConfig(**raw["live_generation"])

    return cfg
