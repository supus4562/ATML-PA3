"""
Configuration schemas and loaders for Task 1 experiments.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    enable_thinking: bool = False
    attn_implementation: str = "sdpa"
    seed: int = 6304


@dataclass
class DataConfig:
    dataset_name: str = "meta-math/MetaMathQA"
    max_examples: int = 40000
    max_seq_length: int = 2048
    filter_gsm8k_leakage: bool = True
    cache_dir: str = "data_cache/task1"


@dataclass
class OptimizerConfig:
    name: str = "AdamW"
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1.0e-8
    weight_decay: float = 0.01
    fused: bool = True


@dataclass
class TrainingConfig:
    num_epochs: int = 1
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 16  # Effective batch size = 32
    total_updates: int = 1250              # 40,000 / 32
    warmup_updates: int = 38
    learning_rate_schedule: str = "cosine"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    gradient_clipping: float = 1.0
    gradient_checkpointing: bool = False
    use_cache: bool = True
    fp16: bool = True
    checkpoint_updates: List[int] = field(default_factory=lambda: [312, 625, 937, 1250])
    evaluation_updates: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "full_ft": [625, 1250],
            "full_ft_l2sp": [1250],
            "lora_4": [1250],
            "lora_16": [1250],
            "oplora_16": [1250],
        }
    )


@dataclass
class FullFTConfig:
    peak_lr: float = 2.0e-5
    lambda_l2sp: float = 0.0


@dataclass
class FullFTL2SPConfig:
    peak_lr: float = 2.0e-5
    lambda_l2sp: float = 1.0e-3


@dataclass
class LoRAConfig:
    peak_lr: float = 2.0e-4
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )


@dataclass
class OPLoRAConfig:
    peak_lr: float = 2.0e-4
    rank: int = 16
    alpha: int = 32
    protection_rank_k: int = 128
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )


@dataclass
class DiagnosticsConfig:
    target_layers: List[int] = field(default_factory=lambda: [0, 13, 27])
    target_module: str = "down_proj"
    protection_rank_k: int = 128
    num_probe_gsm8k: int = 32
    num_probe_hellaswag: int = 32
    seed: int = 6304


@dataclass
class QualitativeConfig:
    prompts_file: str = "task1/first_plot_questions.yaml"
    max_new_tokens: int = 512
    temperature: float = 0.0
    conditions: List[str] = field(
        default_factory=lambda: [
            "baseline",
            "full_ft_midpoint",
            "full_ft_final",
            "full_ft_l2sp_final",
            "lora_4_final",
            "lora_16_final",
            "oplora_16_final",
        ]
    )


@dataclass
class BenchmarkTaskConfig:
    name: str
    split: str
    num_samples: int
    task: str
    metric: str
    chat_template: bool = False
    enable_thinking: bool = False
    num_fewshot: int = 0
    max_new_tokens: int = 1024


@dataclass
class EvaluationConfig:
    benchmarks: List[Dict[str, Any]] = field(default_factory=list)
    batch_size: int = 32


@dataclass
class PathsConfig:
    output_dir: str = "results/task1"
    checkpoint_dir: str = "checkpoints/task1"


@dataclass
class Task1Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    methods: Dict[str, Any] = field(default_factory=dict)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    qualitative: QualitativeConfig = field(default_factory=QualitativeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Task1Config":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = cls()
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "training" in data:
            opt_data = data["training"].get("optimizer", {})
            opt_cfg = OptimizerConfig(**opt_data)
            t_data = {k: v for k, v in data["training"].items() if k != "optimizer"}
            config.training = TrainingConfig(optimizer=opt_cfg, **t_data)
        if "methods" in data:
            config.methods = data["methods"]
        if "diagnostics" in data:
            config.diagnostics = DiagnosticsConfig(**data["diagnostics"])
        if "qualitative" in data:
            config.qualitative = QualitativeConfig(**data["qualitative"])
        if "evaluation" in data:
            config.evaluation = EvaluationConfig(**data["evaluation"])
        if "paths" in data:
            config.paths = PathsConfig(**data["paths"])

        return config
