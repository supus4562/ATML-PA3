"""
Configuration schema for Task 2: Vision-Language Models and Visual Grounding.
Defines typed dataclasses for Vision, Connector, Language Model, Data,
Curriculum Stages, Evaluation, and Paths, with YAML load/dump support.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
import yaml


@dataclass
class VisionConfig:
    model_name_or_path: str = "google/siglip-base-patch16-224"
    image_size: int = 224
    num_patches: int = 196
    hidden_dim: int = 768
    apply_final_norm: bool = True
    compressed_patches: int = 49  # Optional 2x2 spatial pooling experiment


@dataclass
class ConnectorConfig:
    in_features: int = 768
    hidden_features: int = 1024
    out_features: int = 1024
    activation: str = "gelu"
    init_scale: float = 1.0


@dataclass
class LanguageModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    enable_thinking: bool = False
    attn_implementation: str = "sdpa"
    seed: int = 6304


@dataclass
class DataConfig:
    coco_captions_train_count: int = 18000     # 6,000 images x 3 captions
    coco_captions_dev_count: int = 96          # 96 images, 1 caption each
    grounding_dev_pairs: int = 24              # 24 pairs / 48 questions
    grounding_test_pairs: int = 64             # 64 pairs / 128 questions
    natural_vqa_train_pairs: int = 1536        # 640 yes/no, 256 number, 640 other
    natural_vqa_dev_pairs: int = 32            # 12 yes/no, 8 number, 12 other
    natural_vqa_test_pairs: int = 96           # 36 yes/no, 24 number, 36 other
    cache_dir: str = "data_cache/task2"
    feature_cache_dir: str = "data_cache/task2/siglip_features"
    seed: int = 6304


@dataclass
class OptimizerConfig:
    name: str = "AdamW"
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1.0e-8
    weight_decay: float = 0.01
    fused: bool = True


@dataclass
class Stage1Config:
    updates: int = 2250
    connector_peak_lr: float = 1.0e-3
    lora_enabled: bool = False


@dataclass
class Stage2Config:
    updates: int = 1500
    connector_peak_lr: float = 1.0e-4
    lora_peak_lr: float = 2.0e-5
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    sampling_probs: Dict[str, float] = field(default_factory=lambda: {
        "presence": 0.50,
        "counting": 0.25,
        "caption_replay": 0.25,
    })


@dataclass
class Stage3Config:
    updates: int = 750
    connector_peak_lr: float = 5.0e-5
    lora_peak_lr: float = 1.0e-5
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    sampling_probs: Dict[str, float] = field(default_factory=lambda: {
        "natural_vqa": 0.70,
        "presence_replay": 0.10,
        "counting_replay": 0.10,
        "caption_replay": 0.10,
    })


@dataclass
class TrainingConfig:
    seed: int = 6304
    num_epochs: int = 1
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 2       # Effective batch size = 16
    total_updates: int = 4500                  # 2250 (S1) + 1500 (S2) + 750 (S3)
    warmup_ratio: float = 0.05                 # 5% per-stage warmup
    min_lr_ratio: float = 0.10                 # Cosine decay to 10% of peak LR
    gradient_clipping: float = 1.0
    fp16: bool = True
    gradient_checkpointing: bool = False       # Disabled for max speed on A100 / RTX 6000
    question_token_cap: int = 80
    caption_target_cap: int = 64
    short_answer_target_cap: int = 16
    checkpoint_frequency: int = 250
    endpoint_updates: List[int] = field(default_factory=lambda: [2250, 3750, 4500])
    dev_eval_frequency: int = 500
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)


@dataclass
class EvaluationConfig:
    batch_size: int = 8
    max_new_tokens: int = 16
    temperature: float = 0.0                   # Strict greedy decoding
    bootstrap_resamples: int = 4000
    confidence_level: float = 0.95
    conditions: List[str] = field(default_factory=lambda: [
        "correct",
        "complementary_swap",
        "mismatched",
        "no_image",
    ])


@dataclass
class PathsConfig:
    output_dir: str = "results/task2"
    checkpoint_dir: str = "checkpoints/task2"


@dataclass
class Task2Config:
    vision: VisionConfig = field(default_factory=VisionConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Task2Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        vision_data = data.get("vision", {})
        connector_data = data.get("connector", {})
        lm_data = data.get("language_model", {})
        data_data = data.get("data", {})
        training_data = data.get("training", {})
        eval_data = data.get("evaluation", {})
        paths_data = data.get("paths", {})

        opt_data = training_data.get("optimizer", {})
        s1_data = training_data.get("stage1", {})
        s2_data = training_data.get("stage2", {})
        s3_data = training_data.get("stage3", {})

        optimizer = OptimizerConfig(**opt_data) if opt_data else OptimizerConfig()
        stage1 = Stage1Config(**s1_data) if s1_data else Stage1Config()
        stage2 = Stage2Config(**s2_data) if s2_data else Stage2Config()
        stage3 = Stage3Config(**s3_data) if s3_data else Stage3Config()

        t_clean = {k: v for k, v in training_data.items() if k not in ["optimizer", "stage1", "stage2", "stage3"]}
        training = TrainingConfig(
            optimizer=optimizer,
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            **t_clean
        )

        return cls(
            vision=VisionConfig(**vision_data) if vision_data else VisionConfig(),
            connector=ConnectorConfig(**connector_data) if connector_data else ConnectorConfig(),
            language_model=LanguageModelConfig(**lm_data) if lm_data else LanguageModelConfig(),
            data=DataConfig(**data_data) if data_data else DataConfig(),
            training=training,
            evaluation=EvaluationConfig(**eval_data) if eval_data else EvaluationConfig(),
            paths=PathsConfig(**paths_data) if paths_data else PathsConfig(),
        )

    def to_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
