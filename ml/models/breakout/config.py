"""
ml/models/breakout/config.py

Model configuration loaded from YAML.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    timeframe: str
    mfe_threshold: float = 1.0
    mae_threshold: float = 1.0
    confidence_threshold: float = 0.65
    fold_size: int = 50
    min_train: int = 500

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        import yaml
        with open(path) as f:
            return cls(**yaml.safe_load(f))
