"""共通ユーティリティ関数"""

import random

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """乱数シードを固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "configs/config.yaml") -> dict:
    """YAML設定ファイルを読み込み"""
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """利用可能なデバイスを取得"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
