"""Configuration loading utilities"""

import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_config(
    model_name: str, config_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load model configuration from default path or override.

    Args:
        model_name: Name of the model
        config_override: Optional path to override config file

    Returns:
        Model configuration dictionary

    Raises:
        ValueError: If config file not found
    """
    if config_override:
        if not os.path.exists(config_override):
            raise ValueError(f"Config file not found: {config_override}")
        return load_config(config_override)

    model_config_path = f"configs/model/{model_name}.yaml"
    if os.path.exists(model_config_path):
        return load_config(model_config_path)
    else:
        raise ValueError(
            f"No default config found for {model_name} at {model_config_path}"
        )
