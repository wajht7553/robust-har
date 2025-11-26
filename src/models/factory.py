from typing import Union, Dict, Any
from omegaconf import DictConfig
from .DeepConvLSTM import DeepConvLSTM
from .MobileViT import MobileViT
from .Mamba import MambaHAR
from .TinyHAR import TinyHAR
from .MobileNetV3 import MobileNetV3


def create_model(model_name: str, config: Union[Dict[str, Any], DictConfig]):
    """
    Create a model instance based on name and config.

    Args:
        model_name (str): Name of the model (deepconvlstm, mobilevit, mamba, tinyhar, mobilenetv3)
        config (dict or DictConfig): Model configuration

    Returns:
        nn.Module: The requested model
    """
    name = model_name.lower()

    if name == "deepconvlstm":
        return DeepConvLSTM(config)
    elif name == "mobilevit":
        return MobileViT(config)
    elif name == "mamba":
        return MambaHAR(config)
    elif name == "tinyhar":
        return TinyHAR(config)
    elif name == "mobilenetv3":
        return MobileNetV3(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
