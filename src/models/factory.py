from typing import Union, Dict, Any
from omegaconf import DictConfig
from .DeepConvLSTM import DeepConvLSTM, DeepConvContext
from .MobileViT import MobileViT
from .Mamba import MambaHAR
from .TinyHAR import TinyHAR
from .MobileNetV3 import MobileNetV3


def create_model(model_name: str, config: Union[Dict[str, Any], DictConfig]):
    """
    Create a model instance based on name and config.

    Args:
        model_name (str): Name of the model (deepconvlstm, deepconvcontext, mobilevit, mamba, tinyhar, mobilenetv3)
        config (dict or DictConfig): Model configuration

    Returns:
        nn.Module: The requested model
    """
    name = model_name.lower()

    if name == "deepconvlstm":
        return DeepConvLSTM(
            channels=config["channels"],
            classes=config["classes"],
            window_size=config["window_size"],
            conv_kernels=config.get("conv_kernels", 64),
            conv_kernel_size=config.get("conv_kernel_size", 5),
            lstm_units=config.get("lstm_units", 128),
            lstm_layers=config.get("lstm_layers", 2),
            dropout=config.get("dropout", 0.5),
        )
    elif name == "deepconvcontext":
        return DeepConvContext(
            batch_size=config["batch_size"],
            channels=config["channels"],
            classes=config["classes"],
            window_size=config["window_size"],
            conv_kernels=config.get("conv_kernels", 64),
            conv_kernel_size=config.get("conv_kernel_size", 5),
            lstm_units=config.get("lstm_units", 128),
            lstm_layers=config.get("lstm_layers", 2),
            dropout=config.get("dropout", 0.5),
            bidirectional=config.get("bidirectional", False),
            type=config.get("type", "lstm"),
            attention_num_heads=config.get("attention_num_heads", 4),
            transformer_depth=config.get("transformer_depth", 6),
        )
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
