from .DeepConvLSTM import DeepConvLSTM
from .MobileViT import MobileViT
from .Mamba import MambaHAR
from .TinyHAR import TinyHAR
from .MobileNetV3 import MobileNetV3


def create_model(model_name, config):
    """
    Create a model instance based on name and config.

    Args:
        model_name (str): Name of the model (deepconvlstm, mobilevit, mamba, tinyhar, mobilenetv3)
        config (dict): Model configuration

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
