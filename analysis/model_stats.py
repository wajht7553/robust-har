import argparse
import os
import sys
import yaml
import torch

"""
Ensure repo root is on sys.path so `src/...` imports work
when running this script from the project root.
"""
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Local imports
from src.models.DeepConvLSTM import DeepConvLSTM
from src.models.MobileViT import MobileViT


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def try_compute_flops(model: torch.nn.Module, input_tensor: torch.Tensor):
    """Compute FLOPs using thop if available; otherwise return None."""
    try:
        from thop import profile
        flops, params_thop = profile(model, inputs=(input_tensor,))
        return int(flops)
    except Exception:
        return None


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_deepconvlstm(cfg_path: str, device: str = "cpu"):
    cfg = load_yaml(cfg_path)
    model = DeepConvLSTM(
        channels=cfg.get("channels", 6),
        classes=cfg.get("classes", 8),
        window_size=cfg.get("window_size", 250),
        conv_kernels=cfg.get("conv_kernels", 64),
        conv_kernel_size=cfg.get("conv_kernel_size", 5),
        lstm_units=cfg.get("lstm_units", 128),
        lstm_layers=cfg.get("lstm_layers", 2),
        dropout=cfg.get("dropout", 0.5),
    ).to(device)

    # Input: (B, L, C)
    B = 1
    L = cfg.get("window_size", 250)
    C = cfg.get("channels", 6)
    x = torch.randn(B, L, C, device=device)
    return model, x


def build_mobilevit(cfg_path: str, device: str = "cpu"):
    cfg = load_yaml(cfg_path)
    model = MobileViT(cfg).to(device)

    # Input: (B, L, C)
    B = 1
    L = cfg.get("window_size", 250)
    C = cfg.get("nb_channels", 6)
    x = torch.randn(B, L, C, device=device)
    return model, x


def format_num(n: int) -> str:
    if n is None:
        return "N/A"
    # Use millions for params, giga for FLOPs
    return f"{n:,}"


def main():
    parser = argparse.ArgumentParser(
        description="Compute parameters and GFLOPs for models"
    )
    parser.add_argument(
        "--deepconvlstm_cfg",
        default=os.path.join("conf", "model", "deepconvlstm.yaml"),
        help="Path to DeepConvLSTM YAML config",
    )
    parser.add_argument(
        "--mobilevit_cfg",
        default=os.path.join("conf", "model", "mobilevit.yaml"),
        help="Path to MobileViT YAML config",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for FLOPs profiling",
    )
    args = parser.parse_args()

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    results = []

    # DeepConvLSTM
    dcl_model, dcl_input = build_deepconvlstm(args.deepconvlstm_cfg, device)
    dcl_params = count_params(dcl_model)
    dcl_flops = try_compute_flops(dcl_model, dcl_input)
    results.append({
        "model": "DeepConvLSTM",
        "params": dcl_params,
        "flops": dcl_flops,
    })

    # MobileViT
    mv_model, mv_input = build_mobilevit(args.mobilevit_cfg, device)
    mv_params = count_params(mv_model)
    mv_flops = try_compute_flops(mv_model, mv_input)
    results.append({
        "model": "MobileViT",
        "params": mv_params,
        "flops": mv_flops,
    })

    # Print results
    print("\nModel Stats (Params, FLOPs)\n----------------------------")
    for r in results:
        params_m = r["params"] / 1e6
        flops_g = (r["flops"] / 1e9) if r["flops"] is not None else None
        flops_str = f"{flops_g:.3f} G" if flops_g is not None else "N/A"
        print(f"{r['model']}: {params_m:.3f} M params, {flops_str} FLOPs")

    # Detailed numbers
    print("\nRaw counts:")
    for r in results:
        print(f"{r['model']}: params={format_num(r['params'])}, flops={format_num(r['flops'])}")


if __name__ == "__main__":
    main()
