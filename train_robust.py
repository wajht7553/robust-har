"""Robust LOSO cross-validation training script"""

import argparse
from src.utils.config import load_config, load_model_config
from src.experiments import RobustLOSOExperiment


def main():
    """Main entry point for robust LOSO training"""
    parser = argparse.ArgumentParser(
        description="Train models using Robust LOSO cross-validation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., mobilevit, deepconvlstm, mamba)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config override",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train_robust.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--limit_folds",
        type=int,
        default=None,
        help="Limit number of folds for debugging",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to experiment directory to resume from",
    )
    args = parser.parse_args()

    # Load configurations
    train_config = load_config(args.train_config)
    model_config = load_model_config(args.model, args.config)

    # Create and run experiment
    experiment = RobustLOSOExperiment(
        args.model,
        model_config,
        train_config,
        resume_dir=args.resume,
    )
    experiment.run(limit_folds=args.limit_folds)


if __name__ == "__main__":
    main()
