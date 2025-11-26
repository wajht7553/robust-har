"""Robust LOSO cross-validation training script"""

import hydra
from omegaconf import DictConfig, OmegaConf
from src.experiments import RobustLOSOExperiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for robust LOSO training"""
    # Print config for debugging
    print(OmegaConf.to_yaml(cfg))

    # Create and run experiment
    # Pass the whole config, experiment will extract what it needs
    # Model config is now under cfg.model
    # Train config is at root level (or we can structure it better, but for now root)

    experiment = RobustLOSOExperiment(
        model_name=cfg.model.name,
        model_config=cfg.model,
        train_config=cfg,
        resume_dir=cfg.resume_dir,
    )
    experiment.run(limit_folds=cfg.limit_folds)


if __name__ == "__main__":
    main()
