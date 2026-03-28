"""Robust LOSO cross-validation training script"""

import hydra
from omegaconf import DictConfig, OmegaConf
from src.experiments.loso import LOSOExperiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for LOSO training"""
    # Print config for debugging
    print(OmegaConf.to_yaml(cfg))

    # Create and run experiment
    experiment = LOSOExperiment(
        model_name=cfg.model.name,
        model_config=cfg.model,
        train_config=cfg,
        resume_dir=cfg.resume_dir,
    )
    if cfg.get("tune_mode", False):
        f1_macro = experiment.run_tuning()
        return f1_macro
    else:
        experiment.run(limit_folds=cfg.limit_folds)


if __name__ == "__main__":
    main()
