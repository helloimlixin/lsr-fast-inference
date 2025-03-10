#!/usr/bin/env python
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from src.main import run_experiment


@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for Hydra application.

    Args:
        cfg: Hydra configuration
    """
    # Print resolved config.yaml for debugging
    print(OmegaConf.to_yaml(cfg))

    # Create output directory if it doesn't exist
    os.makedirs(cfg.training.output_dir, exist_ok=True)

    # Save configuration for reproducibility
    with open(os.path.join(cfg.training.output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Run the experiment
    return run_experiment(cfg)


if __name__ == "__main__":
    main()