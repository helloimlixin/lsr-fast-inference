#!/usr/bin/env python3
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.main import run_experiment


def _save_run_config(cfg):
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_dir.joinpath("config.yaml").open("w") as handle:
        handle.write(OmegaConf.to_yaml(cfg))


@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint."""
    print(OmegaConf.to_yaml(cfg))
    _save_run_config(cfg)
    return run_experiment(cfg)


if __name__ == "__main__":
    main()
