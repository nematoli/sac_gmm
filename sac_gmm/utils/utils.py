import os
import hydra
from pathlib import Path
import numpy as np
import tqdm
import torch
import pytorch_lightning
from typing import Dict, List, Union
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger


def info_packages() -> Dict[str, str]:
    return {
        "numpy": np.__version__,
        "pyTorch_version": torch.__version__,
        "pyTorch_debug": str(torch.version.debug),
        "pytorch-lightning": pytorch_lightning.__version__,
        "tqdm": tqdm.__version__,
    }


def info_cuda() -> Dict[str, Union[str, List[str]]]:
    return {
        "GPU": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        # 'nvidia_driver': get_nvidia_driver_version(run_lambda),
        "available": str(torch.cuda.is_available()),
        "version": torch.version.cuda,
    }


def print_system_env_info():
    details = {
        "Packages": info_packages(),
        "CUDA": info_cuda(),
    }
    lines = nice_print(details)
    text = os.linesep.join(lines)
    return text


def nice_print(details: Dict, level: int = 0) -> List:
    lines = []
    LEVEL_OFFSET = "\t"
    KEY_PADDING = 20
    for k in sorted(details):
        key = f"* {k}:" if level == 0 else f"- {k}:"
        if isinstance(details[k], dict):
            lines += [level * LEVEL_OFFSET + key]
            lines += nice_print(details[k], level + 1)
        elif isinstance(details[k], (set, list, tuple)):
            lines += [level * LEVEL_OFFSET + key]
            lines += [(level + 1) * LEVEL_OFFSET + "- " + v for v in details[k]]
        else:
            template = "{:%is} {}" % KEY_PADDING
            key_val = template.format(key, details[k])
            lines += [(level * LEVEL_OFFSET) + key_val]
    return lines


def setup_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """
    Set up the callbacks form the hydra config.
    """
    callbacks = [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]
    return callbacks


def setup_logger(cfg: DictConfig, name: str = None, evaluate: bool = False) -> Logger:
    """
    Set up the logger (tensorboard or wandb) from hydra config.
    Args:
        cfg: Hydra config
        name: name of the experiment
        evaluate: if the logger is for evaluation
    Returns:
        logger
    """

    date_time = "_".join(cfg.exp_dir.split("/")[-2:])

    if name is None:
        logger_name = "%s_%s" % (cfg.developer, date_time)
    else:
        logger_name = "%s_%s_%s" % (cfg.developer, name, date_time)

    cfg.logger.name = "%s_%s" % (logger_name, cfg.comment) if "comment" in cfg else logger_name

    # TODO: figure out how this should work with slurm
    if evaluate:
        cfg.logger.name += "_eval"

    if cfg.logger["_target_"].split(".")[-1] == "WandbLogger":
        cfg.logger.id = cfg.logger.name.replace("/", "_")

    logger = hydra.utils.instantiate(cfg.logger)

    return logger


def get_all_checkpoints(experiment_folder: Path) -> List:
    if experiment_folder.is_dir():
        checkpoint_folder = experiment_folder / "model_weights"
        if checkpoint_folder.is_dir():
            checkpoints = sorted(Path(checkpoint_folder).iterdir(), key=lambda chk: chk.stat().st_mtime)
            if len(checkpoints):
                return [chk for chk in checkpoints if chk.suffix == ".ckpt"]
    return []


def get_last_checkpoint(experiment_folder: Path) -> Union[Path, None]:
    # return newest checkpoint according to creation time
    checkpoints = get_all_checkpoints(experiment_folder)
    if len(checkpoints):
        return checkpoints[-1]
    return None
