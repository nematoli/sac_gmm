import os
import sys
import wandb
import hydra
import logging
import csv
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import seed_everything
from sac_gmm.utils.utils import print_system_env_info, setup_logger, get_last_checkpoint
from hydra.core.hydra_config import HydraConfig
from hydra import compose

from sac_gmm.models.sac_gmm_model import SACGMM
from sac_gmm.models.kis_gmm_model import KISGMM


cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, os.path.join(root, "calvin_env"))  # root/calvin_env
sys.path.insert(0, root.as_posix())  # root


logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


@hydra.main(version_base="1.1", config_path="../config", config_name="agent_eval")
def eval_agent(cfg: DictConfig) -> None:
    log_rank_0(
        f"Evaluating {cfg.agent.name} for {cfg.agent.skill.name} skill with the config:\n{OmegaConf.to_yaml(cfg)}"
    )
    log_rank_0(print_system_env_info())

    if cfg.agent.name == "GMM":
        actor = None
        agent = hydra.utils.instantiate(cfg.agent)
    elif cfg.agent.name in ["SACGMM", "KISGMM"]:
        chk = Path(cfg.chk_dir)
        if chk is not None:
            if cfg.agent.name == "SACGMM":
                model = SACGMM
            elif cfg.agent.name == "KISGMM":
                model = KISGMM
            model.load_from_checkpoint(checkpoint_path=chk.as_posix(), agent=cfg.agent).to(cfg.device)
            agent = model.agent
            actor = model.actor
        else:
            raise ValueError("Model not loaded correctly.")

    agent.num_eval_episodes = cfg.num_eval_episodes

    accs = []
    for s in range(cfg.num_eval_seeds):
        seed_everything(cfg.seed + s, workers=True)
        eval_accuracy, eval_return, eval_length = agent.evaluate(actor)
        accs.append(eval_accuracy)

    mean_acc = np.mean(np.array(accs))
    var_acc = np.var(np.array(accs))
    log_rank_0(f"Mean: {mean_acc}, Var: {var_acc}")


if __name__ == "__main__":
    eval_agent()
