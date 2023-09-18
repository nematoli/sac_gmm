import os
import sys
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import seed_everything
from sac_gmm.utils.utils import print_system_env_info, setup_logger

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


@hydra.main(version_base="1.1", config_path="../../config", config_name="gmm_train")
def train_gmm(cfg: DictConfig) -> None:
    cfg.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    seed_everything(cfg.seed, workers=True)
    log_rank_0(f"Training a GMM skill with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())
    log_rank_0(f"Training gmm for skill {cfg.skill.name} with {cfg.skill.state_type} as the input")

    # Load dataset
    cfg.datamodule.dataset.skill.skill = cfg.datamodule.dataset.skill.name
    train_dataset = hydra.utils.instantiate(cfg.datamodule.dataset)
    log_rank_0(f"Skill: {cfg.skill.name}, Train Data: {train_dataset.X.size()}")

    # Model
    gmm = hydra.utils.instantiate(cfg.gmm)

    # Setup logger
    logger_name = f"{cfg.skill.name}_{cfg.skill.state_type}_{gmm.name}_{gmm.n_components}"
    gmm.logger = setup_logger(cfg, name=logger_name)

    gmm.fit(dataset=train_dataset)
    log_rank_0(f"Training Finished. Trained gmm params are saved in the {gmm.model_dir} directory")


if __name__ == "__main__":
    train_gmm()
