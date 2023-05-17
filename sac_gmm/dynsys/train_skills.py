import os
import sys
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, os.path.join(root, "calvin_env"))  # root/calvin_env
sys.path.insert(0, root.as_posix())  # root


class SkillTrainer(object):
    """Python wrapper that allows you to train DS skills on a given dataset"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.state_type = self.cfg.state_type
        self.logger = logging.getLogger("SkillTrainer")
        self.ds_out_dir = None
        # Make skills directory if doesn't exist
        os.makedirs(self.cfg.skills_dir, exist_ok=True)

    def run(self):
        f = open(self.cfg.skills_list, "r")
        skill_set = f.read()
        skill_set = skill_set.split("\n")
        self.logger.info(f"Found {len(skill_set)} skills in the list")
        self.logger.info(f"Training DS with {self.state_type} as the input")
        for idx, skill in enumerate(skill_set):
            # Load dataset
            self.cfg.dataset.skill = skill
            self.cfg.dataset.train = True
            train_dataset = hydra.utils.instantiate(self.cfg.dataset)
            self.cfg.dataset.train = False
            val_dataset = hydra.utils.instantiate(self.cfg.dataset)
            self.logger.info(
                f"Skill {idx}: {skill}, Train Data: {train_dataset.X.size()}, Val. Data: {val_dataset.X.size()}"
            )
            self.cfg.dim = train_dataset.X.shape[-1]
            ds = hydra.utils.instantiate(self.cfg.dyn_sys)
            # Make output dir where trained models will be saved
            ds.model_dir = os.path.join(self.cfg.skills_dir, self.state_type, skill, ds.name)
            os.makedirs(ds.model_dir, exist_ok=True)
            ds.fit(dataset=train_dataset)
        self.logger.info(
            f"Training complete. Trained DS models are saved in the {os.path.join(ds.model_dir)} directory"
        )


@hydra.main(version_base="1.1", config_path="../../config", config_name="train_ds")
def main(cfg: DictConfig) -> None:
    eval = SkillTrainer(cfg)
    eval.run()


if __name__ == "__main__":
    main()
