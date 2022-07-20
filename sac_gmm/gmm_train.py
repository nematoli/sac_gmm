import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig

from extract_demos import extract_demos
from utils.env_maker import make_env

from gmm.gmr.gmm import GMM
# from . import ModifiedGMM

import pdb

logger = logging.getLogger(__name__)

class GMMTrainer(object):
    """Python wrapper that allows you to learn a GMM on a specific 
    skill demonstrations extracted from CALVIN dataset.
    """

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.gmm = None
        if cfg.gmm_type == 'base':
            self.gmm = GMM
        else: 
            raise Exception('GMM type not supported')

    def fit_gmm(self, data):
        gmm = self.gmm(n_components=self.cfg.gmm_components, random_state=self.cfg.seed)
        return gmm.from_samples(X=data)

    def evaluate(self, gmm):
        return None


    def run(self):
        # Extract demonstrations
        if self.cfg.use_existing_demos:
            if (Path(self.cfg.demos_dir) / self.cfg.skill / 'train.npy').is_file() &\
                (Path(self.cfg.demos_dir) / self.cfg.skill / 'val.npy').is_file():
                logger.info(f'Using demonstration available at {self.cfg.demos_dir}/{self.cfg.skill}')
            else:
                raise Exception(f'Missing demonstrations at {Path(self.cfg.demos_dir) / self.cfg.skill}')
        else:
            extract_demos(self.cfg)

        # Train a GMM
        train_data = np.load(Path(self.cfg.demos_dir) / self.cfg.skill / 'train.npy')
        train_data = train_data.reshape(1, -1, train_data.shape[-1]).squeeze(0)
        val_data = np.load(Path(self.cfg.demos_dir) / self.cfg.skill / 'val.npy')
        fitted_gmm = self.fit_gmm(train_data)

        # Evaluate in Calvin environment
        self.evaluate(fitted_gmm)


@hydra.main(config_path="../config", config_name="gmm", version_base=None)
def main(cfg: DictConfig) -> None:
    # env = hydra.utils.instantiate(make_env())
    env = None
    trainer = GMMTrainer(cfg, env)
    trainer.run()



if __name__ == '__main__':
    main()