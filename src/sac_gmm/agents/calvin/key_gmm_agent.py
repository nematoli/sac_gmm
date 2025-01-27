import logging
from omegaconf import DictConfig
import torch
import numpy as np
from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.agents.calvin.calvin_agent import CALVINAgent

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class KeyGMMAgent(CALVINAgent):
    def __init__(
        self,
        name: str,
        calvin_env: DictConfig,
        datamodule: DictConfig,
        num_init_steps: int,
        num_eval_episodes: int,
        skill: DictConfig,
        gmm: DictConfig,
        keypoint: DictConfig,
        adapt_per_episode: int,
        render: bool,
    ) -> None:
        super(KeyGMMAgent, self).__init__(
            name=name,
            env=calvin_env,
            datamodule=datamodule,
            num_init_steps=num_init_steps,
            num_eval_episodes=num_eval_episodes,
            skill=skill,
            gmm=gmm,
            keypoint=keypoint,
            adapt_per_episode=adapt_per_episode,
            render=render,
        )

        self.reset()

    @torch.no_grad()
    def detect_target(self, obs, device):
        if self.keypoint.is_mock():
            raise ValueError("KeyGMM must use a Keypoint detector.")

        # keypoint_out = self.keypoint.keypoint(np.zeros(1))
        # objectness = keypoint_out[0][self.keypoint.dim - 1]

        # keypoint_out = self.keypoint.to_world(keypoint_out)
        # keypoint_pos = keypoint_out[0][: self.keypoint.dim - 1]

        # return keypoint_pos + self.kp_target_shift
