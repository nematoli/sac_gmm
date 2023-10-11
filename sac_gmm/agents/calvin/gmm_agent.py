import logging
from omegaconf import DictConfig
import torch
import numpy as np
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.agents.calvin.calvin_agent import CALVINAgent

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class GMMAgent(CALVINAgent):
    def __init__(
        self,
        name: str,
        calvin_env: DictConfig,
        datamodule: DictConfig,
        num_init_steps: int,
        num_eval_episodes: int,
        skill: DictConfig,
        gmm: DictConfig,
        encoder: DictConfig,
        kp_mock: DictConfig,
        render: bool,
    ) -> None:
        super(GMMAgent, self).__init__(
            name=name,
            env=calvin_env,
            datamodule=datamodule,
            num_init_steps=num_init_steps,
            num_eval_episodes=num_eval_episodes,
            skill=skill,
            gmm=gmm,
            encoder=encoder,
            kp_mock=kp_mock,
            render=render,
        )

        self.reset()

    @torch.no_grad()
    def detect_target(self, obs, device):
        keypoint_out = self.kp_mock.keypoint(np.zeros(1))
        # objectness = keypoint_out[0][self.kp_mock.dim - 1]

        keypoint_out = self.kp_mock.to_world(keypoint_out).squeeze()
        keypoint_pos = keypoint_out[: self.kp_mock.dim - 1]

        return keypoint_pos + self.kp_target_shift
