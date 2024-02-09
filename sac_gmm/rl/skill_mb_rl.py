import logging
import hydra
from omegaconf import DictConfig
from typing import Dict, Union, Any, List
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.datasets.rl_dataset import RLDataset
from torch.utils.data import DataLoader
import wandb
import gym
from PIL import Image


logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


OBS_KEY = "rgb_static"
# OBS_KEY = "rgb_gripper"
# OBS_KEY = "robot_obs"


class SkillMBRL(pl.LightningModule):
    """
    The lightning module used for training/refining a skill.
    Args:
    """

    def __init__(
        self,
        discount: float,
        batch_size: int,
        replay_buffer: DictConfig,
        encoder: DictConfig,
        agent,
        actor: DictConfig,
        critic: DictConfig,
        actor_lr: float,
        critic_lr: float,
        critic_tau: float,
        alpha_lr: float,
        init_alpha: float,
        fixed_alpha: bool,
        eval_frequency: int,
        model: DictConfig,
        model_lr: float,
        model_tau: float,
        horizon: int,
    ):
        super(SkillMBRL, self).__init__()

        self.device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = discount
        self.horizon = horizon

        self.batch_size = batch_size
        self.replay_buffer = hydra.utils.instantiate(replay_buffer)

        # Encoder
        # self.encoder = hydra.utils.instantiate(encoder)
        self.encoder = None

        # Agent
        self.agent = agent
        self.action_space = self.agent.get_action_space()
        self.action_dim = self.action_space.shape[0]
        self.agent.gmm_window = horizon

        # Model
        self.model = hydra.utils.instantiate(model)
        ob_space = gym.spaces.Dict({"obs": self.agent.env.get_observation_space()[OBS_KEY]})
        self.model.make_enc_dec(model, ob_space, model.state_dim)
        self.model.make_model_dynamics(model, model.state_dim, horizon * 3)
        self.model.make_model_reward(model, model.state_dim, self.action_dim)
        self.model.encoder.requires_grad_(False)
        self.model.to(self.device2)

        self.model_target = hydra.utils.instantiate(model)
        self.model_target.make_enc_dec(model, ob_space, model.state_dim)
        self.model_target.make_model_dynamics(model, model.state_dim, horizon * 3)
        self.model_target.make_model_reward(model, model.state_dim, self.action_dim)
        self.model_target.load_state_dict(self.model.state_dict())
        self.model_target.encoder.requires_grad_(False)
        self.model_target.to(self.device2)

        # Actor
        actor.input_dim = model.state_dim
        actor.action_dim = self.action_dim
        self.actor = hydra.utils.instantiate(actor).to(self.device2)
        self.actor.set_action_space(self.action_space)
        # self.actor.set_encoder(self.encoder)

        # Critic
        self.critic_tau = critic_tau
        critic.input_dim = model.state_dim + self.action_dim
        self.critic = hydra.utils.instantiate(critic).to(self.device2)

        self.critic_target = hydra.utils.instantiate(critic).to(self.device2)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy: Set target entropy to -|A|
        self.alpha = init_alpha
        self.log_alpha = nn.Parameter(torch.Tensor([np.log(init_alpha)]), requires_grad=True)
        self.target_entropy = -self.action_dim
        self.fixed_alpha = fixed_alpha

        # Optimizers
        self.critic_lr, self.actor_lr, self.alpha_lr = critic_lr, actor_lr, alpha_lr
        self.model_lr, self.model_tau = model_lr, model_tau

        # Populate Replay Buffer with Random Actions
        self.agent.populate_replay_buffer(self.actor, self.model, self.replay_buffer)

        # Logic values
        self.episode_idx = 0
        self.episode_return = 0
        self.episode_play_steps = 0
        self.eval_frequency = eval_frequency

        # PyTorch Lightning
        self.automatic_optimization = False
        self.save_hyperparameters()

        # Torch compile
        # self.actor = torch.compile(self.actor)
        # self.critic = torch.compile(self.critic)
        # self.critic_target = torch.compile(self.critic_target)
        # self.model = torch.compile(self.model)
        # self.model_target = torch.compile(self.model_target)

    def configure_optimizers(self):
        """Initialize optimizers"""
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_lr)
        optimizers = [critic_optimizer, actor_optimizer, log_alpha_optimizer, model_optimizer]

        return optimizers

    def train_dataloader(self):
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.replay_buffer, self.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=0, pin_memory=True)
        return dataloader

    @staticmethod
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def forward(self) -> None:
        """
        Main forward pass for at each step.
        Args:
        Returns:
        """
        raise NotImplementedError

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Compute and return the training loss.
        Args:
        Returns:
        """

        raise NotImplementedError

    def loss(self, batch) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def log_loss(self, loss: Dict[str, torch.Tensor]):
        for key, val in loss.items():
            if loss[key] != 0:
                self.log(key, loss[key], on_step=True, on_epoch=False)

    def log_metrics(self, metrics: Dict[str, torch.Tensor], on_step: bool, on_epoch: bool):
        for key, val in metrics.items():
            self.log(key, val, on_step=on_step, on_epoch=on_epoch)

    def log_image(self, image: torch.Tensor, name: str):
        pil_image = Image.fromarray(image.cpu().numpy().astype(np.uint8))
        self.logger.experiment.log({name: [wandb.Image(pil_image)]})

    def log_video(self, video_path, name: str):
        self.logger.experiment.log({name: wandb.Video(video_path, fps=30, format="gif")})

    def check_batch(self, batch):
        """Verifies if everything is as expected inside a batch"""
        obs, actions, reward, next_obs, dones = batch
        # Verifying batch shape
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(-1)
        if len(dones.shape) == 1:
            dones = dones.unsqueeze(-1)

        # Verifying input type
        reward = reward.float()
        actions = actions.float()
        dones = dones.int()
        if not isinstance(obs, dict):
            obs = obs.float()
            next_obs = next_obs.float()

        # Verifying device
        if reward.device != self.device2:
            reward = reward.to(self.device2)
        if actions.device != self.device2:
            actions = actions.to(self.device2)
        if dones.device != self.device2:
            dones = dones.to(self.device2)
        batch = obs, actions, reward, next_obs, dones
        return batch

    def on_save_checkpoint(self, checkpoint_dict):
        checkpoint_dict["episode_idx"] = self.episode_idx

    def on_load_checkpoint(self, checkpoint_dict):
        self.episode_idx = checkpoint_dict["episode_idx"]
