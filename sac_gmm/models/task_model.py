import logging
import hydra
from omegaconf import DictConfig
from typing import Dict, Union, Any, List
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.datasets.rl_dataset_task import RLDatasetTask
from torch.utils.data import DataLoader
import wandb

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class TaskModel(pl.LightningModule):
    """
    The lightning module used for training/refining a task.
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
    ):
        super(TaskModel, self).__init__()

        self.discount = discount

        self.batch_size = batch_size
        self.replay_buffer = hydra.utils.instantiate(replay_buffer)

        # Encoder
        self.encoder = hydra.utils.instantiate(encoder)

        # Agent
        self.agent = agent
        self.action_space = self.agent.get_action_space()
        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.agent.get_state_dim(feature_size=self.encoder.feature_size)

        # Actor
        # actor.input_dim = self.state_dim + len(self.agent.task.skills)  # State + Skill Vector
        actor.input_dim = self.state_dim + self.action_dim  # State + Skill Vector (Priors + Means)
        actor.action_dim = self.action_dim
        self.actor = hydra.utils.instantiate(actor)  # .to(self.device)
        self.actor.set_action_space(self.action_space)
        self.actor.set_encoder(self.encoder)

        # Critic
        self.critic_tau = critic_tau
        # critic.input_dim = self.state_dim + len(self.agent.task.skills) + self.action_dim  # State + Skill Vector + Action
        critic.input_dim = (
            self.state_dim + self.action_dim + self.action_dim
        )  # State + Skill Vector (Priors + Means) + Action
        self.critic = hydra.utils.instantiate(critic)  # .to(device)

        self.critic_target = hydra.utils.instantiate(critic)  # .to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy: Set target entropy to -|A|
        self.alpha = init_alpha
        self.log_alpha = nn.Parameter(torch.Tensor([np.log(init_alpha)]), requires_grad=True)
        self.target_entropy = -self.action_dim
        self.fixed_alpha = fixed_alpha

        # Optimizers
        self.critic_lr, self.actor_lr, self.alpha_lr = critic_lr, actor_lr, alpha_lr

        # Populate Replay Buffer with Random Actions
        self.agent.populate_replay_buffer(self.actor, self.replay_buffer)

        # Logic values
        self.episode_idx = 0
        self.episode_return = 0
        self.episode_play_steps = 0
        self.eval_frequency = eval_frequency

        # PyTorch Lightning
        self.automatic_optimization = False

        self.save_hyperparameters()

    def configure_optimizers(self):
        """Initialize optimizers"""
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        optimizers = [critic_optimizer, actor_optimizer, log_alpha_optimizer]

        return optimizers

    def train_dataloader(self):
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDatasetTask(self.replay_buffer, self.batch_size)
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

    def log_video(self, video_path, name: str):
        self.logger.experiment.log({name: wandb.Video(video_path, fps=30, format="gif")})

    def check_batch(self, batch):
        """Verifies if everything is as expected inside a batch"""
        obs, skill_ids, actions, reward, next_obs, next_skill_ids, dones = batch
        # Verifying batch shape
        if len(skill_ids.shape) == 1:
            skill_ids = skill_ids.unsqueeze(-1)
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(-1)
        if len(next_skill_ids.shape) == 1:
            next_skill_ids = next_skill_ids.unsqueeze(-1)
        if len(dones.shape) == 1:
            dones = dones.unsqueeze(-1)

        # Verifying input type
        skill_ids = skill_ids.float()
        reward = reward.float()
        actions = actions.float()
        next_skill_ids = next_skill_ids.float()
        dones = dones.int()
        if not isinstance(obs, dict):
            obs = obs.float()
            next_obs = next_obs.float()

        # Verifying device
        if skill_ids.device != self.device:
            skill_ids = skill_ids.to(self.device)
        if reward.device != self.device:
            reward = reward.to(self.device)
        if actions.device != self.device:
            actions = actions.to(self.device)
        if next_skill_ids.device != self.device:
            next_skill_ids = next_skill_ids.to(self.device)
        if dones.device != self.device:
            dones = dones.to(self.device)
        batch = obs, skill_ids, actions, reward, next_obs, next_skill_ids, dones
        return batch

    def on_save_checkpoint(self, checkpoint_dict):
        checkpoint_dict["episode_idx"] = self.episode_idx

    def on_load_checkpoint(self, checkpoint_dict):
        self.episode_idx = checkpoint_dict["episode_idx"]
