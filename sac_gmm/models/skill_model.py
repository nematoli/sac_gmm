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


logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class SkillModel(pl.LightningModule):
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
        optimize_alpha: bool,
        alpha_lr: float,
        init_alpha: float,
        eval_frequency: int,
    ):
        super(SkillModel, self).__init__()

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
        actor.input_dim = self.state_dim
        actor.action_dim = self.action_dim
        self.actor = hydra.utils.instantiate(actor)  # .to(self.device)
        self.actor.set_action_space(self.action_space)
        self.actor.set_encoder(self.encoder)

        # Critic
        self.critic_tau = critic_tau
        critic.input_dim = self.state_dim + self.action_dim

        self.critic = hydra.utils.instantiate(critic)  # .to(device)

        self.critic_target = hydra.utils.instantiate(critic)  # .to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy: Set target entropy to -|A|
        self.optimize_alpha = optimize_alpha
        self.alpha = init_alpha
        self.log_alpha = nn.Parameter(torch.Tensor([np.log(init_alpha)]), requires_grad=True)
        self.target_entropy = -self.action_dim

        # Optimizers
        self.critic_lr, self.actor_lr, self.alpha_lr = critic_lr, actor_lr, alpha_lr

        # Populate Replay Buffer with Random Actions
        self.agent.populate_replay_buffer(self.actor, self.replay_buffer)

        # Logic values
        self.episode_idx = 0
        self.episode_return = 0
        self.episode_length = 0
        self.eval_frequency = eval_frequency

        # PyTorch Lightning
        self.automatic_optimization = False

        self.save_hyperparameters()

    def configure_optimizers(self):
        """Initialize optimizers"""
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        optimizers = [critic_optimizer, actor_optimizer]
        if self.optimize_alpha:
            log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
            optimizers.append(log_alpha_optimizer)
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
                info = key.split("_")
                self.log(info[0] + "_" + info[1], loss[key], on_step=True, on_epoch=False)

    def log_metrics(self, metrics: Dict[str, torch.Tensor], on_step: bool, on_epoch: bool):
        for key, val in metrics.items():
            info = key.split("_")
            self.log(info[0] + "_" + info[1], metrics[key], on_step=on_step, on_epoch=on_epoch)

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
        if reward.device != self.device:
            reward = reward.to(self.device)
        if actions.device != self.device:
            actions = actions.to(self.device)
        if dones.device != self.device:
            dones = dones.to(self.device)
        batch = obs, actions, reward, next_obs, dones
        return batch

    def on_save_checkpoint(self, checkpoint_dict):
        checkpoint_dict["episode_idx"] = self.episode_idx

    def on_load_checkpoint(self, checkpoint_dict):
        self.episode_idx = checkpoint_dict["episode_idx"]
