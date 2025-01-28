import logging
import hydra
from omegaconf import DictConfig
from typing import Dict, Union, Any, List
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from sac_n_gmm.datasets.rl_dataset_task import RLDatasetTask
from torch.utils.data import DataLoader
import wandb
import gym
from PIL import Image

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


OBS_KEY = "rgb_gripper"
# OBS_KEY = "robot_obs"


class TaskRL(pl.LightningModule):
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
        model: DictConfig,
        actor_lr: float,
        critic_lr: float,
        critic_tau: float,
        alpha_lr: float,
        init_alpha: float,
        fixed_alpha: bool,
        model_lr: float,
        model_tau: float,
        eval_frequency: int,
    ):
        super().__init__()

        self.discount = discount

        self.batch_size = batch_size
        self.replay_buffer = hydra.utils.instantiate(replay_buffer)

        # Encoder
        # self.encoder = hydra.utils.instantiate(encoder)
        self.encoder = None

        # Agent
        self.agent = agent
        self.action_space = self.agent.get_action_space()
        self.action_dim = self.action_space.shape[0]

        # Skill Conditioning Vector (use skill means and priors for now)
        self.skill_vector_size = self.agent.skill_actor.means_size + self.agent.skill_actor.priors_size
        # self.skill_vector_size = len(self.agent.task.skills) # If using one-hot encoding

        # Model only has autoencoder for now (Input encoder, State decoder)
        self.model = hydra.utils.instantiate(model)
        # ob_space = gym.spaces.Dict({"obs": gym.spaces.Box(low=-1, high=1, shape=(model.input_dim,))})
        ob_space = gym.spaces.Dict({"obs": self.agent.env.get_observation_space()[OBS_KEY]})
        self.model.make_enc_dec(model, ob_space, model.state_dim)
        self.model.to(self.device)

        # Actor
        actor.input_dim = model.state_dim + self.skill_vector_size
        actor.action_dim = self.action_dim
        self.actor = hydra.utils.instantiate(actor).to(self.device)
        self.actor.set_action_space(self.action_space)
        # self.actor.set_encoder(self.encoder)

        # Critic
        self.critic_tau = critic_tau
        critic.input_dim = model.state_dim + self.skill_vector_size + self.action_dim
        self.critic = hydra.utils.instantiate(critic).to(self.device)

        self.critic_target = hydra.utils.instantiate(critic).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy: Set target entropy to -|A|
        self.alpha = init_alpha
        self.log_alpha = nn.Parameter(torch.Tensor([np.log(init_alpha)]), requires_grad=True)
        self.target_entropy = -self.action_dim
        self.fixed_alpha = fixed_alpha

        # Optimizers
        self.critic_lr, self.actor_lr, self.alpha_lr = critic_lr, actor_lr, alpha_lr
        self.model_lr, self.model_tau = model_lr, model_tau

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

    def log_metrics_and_videos(self, metrics, video_paths):
        self.log_metrics(metrics, on_step=False, on_epoch=True)

        # Log the video GIF to wandb if exists
        if video_paths is not None and len(video_paths.keys()) > 0:
            for skill_name, video_path in video_paths.items():
                self.log_video(video_path, f"eval/{skill_name}_video")

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
