import logging
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl
import wandb
from sac_gmm.datasets.calvin_play_dataset import CALVINPlayDataset
from torch.utils.data import DataLoader
from collections import Counter
import hydra
import gym
from typing import Dict

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


OBS_KEY = "rgb_gripper"
# OBS_KEY = "robot_obs"


class PretrainOnPlay(pl.LightningModule):
    """SAC-N-GMM implementation using PyTorch Lightning"""

    def __init__(
        self,
        batch_size: int,
        model: DictConfig,
        model_lr: float,
        model_tau: float,
        eval_frequency: int,
        n_skill: int,
        horizon: int,
        pretrain_data_path: str,
        pretrain_train_split: float,
    ):
        super().__init__()

        self.batch_size = batch_size
        self._horizon = horizon
        self._pretrain_data_path = pretrain_data_path
        self._pretrain_train_split = pretrain_train_split
        self._n_skill = n_skill
        self.model_cfg = model

        self.model_lr = model_lr
        self.model_tau = model_tau
        # self.model = hydra.utils.instantiate(model)
        # # ob_space = gym.spaces.Dict({"obs": gym.spaces.Box(low=-1, high=1, shape=(model.input_dim,))})
        # ob_space = gym.spaces.Dict({"obs": gym.spaces.Box(low=-1, high=1, shape=(21,))})
        # self.model.make_enc_dec(model, ob_space, model.state_dim)
        # self.model.make_model_dynamics(model, model.state_dim, self._horizon * 3)
        # self.model.to(self.device)
        self.train_dataloader()

    def configure_optimizers(self):
        """Initialize optimizers"""
        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_lr)
        optimizers = [model_optimizer]

        return optimizers

    def train_dataloader(self):
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = CALVINPlayDataset(
            self._horizon, self._pretrain_data_path, True, self._pretrain_train_split, self.batch_size
        )
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=0, pin_memory=False)
        return dataloader

    def val_dataloader(self):
        dataset = CALVINPlayDataset(
            self._horizon, self._pretrain_data_path, False, self._pretrain_train_split, self.batch_size
        )
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=0, pin_memory=False)
        return dataloader

    def training_step(self, batch, batch_idx):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        """
        losses = self.loss(batch)
        self.log_loss(losses)
        # self.soft_update(self.model_target, self.model, self.model_tau)

    def evaluation_step(self):
        metrics = {}
        eval_accuracy, eval_return, eval_length, eval_skill_ids, eval_video_paths = self.agent.evaluate(
            self.actor, self.model
        )
        eval_metrics = {
            "eval/accuracy": eval_accuracy,
            "eval/episode-avg-return": eval_return,
            "eval/episode-avg-length": eval_length,
            "eval/total-env-steps": self.agent.total_env_steps,
            "eval/nan-counter": self.agent.nan_counter,
            "eval/episode-number": self.episode_idx,
            # The following are for lightning to save checkpoints
            "accuracy": round(eval_accuracy, 3),
            "episode_number": self.episode_idx,
            "total-env-steps": self.agent.total_env_steps,
        }
        metrics.update(eval_metrics)
        # Log the skill distribution
        if len(eval_skill_ids) > 0:
            skill_id_counts = Counter(eval_skill_ids)
            skill_ids = {
                f"eval/{self.agent.task.skills[k]}": v / self.agent.num_eval_episodes
                for k, v in skill_id_counts.items()
            }
            # Add 0 values for skills that were not used at all
            unused_skill_ids = set(range(len(self.agent.task.skills))) - set(skill_id_counts.keys())
            if len(unused_skill_ids) > 0:
                skill_ids.update({f"eval/{self.agent.task.skills[k]}": 0 for k in list(unused_skill_ids)})
        else:
            skill_ids = {f"eval/{k}": 0 for k in self.agent.task.skills}
        metrics.update(skill_ids)
        # Log the video GIF to wandb if exists
        return metrics, eval_video_paths

    def loss(self, batch):
        model_optimizer = self.optimizers()[0]
        model_loss = self.compute_model_loss(batch, model_optimizer)

        losses = {
            "losses/reconstruction": model_loss["recon_loss"],
            "losses/consistency": model_loss["consistency_loss"],
            "losses/model": model_loss["model_loss"],
        }
        return losses

    def compute_model_loss(self, batch, model_optimizer):

        B, H, L = self.batch_size, self._horizon, self._n_skill
        scalars = self.model_cfg.cfg
        mse = torch.nn.MSELoss(reduction="none")

        batch_obs = batch["ob"]
        # ob: Bx(LxH+1)x`ob_dim`, ac: Bx(LxH+1)x`ac_dim`
        ob, ac = batch["ob"], batch["ac"]

        # Reconstruction Loss
        enc_state = self.model.encoder({"obs": batch_obs[OBS_KEY].float()})
        recon_obs = self.model.decoder(enc_state)
        recon_loss = -recon_obs["obs"].log_prob(batch_obs[OBS_KEY].float()).mean()

        model_optimizer.zero_grad()
        self.manual_backward(recon_loss)
        model_optimizer.step()

        model_loss_dict = {}
        model_loss_dict["recon_loss"] = recon_loss

        # Visualize Decoded Images
        if OBS_KEY == "rgb_gripper" and self.episode_done and (self.episode_idx % self.eval_frequency == 0):
            # Log image and decoded image
            rand_idx = torch.randint(0, batch_obs[OBS_KEY].shape[0], (1,)).item()
            image = batch_obs[OBS_KEY][rand_idx].detach()
            decoded_image = recon_obs["obs"].mean[rand_idx].detach()
            self.log_image(image, "eval/gripper")
            self.log_image(decoded_image, "eval/decoded_gripper")
        return model_loss_dict

    def log_loss(self, loss: Dict[str, torch.Tensor]):
        for key, val in loss.items():
            if loss[key] != 0:
                self.log(key, loss[key], on_step=True, on_epoch=False)

    @staticmethod
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
