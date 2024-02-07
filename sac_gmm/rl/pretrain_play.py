import logging
from omegaconf import DictConfig
import torch
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl
import wandb
from sac_gmm.datasets.calvin_play_dataset import CALVINPlayDataset
from torch.utils.data import DataLoader
import hydra
import gym
from typing import Dict
from PIL import Image
import numpy as np

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
        target_update_freq: int,
        train_iter: int,
    ):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self._horizon = horizon
        self._pretrain_data_path = pretrain_data_path
        self._pretrain_train_split = pretrain_train_split
        self._n_skill = n_skill
        self.model_cfg = model
        self.target_update_freq = target_update_freq
        self.eval_frequency = eval_frequency
        self.train_iter = train_iter

        self.model_lr = model_lr
        self.model_tau = model_tau
        self.model = hydra.utils.instantiate(model)
        # ob_space = gym.spaces.Dict({"obs": gym.spaces.Box(low=-1, high=1, shape=(model.input_dim,))})
        # ob_space = gym.spaces.Dict({"obs": gym.spaces.Box(low=-1, high=1, shape=(21,))})
        ob_space = gym.spaces.Dict({"obs": gym.spaces.Box(low=-1, high=1, shape=(64, 64, 3))})
        self.model.make_enc_dec(model, ob_space, model.state_dim)
        self.model.make_model_dynamics(model, model.state_dim, self._horizon * 3)
        self.model.to(self.device2)

        self.model_target = hydra.utils.instantiate(model)
        self.model_target.make_enc_dec(model, ob_space, model.state_dim)
        self.model_target.make_model_dynamics(model, model.state_dim, self._horizon * 3)
        self.model_target.load_state_dict(self.model.state_dict())
        self.model_target.to(self.device2)
        # self.train_dataloader()
        self._update_iter = 0
        self.ob_space = ob_space
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        """
        for _ in range(self.train_iter):
            losses = self.loss(batch)
            self.log_loss(losses)

            self._update_iter += 1
            # Update target networks.
            if self._update_iter % self.target_update_freq == 0:
                self.soft_update(self.model_target, self.model, self.model_tau)

    def validation_step(self, batch, batch_idx):
        # Evaluate model on one validation batch
        with torch.no_grad():
            losses = self.loss(batch, is_train=False)
            self.log_loss(losses)

    def loss(self, batch, is_train=True):
        model_optimizer = self.optimizers()
        model_loss = self.compute_model_loss(batch, model_optimizer, is_train)

        if is_train:
            prefix = ""
        else:
            prefix = "val_"

        losses = {
            f"{prefix}losses/reconstruction": model_loss["recon_loss"],
            f"{prefix}losses/consistency": model_loss["consistency_loss"],
            f"{prefix}losses/model": model_loss["model_loss"],
        }
        return losses

    def compute_model_loss(self, batch, model_optimizer, is_train):
        B, H, L = self.batch_size, self._horizon, self._n_skill
        scalars = self.model_cfg.cfg
        mse = torch.nn.MSELoss(reduction="none")

        # ob: Bx(LxH+1)x`ob_dim`, ac: Bx(LxH+1)x`ac_dim`
        ob, ac = torch.squeeze(batch["ob"]), torch.squeeze(batch["ac"])
        o = dict(obs=ob)
        if ac.shape[1] == L * H + 1:
            ac = ac[:, :-1, :3]  # only positional velocity

        with torch.autocast("cuda", enabled=False):
            # Trains skill dynamics model.

            def flip(x, l=None):
                """Flip dimensions, BxT -> TxB."""
                if isinstance(x, dict):
                    return [{k: v[:, t] for k, v in x.items()} for t in range(l)]
                else:
                    return x.transpose(0, 1)

            z = torch.clone(ac).view(B, L, -1)  # stacked actions i.e., (B, L, 3*H)
            hl_o = dict(obs=o["obs"][:, ::H])
            hl_feat = flip(self.model.encoder(hl_o))
            with torch.no_grad():
                hl_feat_target = flip(self.model_target.encoder(hl_o))
            hl_ac = flip(z)

            # HL observation reconstruction loss.
            recon_ob_pred = self.model.decoder(hl_feat)
            recon_losses = {k: -recon_ob_pred[k].log_prob(flip(v)).mean() for k, v in hl_o.items()}
            recon_loss = sum(recon_losses.values())

            # HL latent state consistency loss.
            h = h_next_pred = hl_feat[0]
            consistency_loss = 0
            hs = [h]
            hl_o = flip(hl_o, L + 1)
            for t in range(L):
                h = h_next_pred
                a = hl_ac[t].detach()
                h_next_pred = self.model.imagine_step(h, a)
                h_next_target = hl_feat_target[t + 1]
                rho = scalars.rho**t
                consistency_loss += rho * mse(h_next_pred, h_next_target).mean(dim=1)
                hs.append(h_next_pred)

            model_loss = scalars.model * recon_loss + scalars.consistency * consistency_loss.clamp(max=1e4).mean()
            if is_train:
                model_loss.register_hook(lambda grad: grad * (1 / L))
                model_optimizer.zero_grad()
                self.manual_backward(model_loss)
                # clip gradients
                self.clip_gradients(model_optimizer, gradient_clip_val=100, gradient_clip_algorithm="norm")
                model_optimizer.step()

        model_loss_dict = {}
        model_loss_dict["model_loss"] = model_loss.item()
        model_loss_dict["consistency_loss"] = consistency_loss.mean().item()
        model_loss_dict["recon_loss"] = recon_loss.item()

        # Visualize Decoded Images
        if not is_train and self.ob_space.spaces["obs"].shape[-1] == 3:
            # Log image and decoded image
            rand_idx = torch.randint(0, ob.shape[0], (1,)).item()
            image = ob[rand_idx, 0].detach() * 255.0
            decoded_image = recon_ob_pred["obs"].mean[0, rand_idx].detach() * 255.0
            self.log_image(image, "image/gripper")
            self.log_image(decoded_image, "image/decoded_gripper")

        return model_loss_dict

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

    def log_loss(self, loss: Dict[str, torch.Tensor]):
        for key, val in loss.items():
            if loss[key] != 0:
                self.log(key, loss[key], on_step=True, on_epoch=False)

    def log_image(self, image: torch.Tensor, name: str):
        pil_image = Image.fromarray(image.cpu().numpy().astype(np.uint8))
        self.logger.experiment.log({name: [wandb.Image(pil_image)]})

    @staticmethod
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
