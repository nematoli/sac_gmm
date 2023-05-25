import torch
import torch.nn.functional as F

from sac_gmm.networks.sac import SACAgent
from sac_gmm.networks.autoencoder import AutoEncoder


class SACGMMAgent(SACAgent):
    """SACGMM algorithm."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        discount,
        init_temperature,
        alpha_lr,
        alpha_betas,
        actor_lr,
        actor_betas,
        actor_update_frequency,
        critic_lr,
        critic_betas,
        critic_tau,
        critic_target_update_frequency,
        batch_size,
        learnable_temperature,
        ae_lr,
        critic,
        actor,
        autoencoder,
    ):
        super().__init__(
            obs_dim,
            action_dim,
            action_range,
            device,
            discount,
            init_temperature,
            alpha_lr,
            alpha_betas,
            actor_lr,
            actor_betas,
            actor_update_frequency,
            critic_lr,
            critic_betas,
            critic_tau,
            critic_target_update_frequency,
            batch_size,
            learnable_temperature,
            critic,
            actor,
        )
        # self.ae = hydra.utils.instantiate(autoencoder).to(self.device)
        self.ae = AutoEncoder(**autoencoder).to(self.device)
        self.ae_optimizer = torch.optim.Adam(self.ae.parameters(), lr=ae_lr, betas=critic_betas)
        self.train()

    def train(self, training=True):
        super().train(training)
        self.ae.train(training)

    def update_autoencoder(self, obs, logger, step):
        hidden = self.ae.encoder(obs)
        pred_obs = self.ae.decoder(hidden)

        rec_loss = F.mse_loss(pred_obs, obs.detach())
        latent_loss = (0.5 * hidden.detach().pow(2).sum(1)).mean()
        ae_loss = rec_loss + self.ae.latent_lambda * latent_loss

        logger.log("train_ae/loss", ae_loss, step)
        logger.log("train_ae/rec_loss", rec_loss, step)
        logger.log("train_ae/latent_loss", latent_loss, step)

        self.ae_optimizer.zero_grad()
        ae_loss.backward()
        self.ae_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, cam_obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)

        logger.log("train/batch_reward", reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

            self.update_autoencoder(cam_obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            self.soft_update_critic()
