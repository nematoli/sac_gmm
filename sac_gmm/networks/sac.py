# pytorch_sac
#
# Copyright (c) 2019 Denis Yarats
#
# The following code is a derative work from the Denis Yarats,
# which is licensed "MIT License".
#
# Source: https://github.com/denisyarats/pytorch_sac

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from . import Agent
import utils

import hydra
import pdb
from sac_gmm.networks.actor import DiagGaussianActor
from sac_gmm.networks.critic import DoubleQCritic
from sac_gmm.networks.autoencoder import AutoEncoder


class SACAgent(Agent):
    """SAC algorithm."""

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
        super().__init__()
        # pdb.set_trace()
        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        # self.critic = hydra.utils.instantiate(critic).to(self.device)
        self.critic = DoubleQCritic(**critic).to(self.device)
        # self.critic_target = hydra.utils.instantiate(critic).to(self.device)
        self.critic_target = DoubleQCritic(**critic).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # self.actor = hydra.utils.instantiate(actor).to(self.device)
        self.actor = DiagGaussianActor(**actor).to(self.device)
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True

        # self.autoencoder = hydra.utils.instantiate(autoencoder).to(self.device)
        self.autoencoder = AutoEncoder(**autoencoder).to(self.device)

        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=alpha_betas)

        self.ae_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=ae_lr, betas=critic_betas)
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.autoencoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def prepare_obs(self, obs_dict):
        gmm_params = torch.from_numpy(obs_dict["gmm_params"]).to(self.device)
        pose = torch.from_numpy(obs_dict["pose"]).to(self.device)
        image_rep = self.autoencoder.get_image_rep(obs_dict["image"].to(self.device))
        obs = torch.concat((gmm_params, pose, image_rep), axis=-1)
        return obs

    def act(self, obs, sample=False):
        obs = self.prepare_obs(obs)
        obs = obs.type(torch.FloatTensor).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.misc.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        logger.log("train_critic/loss", critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log("train_actor/loss", actor_loss, step)
        logger.log("train_actor/target_entropy", self.target_entropy, step)
        logger.log("train_actor/entropy", -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            logger.log("train_alpha/loss", alpha_loss, step)
            logger.log("train_alpha/value", self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update_autoencoder(self, obs, logger, step):
        hidden = self.autoencoder.encoder(obs)
        pred_obs = self.autoencoder.decoder(hidden)

        rec_loss = F.mse_loss(pred_obs, obs.detach())
        latent_loss = (0.5 * hidden.detach().pow(2).sum(1)).mean()
        ae_loss = rec_loss + self.autoencoder.latent_lambda * latent_loss

        logger.log("train_ae/loss", ae_loss, step)
        logger.log("train_ae/rec_loss", rec_loss, step)
        logger.log("train_ae/latent_loss", latent_loss, step)

        self.ae_optimizer.zero_grad()
        ae_loss.backward()
        self.ae_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, cam_obs, action, reward, next_obs, next_cam_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size
        )

        logger.log("train/batch_reward", reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

            all_cam_obs = torch.concat((cam_obs.unsqueeze(1), next_cam_obs.unsqueeze(1)))
            self.update_autoencoder(all_cam_obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            self.soft_update_critic()

    def soft_update_critic(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.critic_tau * param.data + (1 - self.critic_tau) * target_param.data)
