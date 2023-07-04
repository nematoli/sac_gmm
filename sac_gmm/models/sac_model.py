import logging
import hydra
from omegaconf import DictConfig
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.models.skill import SkillModel
from sac_gmm.datasets.replay_buffer import ReplayBuffer


logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class SAC(SkillModel):
    """Basic SAC implementation using PyTorch Lightning"""

    def __init__(
        self,
        discount: float,
        batch_size: int,
        replay_buffer: DictConfig,
        agent: DictConfig,
        actor: DictConfig,
        critic: DictConfig,
        actor_lr: float,
        actor_betas: List[float],
        critic_lr: float,
        critic_tau: float,
        critic_betas: List[float],
        optimize_alpha: bool,
        alpha_lr: float,
        init_alpha: float,
        alpha_betas: List[float],
        eval_frequency: int,
    ):
        super(SAC, self).__init__(
            discount=discount,
            batch_size=batch_size,
            replay_buffer=replay_buffer,
            agent=agent,
            actor=actor,
            critic=critic,
            actor_lr=actor_lr,
            actor_betas=actor_betas,
            critic_lr=critic_lr,
            critic_tau=critic_tau,
            critic_betas=critic_betas,
            optimize_alpha=optimize_alpha,
            alpha_lr=alpha_lr,
            init_alpha=init_alpha,
            alpha_betas=alpha_betas,
            eval_frequency=eval_frequency,
        )

    def training_step(self, batch, batch_idx):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        """
        # log_rank_0(f"this is step: {batch_idx}")

        reward, self.episode_done = self.agent.play_step(self.actor, "stochastic", self.replay_buffer)
        self.episode_return += reward
        self.episode_length += 1

        losses = self.loss(self.check_batch(batch))
        self.log_loss(losses)
        self.soft_update(self.critic_target, self.critic, self.critic_tau)

        if self.episode_done:
            # When an episode ends, log episode metrics
            metrics = {
                "train/episode_return": self.episode_return,
                "train/episode_length": self.episode_length,
                "train/episode_number": self.episode_idx.item(),
            }
            # eval_return = float("-inf")
            # eval_accuracy = float("-inf")
            if self.episode_idx % self.eval_frequency == 0:
                eval_accuracy, eval_return, eval_length, _ = self.agent.evaluate(self.actor)
                eval_metrics = {
                    "eval/accuracy": eval_accuracy,
                    "eval/episode_avg_return": eval_return,
                    "eval/episode_avg_length": eval_length,
                    "eval/total_env_steps": self.agent.total_env_steps,
                }
                metrics.update(eval_metrics)

            self.log_metrics(metrics, on_step=True, on_epoch=False)
            self.episode_return, self.episode_length = 0, 0
            self.episode_idx += 1

            # Save the replay buffer when you evaluate
            self.replay_buffer.save()

        # # Monitored metric to save model
        # self.log("eval_episode_return", eval_return, on_epoch=True)
        # self.log("eval_accuracy", eval_accuracy, on_epoch=True)

    # def on_train_epoch_end(self):
    #     log_rank_0("this was one epoch")

    #     """
    #     This function is called every time a training epoch ends.
    #     One training epoch = replay_buffer size / batch_size iterations
    #     It evaluates actor (when appropriate) by simulating
    #     in the environment.
    #     """
    #     eval_return = float("-inf")
    #     eval_accuracy = float("-inf")
    #     if self.episode_idx % self.eval_frequency == 0:
    #         eval_accuracy, eval_return, eval_length, _ = self.agent.evaluate(self.actor)
    #         metrics = {
    #             "eval/accuracy": eval_accuracy,
    #             "eval/episode_avg_return": eval_return,
    #             "eval/episode_avg_length": eval_length,
    #             "eval/total_env_steps": self.agent.total_env_steps,
    #         }

    #         self.log_metrics(metrics, on_step=False, on_epoch=True)
    #         self.episode_return, self.episode_length = 0, 0
    #         self.episode_idx += 1

    #     # Save the replay buffer when you evaluate
    #     self.replay_buffer.save()
    #     # Monitored metric to save model
    #     self.log("eval_episode_return", eval_return, on_epoch=True)
    #     self.log("eval_accuracy", eval_accuracy, on_epoch=True)

    def loss(self, batch):
        critic_optimizer, actor_optimizer, alpha_optimizer = self.optimizers()
        critic_loss = self.compute_critic_loss(batch, critic_optimizer)
        actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(batch, actor_optimizer, alpha_optimizer)

        losses = {
            "loss/critic": critic_loss,
            "loss/actor": actor_loss,
            "loss/alpha": alpha_loss,
            "alpha/value": self.alpha,
        }
        return losses

    def compute_critic_loss(self, batch, critic_optimizer):
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        with torch.no_grad():
            policy_actions, log_pi = self.actor.get_actions(
                batch_next_obs,
                deterministic=False,
                reparameterize=False,
            )
            q1_next_target, q2_next_target = self.critic_target(batch_next_obs, policy_actions)
            q_next_target = torch.min(q1_next_target, q2_next_target)
            q_target = batch_rewards + (1 - batch_dones) * self.discount * (q_next_target - self.alpha * log_pi)

        # Bellman loss
        q1_pred, q2_pred = self.critic(batch_obs, batch_actions.float())
        bellman_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        critic_optimizer.zero_grad()
        self.manual_backward(bellman_loss)
        critic_optimizer.step()

        return bellman_loss

    def compute_actor_and_alpha_loss(self, batch, actor_optimizer, alpha_optimizer):
        batch_obs = batch[0]
        policy_actions, log_pi = self.actor.get_actions(batch_obs, deterministic=False, reparameterize=True)
        q1, q2 = self.critic(batch_obs, policy_actions)
        Q_value = torch.min(q1, q2)
        actor_loss = (self.alpha * log_pi - Q_value).mean()

        actor_optimizer.zero_grad()
        self.manual_backward(actor_loss)
        actor_optimizer.step()

        self.log_alpha = self.log_alpha.to(log_pi.device)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        if self.optimize_alpha:
            alpha_optimizer.zero_grad()
            self.manual_backward(alpha_loss)
            alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        return actor_loss, alpha_loss
