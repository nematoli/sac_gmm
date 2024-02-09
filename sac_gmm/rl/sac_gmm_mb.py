import logging
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.rl.skill_mb_rl import SkillMBRL
import os
from sac_gmm.gmm.batch_gmm import BatchGMM
import numpy as np


logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


OBS_KEY = "rgb_static"
# OBS_KEY = "rgb_gripper"
# OBS_KEY = "robot_obs"


class SACGMM_MB(SkillMBRL):
    """Basic SAC-GMM implementation using PyTorch Lightning"""

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
        model_lr: float,
        model_tau: float,
        model: DictConfig,
        model_ckpt: str,
        horizon: int,
    ):
        super(SACGMM_MB, self).__init__(
            discount=discount,
            batch_size=batch_size,
            replay_buffer=replay_buffer,
            encoder=encoder,
            agent=agent,
            actor=actor,
            critic=critic,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            critic_tau=critic_tau,
            alpha_lr=alpha_lr,
            init_alpha=init_alpha,
            fixed_alpha=fixed_alpha,
            eval_frequency=eval_frequency,
            model_lr=model_lr,
            model_tau=model_tau,
            model=model,
            horizon=horizon,
        )
        self.episode_done = False
        self.save_hyperparameters()
        self.max_env_steps = None

        self.load_checkpoint(model_ckpt, agent.root_dir)
        self.n_skill = 1
        self.scalars = self.model.cfg.cfg

    def training_step(self, batch, batch_idx):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        """
        reward, self.episode_done = self.agent.play_step_with_critic(
            self.actor, self.model, self.critic, "cem", self.replay_buffer, self.device2
        )
        self.episode_return += reward
        self.episode_play_steps += 1

        losses = self.loss(self.check_batch(batch))
        self.log_loss(losses)
        self.soft_update(self.critic_target, self.critic, self.critic_tau)

    def evaluation_step(self):
        metrics = {}
        eval_accuracy, eval_return, eval_length, eval_video_path = self.agent.evaluate(self.actor, self.model)
        eval_metrics = {
            "eval/accuracy": eval_accuracy,
            "eval/episode-avg-return": eval_return,
            "eval/episode-avg-length": eval_length,
            "eval/total-env-steps": self.agent.total_env_steps,
            "eval/episode-number": self.episode_idx,
            # The following are for lightning to save checkpoints
            "accuracy": round(eval_accuracy, 3),
            "episode_number": self.episode_idx,
            "total-env-steps": self.agent.total_env_steps,
        }
        metrics.update(eval_metrics)

        return metrics, eval_video_path

    def on_train_epoch_end(self):
        if self.episode_done:
            metrics = {"eval/accuracy": float("-inf")}
            metrics = {"accuracy": float("-inf")}
            log_rank_0(f"Episode Done: {self.episode_idx}")
            train_metrics = {
                "train/episode-return": self.episode_return,
                "train/episode-play-steps": self.episode_play_steps,
                "train/episode-length": self.episode_play_steps * self.agent.gmm_window,
                "train/episode-number": self.episode_idx,
                "train/nan-counter": self.agent.nan_counter,
                "train/total-env-steps": self.agent.total_env_steps,
            }
            metrics.update(train_metrics)

            if self.episode_idx % self.eval_frequency == 0:
                eval_metrics, eval_video_path = self.evaluation_step()
                metrics.update(eval_metrics)
                if eval_video_path is not None:
                    self.log_video(eval_video_path, name="eval/video")

            self.episode_return, self.episode_play_steps = 0, 0
            self.episode_idx += 1
            self.replay_buffer.save()

            # Programs exits when maximum env steps is reached
            # Before exiting, logs the evaluation metrics and videos
            if self.agent.total_env_steps > self.max_env_steps:
                eval_metrics, eval_video_path = self.evaluation_step()
                metrics.update(eval_metrics)
                self.log_metrics(metrics, on_step=False, on_epoch=True)
                if eval_video_path is not None:
                    self.log_video(eval_video_path, name="eval/video")
                raise KeyboardInterrupt
            self.log_metrics(metrics, on_step=False, on_epoch=True)

    def loss(self, batch):
        critic_optimizer, actor_optimizer, alpha_optimizer, model_optimizer = self.optimizers()
        critic_loss = self.compute_critic_loss(batch, critic_optimizer)
        actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(batch, actor_optimizer, alpha_optimizer)
        model_loss = self.compute_model_loss(batch, model_optimizer)

        losses = {
            "losses/critic": critic_loss,
            "losses/actor": actor_loss,
            "losses/alpha": alpha_loss,
            "losses/alpha_value": self.alpha,
            # "losses/reconstruction": model_loss["recon_loss"],
            "losses/model": model_loss["model_loss"],
            "losses/consistency": model_loss["consistency"],
            "losses/reward": model_loss["reward"],
        }
        return losses

    def compute_critic_loss(self, batch, critic_optimizer):
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        with torch.no_grad():
            state = self.model.encoder({"obs": batch_next_obs[OBS_KEY].float()}).squeeze(0)
            policy_actions, log_pi = self.actor.get_actions(state, deterministic=False, reparameterize=False)
            q1_next_target, q2_next_target = self.critic_target(state, policy_actions)

            q_next_target = torch.min(q1_next_target, q2_next_target)
            q_target = batch_rewards + (1 - batch_dones) * self.discount * (q_next_target - self.alpha * log_pi)

        # Bellman loss
        with torch.no_grad():
            state = self.model.encoder({"obs": batch_obs[OBS_KEY].float()}).squeeze(0)
        q1_pred, q2_pred = self.critic(state, batch_actions.float())
        bellman_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        critic_optimizer.zero_grad()
        self.manual_backward(bellman_loss)
        critic_optimizer.step()

        return bellman_loss

    def compute_actor_and_alpha_loss(self, batch, actor_optimizer, alpha_optimizer):
        batch_obs = batch[0]
        with torch.no_grad():
            state = self.model.encoder({"obs": batch_obs[OBS_KEY].float()}).squeeze(0)
        policy_actions, log_pi = self.actor.get_actions(state, deterministic=False, reparameterize=True)
        q1, q2 = self.critic(state, policy_actions)
        Q_value = torch.min(q1, q2)
        actor_loss = (self.alpha * log_pi - Q_value).mean()

        actor_optimizer.zero_grad()
        self.manual_backward(actor_loss)
        actor_optimizer.step()

        if not self.fixed_alpha:
            self.log_alpha = self.log_alpha.to(log_pi.device)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            self.manual_backward(alpha_loss)
            alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0)

        return actor_loss, alpha_loss

    def compute_model_loss(self, batch, model_optimizer):

        mse = torch.nn.MSELoss(reduction="none")
        # Reconstruction Loss
        # enc_state = self.model.encoder({"obs": batch_obs[OBS_KEY].float()})
        # recon_obs = self.model.decoder(enc_state)
        # recon_loss = -recon_obs["obs"].log_prob(batch_obs[OBS_KEY].float()).mean()
        robot_obs = batch[0]["robot_obs"].float()
        rgb_ob = batch[0][OBS_KEY].float()
        rv = batch[1].float()
        rew = batch[2].float()
        next_rgb_ob = batch[3][OBS_KEY].float()
        done = batch[4].float()

        # Get refined stacked actions BxT
        ac = batch_actions(
            self.agent.initial_gmm,
            robot_obs[:, :3].cpu(),
            rv,
            self.horizon,
        ).to(self.device2)

        # Flip dimensions, BxT -> TxB
        def flip(x, l=None):
            if isinstance(x, dict):
                return [{k: v[:, t] for k, v in x.items()} for t in range(l)]
            else:
                return x.transpose(0, 1)

        # obs = torch.concatenate([rgb_ob.unsqueeze(0), next_rgb_ob.unsqueeze(0)], dim=0)
        state = self.model.encoder({"obs": rgb_ob})
        state_next = self.model.encoder({"obs": next_rgb_ob})
        states = [state, state_next]
        # Avoid gradients for the model target
        with torch.no_grad():
            state_target = self.model_target.encoder({"obs": rgb_ob})
            state_next_target = self.model_target.encoder({"obs": next_rgb_ob})
            states_target = [state_target, state_next_target]

        # Trians skill dynamics model.
        z = z_next_pred = states[0]
        rewards = []

        consistency_loss = 0
        reward_loss = 0
        value_loss = 0
        q_preds = [[], []]
        q_targets = []

        # for t in range(self.n_skill):
        z = z_next_pred
        z_next_pred = self.model.imagine_step(z, ac)
        reward_pred = self.model.imagine_reward(z, rv)
        q_pred = self.critic(z, rv)

        with torch.no_grad():
            # `z` for contrastive learning
            z_next = states_target[1]

            # `z` for `q_target`
            z_next_q = states[1]
            # TODO: Talk to Iman. This is not possible since
            # we need obs_next for meta action of next state
            # And what should be the skill_id
            rv_next_mean, rv_next_std = self.actor(z_next_q)
            rv_next, _ = self.actor.sample_actions(rv_next_mean, rv_next_std, reparameterize=True)
            q_next = torch.min(*self.critic(z_next_q, rv_next))

            q_target = rew + (1 - done.long()) * self.discount * q_next
        rewards.append(reward_pred.detach())
        q_preds[0].append(q_pred[0].detach())
        q_preds[1].append(q_pred[1].detach())
        q_targets.append(q_target)

        rho = self.scalars.rho
        consistency_loss += rho * mse(z_next_pred, z_next).mean(dim=1)
        reward_loss += rho * mse(reward_pred, rew)
        value_loss += rho * (mse(q_pred[0], q_target) + mse(q_pred[1], q_target))

        # Additional reward prediction loss.
        reward_pred = self.model.reward(torch.cat([state, rv], dim=-1))
        reward_loss = reward_loss + mse(reward_pred, rew)
        # # Additional value prediction loss.
        # q_pred = self.critic(state, rv)
        # value_loss += mse(q_pred[0], q_target) + mse(q_pred[1], q_target)

        model_loss = (
            self.scalars.consistency * consistency_loss.clamp(max=1e5)
            + self.scalars.reward * reward_loss.clamp(max=1e5) * 0.5
            # + self.scalars.value * value_loss.clamp(max=1e5)
        ).mean()
        model_loss.register_hook(lambda grad: grad * (1 / self.n_skill))

        model_optimizer.zero_grad()
        self.manual_backward(model_loss)
        # clip gradients
        self.clip_gradients(model_optimizer, gradient_clip_val=100, gradient_clip_algorithm="norm")
        model_optimizer.step()

        model_loss_dict = {}
        model_loss_dict["model_loss"] = model_loss
        model_loss_dict["consistency"] = consistency_loss.mean()
        model_loss_dict["reward"] = reward_loss.mean()
        # model_loss_dict["recon_loss"] = recon_loss

        # Visualize Decoded Images
        # if "rgb" in OBS_KEY and self.episode_done and (self.episode_idx % self.eval_frequency == 0):
        #     # Log image and decoded image
        #     rand_idx = torch.randint(0, batch_obs[OBS_KEY].shape[0], (1,)).item()
        #     image = batch_obs[OBS_KEY][rand_idx].detach() * 255.0
        #     decoded_image = recon_obs["obs"].mean[rand_idx].detach() * 255.0
        #     self.log_image(image, f"eval/{OBS_KEY}")
        #     self.log_image(decoded_image, f"eval/decoded_{OBS_KEY}")
        return model_loss_dict

    def load_checkpoint(self, model_ckpt, root_dir):
        """Load pretrained model weights"""

        ckpt = torch.load(os.path.join(root_dir, model_ckpt))
        log_rank_0(f"Loading model from {model_ckpt}")

        # Get only model related state_dict
        model_state_dict = {
            k.replace("model.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.replace("model.", "") in self.model.state_dict()
        }
        self.model.load_state_dict(model_state_dict)


def batch_actions(gmm, batch_x, batch_rv, skill_horizon):
    """
    Batch acts skill_horizon times
    """
    batch_size = batch_x.shape[0]
    out = torch.zeros((batch_size, skill_horizon, batch_x.shape[1]))

    batch_priors = np.repeat(np.expand_dims(gmm.priors, 0), batch_size, axis=0)
    batch_means = np.repeat(np.expand_dims(gmm.means, 0), batch_size, axis=0)
    batch_covariances = np.repeat(np.expand_dims(gmm.covariances, 0), batch_size, axis=0)
    # Batch refine (only means)
    batch_means += batch_rv.cpu().numpy().reshape(batch_means.shape) * 0.05

    # Batch predict
    for i in range(skill_horizon):
        batch_dx = batch_predict(
            gmm.n_components,
            batch_priors,
            batch_means,
            batch_covariances,
            batch_x,
            gmm.random_state,
        )
        out[:, i, :] = batch_dx
        batch_x += batch_dx * gmm.pos_dt

    return out.reshape((batch_size, -1))


def batch_predict(n_components, batch_priors, batch_means, batch_covariances, batch_x, random_state):
    """
    Batch Predict function for BayesianGMM

    Along the batch dimension, you have different means, covariances, and priors and input.
    The function outputs the predicted delta x for each batch.
    """
    batch_condition = BatchGMM(
        n_components=n_components,
        priors=batch_priors,
        means=batch_means,
        covariances=batch_covariances,
        random_state=random_state,
    ).condition([0, 1, 2], batch_x)
    return batch_condition.one_sample_confidence_region(alpha=0.7)
