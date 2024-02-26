import logging
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.rl.task_rl import TaskRL
import wandb
import os
from collections import Counter

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


OBS_KEY = "rgb_gripper"
# OBS_KEY = "robot_obs"


class SACNGMM_FT(TaskRL):
    """SAC-N-GMM Finetuning implementation using PyTorch Lightning"""

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
        model_ckpt: str,
        rb_dir: str,
    ):
        super().__init__(
            discount=discount,
            batch_size=batch_size,
            replay_buffer=replay_buffer,
            encoder=encoder,
            agent=agent,
            actor=actor,
            critic=critic,
            model=model,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            critic_tau=critic_tau,
            alpha_lr=alpha_lr,
            init_alpha=init_alpha,
            fixed_alpha=fixed_alpha,
            model_lr=model_lr,
            model_tau=model_tau,
            eval_frequency=eval_frequency,
        )
        self.episode_done = False
        self.save_hyperparameters()
        self.max_env_steps = None

        # Populate the replay buffer with random actions
        self.agent.populate_replay_buffer(self.actor, self.model, self.replay_buffer)

        self.load_checkpoint(model_ckpt, agent.root_dir)
        if rb_dir is not None:
            self.load_replay_buffer(rb_dir, agent.root_dir)

    def training_step(self, batch, batch_idx):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        """
        reward, self.episode_done = self.agent.play_step(
            self.actor, self.model, "stochastic", self.replay_buffer, self.device2
        )
        self.episode_return += reward
        self.episode_play_steps += 1

        losses = self.loss(self.check_batch(batch))
        self.log_loss(losses)
        self.soft_update(self.critic_target, self.critic, self.critic_tau)

    def evaluation_step(self):
        metrics = {}
        eval_accuracy, eval_return, eval_length, eval_skill_ids, eval_video_path = self.agent.evaluate(
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
        return metrics, eval_video_path

    def on_train_epoch_end(self):
        if self.episode_done:
            video_path = None
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
                eval_metrics, video_path = self.evaluation_step()
                metrics.update(eval_metrics)

            self.episode_return, self.episode_play_steps = 0, 0
            self.episode_idx += 1
            self.replay_buffer.save()

            # Programs exits when maximum env steps is reached
            # Before exiting, logs the evaluation metrics and videos
            if self.agent.total_env_steps > self.max_env_steps:
                eval_metrics, video_path = self.evaluation_step()
                metrics.update(eval_metrics)
                self.log_metrics(metrics, on_step=False, on_epoch=True)
                if video_path is not None and isinstance(video_path, str):
                    self.log_video(video_path, "eval/video")
                log_rank_0("Maximum env steps reached. Exiting...")
                wandb.finish()
                raise KeyboardInterrupt
            self.log_metrics(metrics, on_step=False, on_epoch=True)
            if video_path is not None and isinstance(video_path, str):
                self.log_video(video_path, "eval/video")

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
            "losses/reconstruction": model_loss["recon_loss"],
        }
        return losses

    def compute_critic_loss(self, batch, critic_optimizer):
        (
            batch_obs,
            batch_skill_ids,
            batch_actions,
            batch_rewards,
            batch_next_obs,
            batch_next_skill_ids,
            batch_dones,
        ) = batch

        with torch.no_grad():
            skill_vector = self.agent.get_skill_vector(batch_next_skill_ids)
            enc_ob = self.model.encoder({"obs": batch_next_obs["rgb_gripper"].float()}).squeeze(0)
            actor_input = torch.cat((enc_ob, skill_vector), dim=-1).cuda().float()

            policy_actions, log_pi = self.actor.get_actions(actor_input, deterministic=False, reparameterize=False)
            q1_next_target, q2_next_target = self.critic_target(actor_input, policy_actions)

            q_next_target = torch.min(q1_next_target, q2_next_target)
            q_target = batch_rewards + (1 - batch_dones) * self.discount * (q_next_target - self.alpha * log_pi)

        # Bellman loss
        with torch.no_grad():
            skill_vector = self.agent.get_skill_vector(batch_skill_ids)
            enc_ob = self.model.encoder({"obs": batch_obs["rgb_gripper"].float()}).squeeze(0)
            actor_input = torch.cat((enc_ob, skill_vector), dim=-1).cuda().float()
        q1_pred, q2_pred = self.critic(actor_input, batch_actions.float())
        bellman_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        critic_optimizer.zero_grad()
        self.manual_backward(bellman_loss)
        critic_optimizer.step()

        return bellman_loss

    def compute_actor_and_alpha_loss(self, batch, actor_optimizer, alpha_optimizer):
        batch_obs = batch[0]
        batch_skill_ids = batch[1]
        with torch.no_grad():
            skill_vector = self.agent.get_skill_vector(batch_skill_ids)
            enc_ob = self.model.encoder({"obs": batch_obs["rgb_gripper"].float()}).squeeze(0)
            actor_input = torch.cat((enc_ob, skill_vector), dim=-1).cuda().float()
        policy_actions, log_pi = self.actor.get_actions(actor_input, deterministic=False, reparameterize=True)
        q1, q2 = self.critic(actor_input, policy_actions)
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
        batch_obs = batch[0]

        # Reconstruction Loss
        enc_state = self.model.encoder({"obs": batch_obs["rgb_gripper"].float()})
        recon_obs = self.model.decoder(enc_state)
        recon_loss = -recon_obs["obs"].log_prob(batch_obs["rgb_gripper"].float()).mean()

        model_optimizer.zero_grad()
        self.manual_backward(recon_loss)
        model_optimizer.step()

        model_loss_dict = {}
        model_loss_dict["recon_loss"] = recon_loss

        # Visualize Decoded Images
        if OBS_KEY == "rgb_gripper" and self.episode_done and (self.episode_idx % self.eval_frequency == 0):
            # Log image and decoded image
            rand_idx = torch.randint(0, batch_obs["rgb_gripper"].shape[0], (1,)).item()
            image = batch_obs["rgb_gripper"][rand_idx].detach() * 255.0
            decoded_image = recon_obs["obs"].mean[rand_idx].detach() * 255.0
            self.log_image(image, "eval/gripper")
            self.log_image(decoded_image, "eval/decoded_gripper")
        return model_loss_dict

    def load_checkpoint(self, model_ckpt, root_dir):
        """Load pretrained weights of actor and critic"""

        ckpt = torch.load(os.path.join(root_dir, model_ckpt))
        log_rank_0(f"Loading actor, critic and model from {model_ckpt}")

        # Get only actor related state_dict
        actor_state_dict = {
            k.replace("actor.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.replace("actor.", "") in self.actor.state_dict()
        }
        self.actor.load_state_dict(actor_state_dict)

        # Get only critic related state_dict
        # critic_state_dict = {
        #     k.replace("critic.", ""): v
        #     for k, v in ckpt["state_dict"].items()
        #     if k.replace("critic.", "") in self.critic.state_dict()
        # }
        # self.critic.load_state_dict(critic_state_dict)

        # # Get only critic_target related state_dict
        # critic_state_dict = {
        #     k.replace("critic_target.", ""): v
        #     for k, v in ckpt["state_dict"].items()
        #     if k.replace("critic_target.", "") in self.critic_target.state_dict()
        # }
        # self.critic_target.load_state_dict(critic_state_dict)

        # Get only model related state_dict
        model_state_dict = {
            k.replace("model.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.replace("model.", "") in self.model.state_dict()
        }
        self.model.load_state_dict(model_state_dict)

    def load_replay_buffer(self, replay_buffer_dir, root_dir):
        # Load replay buffer
        log_rank_0(f"Loading replay buffer from {replay_buffer_dir}")
        temp = self.replay_buffer.save_dir
        self.replay_buffer.save_dir = os.path.join(root_dir, replay_buffer_dir)
        self.replay_buffer.load()
        self.replay_buffer.save_dir = temp

    def evaluate(self):
        eval_accuracy, eval_return, eval_length, eval_skill_ids, eval_video_path = self.agent.evaluate(
            self.actor.cuda(), self.model.cuda()
        )
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

        log_rank_0(
            f"Accuracy: {eval_accuracy}, Average Return: {eval_return}, Average Trajectory Length: {eval_length}"
        )
        log_rank_0(f"Skill Distribution: {skill_ids}")
        log_rank_0(f"Saved video at {eval_video_path}")
