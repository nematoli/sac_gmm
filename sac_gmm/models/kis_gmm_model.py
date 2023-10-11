import logging
import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.models.skill_model import SkillModel
from sac_gmm.utils.projections import world_to_cam, project_to_image

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class KISGMM(SkillModel):
    """Basic KIS-GMM implementation using PyTorch Lightning"""

    def __init__(
        self,
        sac: DictConfig,
        agent: DictConfig,
        kp_det: DictConfig,
        kp_lr: float,
        augmenter: DictConfig,
    ):
        super(KISGMM, self).__init__(
            discount=sac.discount,
            batch_size=sac.batch_size,
            replay_buffer=sac.replay_buffer,
            actor=sac.actor,
            critic=sac.critic,
            actor_lr=sac.actor_lr,
            critic_lr=sac.critic_lr,
            critic_tau=sac.critic_tau,
            optimize_alpha=sac.optimize_alpha,
            alpha_lr=sac.alpha_lr,
            init_alpha=sac.init_alpha,
            eval_frequency=sac.eval_frequency,
            agent=agent,
        )
        self.episode_done = False

        # # Keypoint Detector
        self.kp_det = hydra.utils.instantiate(kp_det)
        self.kp_lr = kp_lr
        self.agent.set_keypoint_detector(self.kp_det)

        self.augmenter = hydra.utils.instantiate(augmenter)

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizers = super(KISGMM, self).configure_optimizers()
        self.target_det_opt_id = len(optimizers)
        keypoint_optimizer = torch.optim.Adam(self.kp_det.parameters(), lr=self.kp_lr)
        optimizers.append(keypoint_optimizer)

        return optimizers

    def training_step(self, batch, batch_idx):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        """
        reward, self.episode_done = self.agent.play_step(self.actor, "stochastic", self.replay_buffer, self.device)
        self.episode_return += reward
        self.episode_length += 1

        batch = self.check_batch(batch)
        if self.augmenter.augment:
            batch = self.augmenter.augment_rgb(batch, self.agent.get_gt_keypoint())

        losses = self.loss(self.check_batch(batch))
        self.log_loss(losses)
        self.soft_update(self.critic_target, self.critic, self.critic_tau)

    def on_train_epoch_end(self):
        metrics = {"eval_episode-avg-return": float("-inf")}
        if self.episode_done:
            log_rank_0(f"episode {self.episode_idx} done")
            train_metrics = {
                "train_episode-return": self.episode_return,
                "train_episode-length": self.episode_length,
                "train_episode-number": self.episode_idx,
            }
            metrics.update(train_metrics)
            eval_return = float("-inf")
            eval_accuracy = float("-inf")
            if self.episode_idx % self.eval_frequency == 0:
                eval_accuracy, eval_return, eval_length = self.agent.evaluate(self.actor)
                eval_metrics = {
                    "eval_accuracy": eval_accuracy,
                    "eval_episode-avg-return": eval_return,
                    "eval_episode-avg-length": eval_length,
                    "eval_total-env-steps": self.agent.total_env_steps,
                }
                metrics.update(eval_metrics)

            self.episode_return, self.episode_length = 0, 0
            self.episode_idx += 1

            self.replay_buffer.save()
        self.log_metrics(metrics, on_step=False, on_epoch=True)

    def loss(self, batch):
        if self.optimize_alpha:
            critic_optimizer, actor_optimizer, alpha_optimizer, kp_optimizer = self.optimizers()
        else:
            critic_optimizer, actor_optimizer, kp_optimizer = self.optimizers()
            alpha_optimizer = None

        features = self.agent.get_features_from_observation(batch[0])
        critic_loss = self.compute_critic_loss(batch, features, critic_optimizer)
        actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(batch, features, actor_optimizer, alpha_optimizer)
        kp_ce_loss, kp_mse_loss = self.compute_kp_loss(batch[0], features, kp_optimizer)

        losses = {
            "loss_critic": critic_loss,
            "loss_actor": actor_loss,
            "loss_alpha": alpha_loss,
            "loss_ce": kp_ce_loss,
            "loss_mse": kp_mse_loss,
            "alpha_value": self.alpha,
        }
        return losses

    def compute_critic_loss(self, batch, features, critic_optimizer):
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        with torch.no_grad():
            next_features = self.agent.get_features_from_observation(batch_next_obs)
            next_state = self.agent.get_state_from_observation(next_features, batch_next_obs)
            policy_actions, log_pi = self.actor.get_actions(next_state, deterministic=False, reparameterize=False)
            q1_next_target, q2_next_target = self.critic_target(next_state, policy_actions)

            q_next_target = torch.min(q1_next_target, q2_next_target)
            q_target = batch_rewards + (1 - batch_dones) * self.discount * (q_next_target - self.alpha * log_pi)

        # Bellman loss
        state = self.agent.get_state_from_observation(features, batch_obs)
        q1_pred, q2_pred = self.critic(state, batch_actions.float())
        bellman_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        critic_optimizer.zero_grad()
        self.manual_backward(bellman_loss)
        critic_optimizer.step()

        return bellman_loss

    def compute_actor_and_alpha_loss(self, batch, features, actor_optimizer, alpha_optimizer):
        state = self.agent.get_state_from_observation(features, batch[0])
        policy_actions, log_pi = self.actor.get_actions(state, deterministic=False, reparameterize=True)
        q1, q2 = self.critic(state, policy_actions)
        Q_value = torch.min(q1, q2)
        actor_loss = (self.alpha * log_pi - Q_value).mean()

        actor_optimizer.zero_grad()
        self.manual_backward(actor_loss)
        actor_optimizer.step()

        if self.optimize_alpha:
            self.log_alpha = self.log_alpha.to(log_pi.device)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            self.manual_backward(alpha_loss)
            alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0

        return actor_loss, alpha_loss

    def compute_kp_loss(self, batch_obs, features, kp_optimizer):
        xyzo = self.agent.kp_det.keypoint(features)

        view = torch.tensor(batch_obs["view_mtx"]).float().type_as(xyzo)
        intrinsics = torch.tensor(batch_obs["intrinsics"]).float().type_as(xyzo)

        labels_in_world = torch.tensor(self.agent.get_gt_keypoint()).repeat(xyzo.shape[0], 1).type_as(xyzo)
        h, w = 84, 84

        label_in_cam = world_to_cam(labels_in_world, view)
        zs = label_in_cam[:, 2]
        ps = project_to_image(label_in_cam, intrinsics)

        visibility_border_gap_pixels = self.augmenter.margin
        cmp = torch.tensor([h, w]).type_as(xyzo) - visibility_border_gap_pixels
        # checks is the learned target is in the field of view of the camera
        ce_labels = torch.all((cmp > ps), dim=1) & torch.all((ps >= visibility_border_gap_pixels), dim=1)

        ce_outs = xyzo[:, 3]
        kp_ce_loss = F.binary_cross_entropy(ce_outs.unsqueeze(1), ce_labels.float().unsqueeze(1))

        xy_mse_l = F.mse_loss(xyzo[ce_labels][:, :2], ps[ce_labels].float())
        z_mse_l = F.mse_loss(xyzo[ce_labels][:, 2], zs[ce_labels].float())
        kp_mse_loss = xy_mse_l + 10 * z_mse_l

        kp_loss = 50 * kp_ce_loss + kp_mse_loss

        kp_optimizer.zero_grad()
        self.manual_backward(kp_loss)
        kp_optimizer.step()

        if self.episode_idx % self.eval_frequency == 0:
            self.agent.kp_det.eval()
            heatmap, _, _ = self.agent.kp_det.forward(features[0].unsqueeze(0))
            self.agent.kp_det.train()
            self.log_keypoint(batch_obs["rgb_gripper"][0], heatmap, zs[0], ps[0], xyzo[0])

        return kp_ce_loss, kp_mse_loss
