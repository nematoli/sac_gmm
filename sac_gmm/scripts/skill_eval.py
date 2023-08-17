import logging
import hydra
import os
import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.utilities import rank_zero_only
from sac_gmm.utils.utils import print_system_env_info, setup_logger, setup_callbacks, get_last_checkpoint

import sac_gmm.models.sac_gmm_model as models_m

cwd_path = Path(__file__).absolute().parents[0]  # scripts
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

sys.path.insert(0, sac_gmm_path.as_posix())
sys.path.insert(0, os.path.join(root, "calvin_env"))  # root/calvin_env
sys.path.insert(0, root.as_posix())

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


def eval_skill_model(skill, env, target, num_test_episodes, device="cuda"):
    succesful_rollouts, rollout_returns, rollout_lengths = 0, [], []
    for idx in range(num_test_episodes):
        if (idx % 5 == 0) or (idx == num_test_episodes):
            log_rank_0(f"Test Trajectory {idx+1}/{num_test_episodes}")

    episode_steps = 0

    observation = env.reset()
    while episode_steps < skill.max_steps:
        # optional adaptation of dynamical system

        skill.agent.gmm.copy_model(skill.agent.initial_gmm)
        gmm_change = skill.agent.get_action(skill.actor, observation, "deterministic", device)
        skill.agent.update_gaussians(gmm_change)

        d_x = skill.agent.gmm.predict(observation["position"] - target)
        observation, reward, done, info = env.step(d_x)
        episode_steps += 1
        if done:
            break

    if info["success"]:
        succesful_rollouts += 1
        status = "Success"
    else:
        status = "Fail"
    log_rank_0(f"{idx+1}: {status}!")

    accuracy = succesful_rollouts / num_test_episodes

    return accuracy


# @hydra.main(version_base="1.1", config_path="../../config", config_name="skill_eval")
# def eval(cfg: DictConfig) -> None:


#     # Load Model

#     # pass to evaluate function

# if __name__ == "__main__":
#     eval()
