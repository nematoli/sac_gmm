import os
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig
from sac_gmm.envs.calvin.skill_env import SkillSpecificEnv


def make_env(cfg_env: DictConfig):
    if cfg_env.env_name == "calvin":
        env_cfg = {**cfg_env.calvin_env.env}
        env_cfg["use_egl"] = False
        env_cfg["show_gui"] = False
        env_cfg["use_vr"] = False
        env_cfg["use_scene_info"] = True
        env_cfg["tasks"] = cfg_env.calvin_env.tasks
        env_cfg.pop("_target_", None)
        env_cfg.pop("_recursive_", None)

        env = SkillSpecificEnv(**env_cfg)
        # env.set_skill(cfg.skill)
        # env.set_state_type(cfg.state_type)
        # # TODO: find the transforms for this
        # env.set_obs_transforms(cfg.datamodule.transforms)

        # video_dir = os.path.join(cfg.exp_dir, "videos")
        # os.makedirs(video_dir, exist_ok=True)
        # env.set_outdir(video_dir)

        # env.set_obs_allowed(cfg.env_obs_allowed)
        # env.observation_space = env.get_obs_space()
    else:
        raise NotImplementedError

    return env


if __name__ == "__main__":
    make_env()
