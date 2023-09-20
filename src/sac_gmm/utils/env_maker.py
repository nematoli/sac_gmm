from omegaconf import DictConfig
from sac_gmm.envs.calvin.skill_env import CalvinSkillEnv

try:
    from rl_tasks import ENV_TYPES
except ModuleNotFoundError:
    ENV_TYPES = None


def make_env(cfg_env: DictConfig, skill=None, start_point=None):
    if cfg_env.env_name == "calvin":
        env = CalvinSkillEnv(cfg_env, skill, start_point)
    elif cfg_env.env_name == "bullet":
        env = ENV_TYPES[cfg_env.bullet.type](cfg_env.bullet, show_gui=cfg_env.show_gui)
        env.skill = skill

    else:
        raise NotImplementedError

    return env


if __name__ == "__main__":
    make_env()
