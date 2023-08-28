from omegaconf import DictConfig
from sac_gmm.envs.calvin.skill_env import CalvinSkillEnv
from sac_gmm.envs.calvin.task_env import CalvinTaskEnv


def make_env(cfg_env: DictConfig):
    if cfg_env.env_name == "calvin":
        env = CalvinSkillEnv(cfg_env)
    elif cfg_env.env_name == "calvin_task":
        env = CalvinTaskEnv(cfg_env)
    else:
        raise NotImplementedError

    return env


if __name__ == "__main__":
    make_env()
