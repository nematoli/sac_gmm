from omegaconf import DictConfig
from sac_n_gmm.envs.calvin.skill_env import CalvinSkillEnv
from sac_n_gmm.envs.calvin.task_env import CalvinTaskEnv
from sac_n_gmm.envs.calvin.rand_skill_env import CalvinRandSkillEnv
from sac_n_gmm.envs.calvin.incr_skill_env import CalvinIncrSkillEnv


def make_env(cfg_env: DictConfig):
    if cfg_env.env_name == "calvin":
        env = CalvinSkillEnv(cfg_env)
    elif cfg_env.env_name == "calvin_task":
        env = CalvinTaskEnv(cfg_env)
    elif cfg_env.env_name == "calvin_rand_skill":
        env = CalvinRandSkillEnv(cfg_env)
    elif cfg_env.env_name == "calvin_incr_skill":
        env = CalvinIncrSkillEnv(cfg_env)
    else:
        raise NotImplementedError

    return env


if __name__ == "__main__":
    make_env()
