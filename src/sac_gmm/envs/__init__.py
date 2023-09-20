from gym.envs.registration import register

register(id="skill-env", entry_point="sac_gmm.envs.calvin:CalvinSkillEnv", max_episode_steps=64)
