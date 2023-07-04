from gym.envs.registration import register

register(id="calvin-env-v0", entry_point="sac_gmm.envs.calvin:CalvinEnvV0", max_episode_steps=64)
