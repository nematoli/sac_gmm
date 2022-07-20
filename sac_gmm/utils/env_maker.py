import hydra
from hydra import initialize, compose
from omegaconf import DictConfig


def make_env():
    """
    """
    with initialize(config_path="./config/calvin_env/conf/"):
        cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
        cfg.env["use_egl"] = False
        cfg.env["show_gui"] = False
        cfg.env["use_vr"] = False
        cfg.env["use_scene_info"] = True
    return cfg.env

if __name__ == "__main__":
    make_env()