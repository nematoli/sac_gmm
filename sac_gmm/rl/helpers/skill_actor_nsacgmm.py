import os
import copy
import logging
import numpy as np
import hydra
import torch


class SkillActorWithNSACGMM:
    """
    This class handles all functions related to CALVIN skills
    """

    def __init__(self, cfg):
        self.skill_names = None
        self.skills_dir = cfg.skills_dir
        self.skills = None
        self.sacgmms = None
        self.logger = logging.getLogger("TaskActorWithNSACGMM")
        self.name = None

    def make_skills(self, gmm_cfg):
        self.skills = []
        for skill in self.skill_names:
            gmm_cfg.skill.skill = skill
            self.skills.append(hydra.utils.instantiate(gmm_cfg))
        self.name = self.skills[0].name

    def make_sacgmms(self, sacgmm_cfg, model_ckpts, root_dir, device, action_space):
        sacgmms = []
        encoder = hydra.utils.instantiate(sacgmm_cfg.encoder)
        for id, skill in enumerate(self.skill_names):
            ckpt = torch.load(os.path.join(root_dir, model_ckpts[id]))
            sacgmm_cfg.actor.input_dim = ckpt["hyper_parameters"]["actor"]["input_dim"]
            sacgmm_cfg.actor.action_dim = ckpt["hyper_parameters"]["actor"]["action_dim"]
            actor = hydra.utils.instantiate(sacgmm_cfg.actor)
            # Get only actor related state_dict
            actor_state_dict = {
                k.replace("actor.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.replace("actor.", "") in actor.state_dict()
            }
            actor.load_state_dict(actor_state_dict)
            actor.set_action_space(action_space)
            actor.set_encoder(encoder)
            actor.eval()
            sacgmms.append(actor.to(device))
        return sacgmms

    def set_skill_params(self, dataset_cfg):
        for id, skill in enumerate(self.skill_names):
            dataset_cfg.skill.skill = skill
            dataset = hydra.utils.instantiate(dataset_cfg)
            self.skills[id].set_skill_params(dataset)

    def make_manifolds(self):
        for id, skill in enumerate(self.skills):
            skill.manifold = skill.make_manifold()

    def load_models(self):
        for _, ds in enumerate(self.skills):
            ds.load_model()

    def copy_model(self, initial_gmms, skill_id):
        self.skills[skill_id].copy_model(initial_gmms[skill_id])

    def update_model(self, delta, skill_id):
        self.skills[skill_id].update_model(delta)

    def act(self, x, skill_id):
        dx_pos, dx_ori, is_nan = self.skills[skill_id].predict(x)
        return np.append(dx_pos, np.append(dx_ori, -1)), is_nan
