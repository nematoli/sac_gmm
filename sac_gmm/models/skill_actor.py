import os
import copy
import logging
import numpy as np
import hydra
import torch


class SkillActor:
    """
    This class handles all functions related to CALVIN skills
    """

    def __init__(self, cfg):
        self.skill_names = None
        self.skills_dir = cfg.skills_dir
        self.skills = None
        self.logger = logging.getLogger("TaskActorWithNSACGMM")
        self.name = None

    def make_skills(self, gmm_cfg):
        self.skills = []
        for skill in self.skill_names:
            gmm_cfg.skill.skill = skill
            self.skills.append(hydra.utils.instantiate(gmm_cfg))
        self.name = self.skills[0].name

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
        dx_pos, dx_ori = self.skills[skill_id].predict(x)
        return np.append(dx_pos, np.append(dx_ori, -1))
