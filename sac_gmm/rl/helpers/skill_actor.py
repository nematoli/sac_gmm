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
        self.logger = logging.getLogger("SkillActor")
        self.name = None
        self.gmm_type = None
        self.priors_size = None
        self.means_size = None
        self.cov_size = None

    def make_skills(self, gmm_cfg):
        self.skills = []
        for skill in self.skill_names:
            gmm_cfg.skill.skill = skill
            self.skills.append(hydra.utils.instantiate(gmm_cfg))
        self.name = self.skills[0].name
        self.gmm_type = self.skills[0].gmm_type

    def set_skill_params(self, dataset_cfg):
        for id, skill in enumerate(self.skill_names):
            dataset_cfg.skill.skill = skill
            dataset = hydra.utils.instantiate(dataset_cfg)
            self.skills[id].set_skill_params(dataset)

    def make_manifolds(self):
        for id, skill in enumerate(self.skills):
            skill.manifold, skill.manifold2 = skill.make_manifold()

    def load_models(self):
        for _, ds in enumerate(self.skills):
            ds.load_model()
        self.priors_size, self.means_size, self.cov_size = self.skills[0].get_params_size()

    def copy_model(self, initial_gmms, skill_id):
        self.skills[skill_id].copy_model(initial_gmms[skill_id])

    def update_model(self, delta, skill_id):
        self.skills[skill_id].update_model(delta)

    def act(self, x, skill_id):
        dx_pos, dx_ori, is_nan = self.skills[skill_id].predict(x)
        return np.append(dx_pos, np.append(dx_ori, -1)), is_nan

    def get_all_skill_params(self, initial_gmms):
        skill_params = []
        for skill in initial_gmms:
            skill_params.append(skill.model_params())
        return np.array(skill_params)
