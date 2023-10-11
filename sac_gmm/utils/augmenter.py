import torch
from collections.abc import Iterable
from sac_gmm.utils.projections import world_to_cam, project_to_image


class Augmenter:
    def __init__(self, augment, cover_dist, margin, min_cover_dist):
        super(Augmenter, self).__init__()

        self.augment = augment
        self.cover_dist = cover_dist  # in pixel
        self.margin = margin
        self.min_cover_dist = min_cover_dist

    def labels_far_enough(self, positions, labels):
        labels = labels.to(positions.device)
        takes = (labels - positions).pow(2).sum(dim=1).sqrt() > self.min_cover_dist
        return takes

    def cover_side(self, indexes, axis_coords, length):
        dax = (torch.rand(axis_coords.shape[0]) - 0.5).sign() * self.cover_dist
        dax = dax.to(indexes.device)
        border_plus = dax + axis_coords
        border_minus = dax - axis_coords
        border_in_frame = torch.logical_and(border_plus >= 0, border_plus < length)
        border = torch.where(border_in_frame, border_plus, border_minus)
        lr = (border - axis_coords) > 0
        border = border.reshape([-1, 1, 1, 1])
        lr = lr.reshape([-1, 1, 1, 1])
        masks = torch.where(lr, indexes < border, indexes > border)
        return masks

    def random_cover(self, images, points, cover_probability=1.0):
        hw = torch.tensor(images.shape[-2:]).to(images.device).flip(0)
        o = torch.zeros_like(images).type_as(images)
        points = points.to(o.device)

        x = torch.arange((o.shape[2])).to(o.device)
        y = torch.arange((o.shape[3])).to(o.device)
        x_o = (o.permute((0, 1, 3, 2)) + x).permute((0, 1, 3, 2))
        y_o = o + y

        if isinstance(self.cover_dist, Iterable):
            mn, mx = min(self.cover_dist[0], self.cover_dist[1]), max(self.cover_dist[0], self.cover_dist[1])
            self.cover_dist = torch.rand(images.shape[0])
            self.cover_dist = self.cover_dist * (mx - mn) + mn

        x_r = self.cover_side(x_o, points[:, 1], hw[1])
        y_r = self.cover_side(y_o, points[:, 0], hw[0])

        over_x = (torch.rand(images.shape[0]) > 0.5).reshape([-1, 1, 1, 1])
        over_x = over_x.to(x_r.device)
        r_r = torch.where(over_x, x_r, y_r)

        if cover_probability < 1.0:
            cover = (torch.rand(images.shape[0]) < cover_probability).reshape([-1, 1, 1, 1])
            cover = cover.to(r_r.device)
            mask = torch.where(cover, r_r, torch.ones_like(r_r).to(r_r.device))
        else:
            mask = r_r

        takes = torch.logical_and(points >= self.margin, points < (hw - self.margin)).all(dim=1).reshape([-1, 1, 1, 1])
        mask = torch.where(takes, mask, torch.ones_like(mask).to(mask.device))
        random_noise = (torch.rand(images.size()).to(images.device) * 255).byte()
        re = torch.where(mask, images, random_noise)
        return re

    def cover_batch_obs(self, batch_obs, gt_keypoint):
        hw = (84, 84)
        h, w = hw
        view = batch_obs["view_mtx"]
        intrinsics = batch_obs["intrinsics"]
        if len(intrinsics.shape) == 1:
            intrinsics = intrinsics.unsqueeze(0)
            view = view.unsqueeze(0)

        pos = batch_obs["position"]
        labels_pos_world = torch.tensor(gt_keypoint).repeat(pos.shape[0], 1)
        takes = self.labels_far_enough(pos, labels_pos_world)
        takes = takes.reshape((-1, 1, 1, 1))

        labels_pos_world = torch.tensor(gt_keypoint).repeat(intrinsics.shape[0], 1)
        intrinsics = intrinsics.type_as(labels_pos_world)
        view = view.type_as(labels_pos_world)

        label_in_cam = world_to_cam(labels_pos_world, view)
        ps = project_to_image(label_in_cam, intrinsics)

        covers = self.random_cover(batch_obs["rgb_gripper"], ps)

        batch_obs["rgb_gripper"] = torch.where(takes, covers, batch_obs["rgb_gripper"])

    def augment_rgb(self, batch, gt_keypoint):
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch
        batch = (batch_obs.copy(), batch_actions, batch_rewards, batch_next_obs, batch_dones)
        self.cover_batch_obs(batch[0], gt_keypoint)
        return batch
