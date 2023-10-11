import torch


def xyz_to_XYZ(z, cv, p, hw, intrinsics):
    T_world_cam = torch.linalg.inv(torch.tensor(cv).reshape((4, 4)).T)
    h, w = hw
    u, v = p
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]

    world_pos = T_world_cam.float() @ torch.tensor([x, y, z, 1])
    world_pos = world_pos[:3]

    return world_pos


def cam_view_point_to_pix(projection_matrix, img_res, point):
    h, w = img_res
    pnt = point
    if len(point.shape) == 1:
        projection_matrix = projection_matrix.unsqueeze(0)
        pnt = point.unsqueeze(0)
    persp_m = torch.tensor(projection_matrix).reshape((-1, 4, 4)).transpose(1, 2)
    world_pix_tran = torch.bmm(persp_m, pnt.unsqueeze(2)).squeeze(2)
    world_pix_tran = world_pix_tran / world_pix_tran[:, [-1]]  # divide by w
    world_pix_tran[:, :3] = (world_pix_tran[:, :3] + 1) / 2
    x, y = world_pix_tran[:, 0] * w, (1 - world_pix_tran[:, 1]) * h
    x, y = torch.floor(x).int(), torch.floor(y).int()
    re = torch.stack([x, y], dim=0).T
    if len(point.shape) == 1:
        re = re.squeeze()
    return re


def cam_to_world(points_cam, T_cam_world):
    if points_cam.dim() == 2:
        points_cam = points_cam.unsqueeze(1)
    batch_size, N, _ = points_cam.shape
    T_cam_world = T_cam_world.reshape(batch_size, 4, 4).transpose(1, 2)

    points_cam_hom = torch.cat([points_cam, torch.ones(batch_size, N, 1, device=points_cam.device)], dim=-1)

    points_world_hom = torch.bmm(points_cam_hom, T_cam_world.inverse().transpose(1, 2))

    points_world = points_world_hom[:, :, :3]
    return points_world.squeeze()


def world_to_cam(points_world, T_cam_world):
    if points_world.dim() == 2:
        points_world = points_world.unsqueeze(1)
    batch_size, N, _ = points_world.shape
    T_cam_world = T_cam_world.reshape(batch_size, 4, 4).transpose(1, 2)

    points_world_hom = torch.cat([points_world, torch.ones(batch_size, N, 1, device=points_world.device)], dim=-1)

    points_cam_homogeneous = torch.bmm(points_world_hom, T_cam_world.transpose(1, 2))

    points_cam = points_cam_homogeneous[:, :, :3]
    return points_cam.squeeze()


def project_to_image(points_cam, intrinsics):
    if points_cam.dim() == 2:
        points_cam = points_cam.unsqueeze(1)
    batch_size, N, _ = points_cam.shape

    # Transform points to homogeneous coordinates
    points_cam_homogeneous = points_cam / points_cam[:, :, 2:]

    # Multiply with the intrinsic matrix
    points_img_homogeneous = torch.bmm(points_cam_homogeneous, intrinsics.transpose(1, 2))

    pixels_cam = points_img_homogeneous[:, :, :2].squeeze()

    return torch.floor(pixels_cam).int()
