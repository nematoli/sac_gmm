import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_inverse(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def multiply_quaternions(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    w = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
    x = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
    y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
    z = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2

    return np.array([x, y, z, w])


def get_relative_quaternion(q_current, q_target):
    # Calculate the inverse of the current quaternion
    q_current_inv = quaternion_inverse(q_current)

    # Calculate the relative quaternion
    q_relative = multiply_quaternions(q_target, q_current_inv)

    return q_relative


def compute_euler_difference2(goal_quat, current_euler):
    """
    1. Convert Quaternions to Rotation Matrices
    2. Find the Relative Rotation Matrix
    3. Extract Euler Angles from the Relative Rotation Matrix
    """
    R_goal = Rotation.from_quat(goal_quat)
    R_current = Rotation.from_euler("xyz", current_euler, degrees=False)
    R_rel = R_goal * R_current.inv()
    euler_rel = R_rel.as_euler("xyz", degrees=False)
    return euler_rel


def compute_euler_difference(goal_angles, current_angles):
    """
    1. Convert Euler Angles to Rotation Matrices
    2. Find the Relative Rotation Matrix
    3. Extract Euler Angles from the Relative Rotation Matrix
    """
    R_rel = relative_rotation_matrix(current_angles, goal_angles)
    euler_rel = rotation_matrix_to_euler(R_rel)
    return euler_rel


def euler_to_rotation_matrix(theta, phi, psi):
    """
    Converts Euler angles to a rotation matrix.
    """
    R_x = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    R_y = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])

    return np.dot(R_z, np.dot(R_y, R_x))


def relative_rotation_matrix(angles1, angles2):
    """
    Computes the relative rotation matrix between two sets of Euler angles.
    """
    R1 = euler_to_rotation_matrix(angles1[0], angles1[1], angles1[2])
    R2 = euler_to_rotation_matrix(angles2[0], angles2[1], angles2[2])
    return np.dot(R2, R1.T)


def rotation_matrix_to_euler(R):
    """
    Extracts Euler angles from a rotation matrix.
    """
    theta = np.arctan2(R[2, 1], R[2, 2])
    phi = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    psi = np.arctan2(R[1, 0], R[0, 0])
    return np.array((theta, phi, psi))
