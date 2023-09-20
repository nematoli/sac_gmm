import numpy as np


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
    return np.dot(R2, np.linalg.inv(R1))


def rotation_matrix_to_euler(R):
    """
    Extracts Euler angles from a rotation matrix.
    """
    theta = np.arctan2(R[2, 1], R[2, 2])
    phi = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    psi = np.arctan2(R[1, 0], R[0, 0])
    return np.array((theta, phi, psi))
