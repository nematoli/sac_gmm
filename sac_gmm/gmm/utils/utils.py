import numpy as np
import numbers
import matplotlib.pyplot as plt


# Taken from scikit learn to get rid of the dependency


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState" " instance" % seed)


def plot_3d_trajectories(demos, repro=None):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")
    if repro is None:
        for i in range(demos.shape[0]):
            x_val = demos[i, :, 0]
            y_val = demos[i, :, 1]
            z_val = demos[i, :, 2]
            ax.scatter(x_val, y_val, z_val, s=10)
    else:
        ax.scatter(demos[:, 0], demos[:, 1], demos[:, 2], alpha=0.5, s=1, label="Demonstration")
        ax.scatter(repro[:, 0], repro[:, 1], repro[:, 2], s=5, label="Reproduction")
    plt.legend()
    plt.tight_layout()
    plt.show()
