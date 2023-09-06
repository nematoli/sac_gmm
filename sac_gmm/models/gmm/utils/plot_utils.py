import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def visualize_3d_gmm(points, priors, means, covariances, save_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colors = ["r", "g", "b", "c", "m", "y"]  # Color for each component

    def update_frame(i):
        ax.cla()  # Clear the previous frame

        for j in range(len(priors)):
            eigenvalues, eigenvectors = np.linalg.eigh(covariances[j])
            scaling_factors = np.sqrt(eigenvalues)
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = scaling_factors[0] * np.outer(np.cos(u), np.sin(v))
            y = scaling_factors[1] * np.outer(np.sin(u), np.sin(v))
            z = scaling_factors[2] * np.outer(np.ones_like(u), np.cos(v))

            for k in range(len(x)):
                for l in range(len(x[k])):
                    [x[k][l], y[k][l], z[k][l]] = np.dot([x[k][l], y[k][l], z[k][l]], eigenvectors) + means[j]

            ax.plot_surface(x, y, z, color=colors[j], alpha=0.3)

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.3, c=[[0, 0, 0]])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Gaussian Mixture Model")

        ax.view_init(elev=28, azim=i * 4)  # Adjust the view angle

    frames = 90  # Number of frames in the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=frames, interval=50)

    ani.save(os.path.join(save_dir, "gmm.gif"), writer="imagemagick")  # Save the animation as a GIF
    return os.path.join(save_dir, "gmm.gif")
