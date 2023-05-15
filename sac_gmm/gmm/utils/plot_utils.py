import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.animation as animation

# Credits: Yizhak Ben-Shabat
# Source: https://github.com/sitzikbs/gmm_tutorial/
# Notes: Slightly modified
def visualize_3d_gmm(points, w, mu, stdev, skill, export_dir, export_type='gif'):
    '''
    plots points and their corresponding gmm model in 3D
    Input: 
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: 3 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''
    n_gaussians = mu.shape[1]
    N = int(np.floor(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection='3d')
    plt.set_cmap('Set1')
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        axes.scatter(points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=[[0,0,0]])
        plot_sphere3d(w=w[i], c=mu[:, i], r=stdev[:, i], ax=axes)

    plt.title(f'3D GMM: {skill}')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')

    outfile = None
    if export_type == 'gif':
        def init():
            axes.view_init(elev=28, azim=0)
            return fig,

        def animate(i):
            axes.view_init(elev=28, azim=i)
            return fig,

        # Animate
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                        frames=180, interval=30, blit=True)
        # Save
        anim.save(os.path.join(export_dir, 'gmm.gif'), fps=20)
        outfile = os.path.join(export_dir, 'gmm.gif')
    else:
        axes.view_init(elev=35.246, azim=45)
        plt.savefig(os.path.join(export_dir, 'gmm.png'), dpi=100, format='png')
        outfile = os.path.join(export_dir, 'gmm.png')
    return outfile

# Credits: Yizhak Ben-Shabat
# Source: https://github.com/sitzikbs/gmm_tutorial/
def plot_sphere3d(w=0, c=[0,0,0], r=[1, 1, 1], subdev=10, ax=None, sigma_multiplier=3):
    '''
        plot a sphere surface
        Input: 
            c: 3 elements list, sphere center
            r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
            subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
            ax: optional pyplot axis object to plot the sphere in.
            sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
        Output:
            ax: pyplot axis object
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:complex(0,subdev), 0.0:2.0 * pi:complex(0,subdev)]
    x = sigma_multiplier*r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier*r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier*r[2] * cos(phi) + c[2]
    cmap = cmx.ScalarMappable()
    cmap.set_cmap('prism')
    c = cmap.to_rgba(w)

    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)

    return ax