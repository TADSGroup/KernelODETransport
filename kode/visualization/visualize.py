import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_distribution(ax, reference, target, prediction):
    '''
    Plot the 1d measure distributions
    '''
    ax.hist(np.array(reference.flatten()), bins=100, density=True,
                label='reference')
    ax.hist(np.array(target.flatten()), bins=100, density=True, label='target')
    ax.hist(np.array(prediction.flatten()), bins=100, density=True,
            label='predictions', alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlim(-10, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('P(x)')
    plt.legend()
    return ax


def plot_loss(ax, train_epochs, train_mmd_loss, h1_norm, rkhs_norm,
              test_epochs, test_mmd_loss):
    ax.semilogy(train_epochs, train_mmd_loss, label='Train MMD loss')
    ax.semilogy(train_epochs, rkhs_norm, label='RKHS norm')
    ax.semilogy(train_epochs, h1_norm, label='H1 norm')
    ax.semilogy(test_epochs, test_mmd_loss, label='Test MMD loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.legend()
    return ax


def plot_trajectory(ax, trajectory, ts):
    for traj in trajectory:
        ax.plot(traj, ts, lw=0.5, c='b')
    ax.set_xlabel('input / output')
    ax.set_ylabel('time / depth')
    return ax


def plot_2d_histogram(ax, samples, xedges, yedges, cmap='magma', vmin=0,
                      vmax=0.15, axis=False):

    xmin, xmax = xedges[0], xedges[-1]
    ymin, ymax = yedges[0], yedges[-1]

    ax.hist2d(samples[:, 0], samples[:, 1], bins=[xedges, yedges],
               density=True, cmap=cmap, vmin=vmin, vmax=vmax,
              range=[[xmin, xmax],[ymin, ymax]])
    if axis is False:
        ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    return ax


def plot_2d_trajectories(ax, trajectory, num_points, seed=20):
    np.random.seed(seed)
    idx = np.random.randint(0, trajectory.shape[1], num_points)
    for i in idx:
        t = trajectory[:, i, :]
        plt.plot(t[:, 0], t[:, 1], '-', c='r', alpha=1)
    plt.scatter(trajectory[-1, idx, 0], trajectory[-1, idx, 1],
                s=20, facecolors='k', edgecolors='k', label='target')
    plt.scatter(trajectory[0, idx, 0], trajectory[0, idx, 1],
                s=20, facecolors='none', edgecolors='k', label='reference')
    return ax


def plot_hist(x, y, xedges=None, yedges=None, cmap='magma', vmin=0,
              vmax=0.15, **kwargs):
    if xedges is None:
        xedges = np.linspace(-4, 4, 50)
    if yedges is None:
        yedges = np.linspace(-4, 4, 50)
    plt.hist2d(x, y, bins=[xedges, yedges],
               density=True, cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.grid('False')
    pass


def plot_2d_distributions(fig, X1, X2, X3, xedges=None, yedges=None):
    if xedges is None:
        xedges = np.linspace(-4, 4, 50)
    if yedges is None:
        yedges = np.linspace(-4, 4, 50)

    ax1 = plot_2d_histogram(fig.add_subplot(131), X1, xedges, yedges)
    ax2 = plot_2d_histogram(fig.add_subplot(132), X2, xedges, yedges)
    ax3 = plot_2d_histogram(fig.add_subplot(133), X3, xedges, yedges)
    return ax1, ax2, ax3

def plot_multidim_marginals(X, bins=None,  vmin=0, vmax=0.15, title=None,
                            cmap='magma'):
    if bins is None:
        bins = np.linspace(-4, 4, 50)

    X_df = pd.DataFrame(np.array(X))
    g = sns.PairGrid(X_df, corner=True)
    g.map_diag(sns.histplot, stat='density', bins=bins)
    g.map_lower(plot_hist, xedges=bins, yedges=bins, vmin=vmin,
                vmax=vmax, cmap=cmap)
    g.figure.suptitle(title)
    plt.show()
    return g


def plot_conditional_density(ax, Y_true, Y_predicted, labels=['True',
                                                          'Predicted']):
    ax = sns.kdeplot(np.array(Y_true), color='r', linestyle='-',
                     bw_method=0.1, label=labels[0])
    ax = sns.kdeplot(np.array(Y_predicted), color='r', lw=4, linestyle='--',
                     bw_method=0.1, label=labels[1])
    plt.xlabel('u')
    plt.ylabel('P(u)')
    plt.legend()
    return ax