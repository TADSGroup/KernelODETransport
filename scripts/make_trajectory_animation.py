import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import jax.random as jrandom
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['figure.dpi'] = 150
plt.ioff()

from kode.data import utils as data_utils, load_dataset
from kode.visualization import visualize


def parse_args():
    parser = argparse.ArgumentParser(description='Load and evaluate Kernel '
                                                 'ODE model.')
    parser.add_argument("--file-name", type=str, required=True,
                        help="Name of the file name with the trained model. "
                             "It will be loaded from models/ folder.")
    parser.add_argument('--num-steps', type=int, default=10,
                        help='Number of discrete steps for ODE solver.')
    parser.add_argument("--cuda-device", type=str, default='2',
                        help="CUDA device ID.")
    return parser.parse_args()



def main():
    args = parse_args()
    num_steps = args.num_steps
    file_name = args.file_name

    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # load data
    key = jrandom.PRNGKey(22)
    X = jrandom.normal(key, shape=(10000, 2))

    # load model
    model_path = 'models/'
    to_load = data_utils.load_file(model_path + file_name + '.pickle')
    transport_model = to_load['model']
    print(to_load['hyperparameters'])

    # model predictions
    predictions, trajectory = transport_model.transform(X, num_steps=num_steps,
                                                     trajectory=True)

    idx = np.random.randint(0, trajectory.shape[1], 100)
    sel_trajectory = trajectory[:, idx, :]

    # Setup the figure and axis
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    ax.set_title('Trajectory')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])

    # Initialize plot elements for each trajectory
    paths = []
    start_markers = []
    end_markers = []

    for i in range(sel_trajectory.shape[1]):  # Loop over particles
        path, = ax.plot([], [], 'r-', lw=0.5)  # Path for each trajectory
        start_marker, = ax.plot([], [], 'o', mec='black', mfc='None',
                                markersize=3)  # Start marker
        end_marker, = ax.plot([], [], 'o', mec='black', mfc='black',
                              markersize=3)  # End marker
        paths.append(path)
        start_markers.append(start_marker)
        end_markers.append(end_marker)

    # Initialization function
    def init():
        for path, start_marker, end_marker in zip(paths, start_markers,
                                                  end_markers):
            path.set_data([], [])
            start_marker.set_data([], [])
            end_marker.set_data([], [])
        return paths + start_markers + end_markers

    # Update function
    def update(frame):
        for i, (path, start_marker, end_marker) in enumerate(
                zip(paths, start_markers, end_markers)):
            x_values = sel_trajectory[:, i, 0]
            y_values = sel_trajectory[:, i, 1]
            path.set_data(x_values[:frame + 1], y_values[:frame + 1])
            if frame == 0:
                start_marker.set_data(sel_trajectory[0, i, 0], sel_trajectory[0, i, 1])
            if frame == sel_trajectory.shape[0] - 6:  # Update the end marker
                end_marker.set_data(sel_trajectory[-1, i, 0], sel_trajectory[-1, i, 1])
        return paths + start_markers + end_markers


    print('Creating animation...')
    plt.tight_layout()
    # Create animation
    ani = FuncAnimation(fig, update, frames=range(sel_trajectory.shape[0]),
                        init_func=init, blit=True)

    # Save the animation as mp4 file
    reports_path = 'reports/figures/'
    figure_name = f'{file_name}_trajectory_video.gif'
    writergif = PillowWriter(fps=10)
    ani.save(reports_path + figure_name, dpi=600, writer=writergif)
    print(f'Saved figure {figure_name}')

if __name__ == "__main__":
    main()