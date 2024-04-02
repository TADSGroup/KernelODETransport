import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import jax.random as jrandom
import numpy as np
import matplotlib.pyplot as plt

from kode.data import utils as data_utils, load_dataset
from kode.visualization import visualize
from kode.models import losses, kernels, utils



def parse_args():
    parser = argparse.ArgumentParser(description='Load and evaluate Kernel '
                                                 'ODE model.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to use.')
    parser.add_argument("--file-name", type=str, required=True,
                        help="Name of the file name with the trained model. "
                             "It will be loaded from models/ folder.")
    parser.add_argument('--num-steps', type=int, default=10,
                        help='Number of discrete steps for ODE solver.')
    parser.add_argument("--cuda-device", type=str, default='0',
                        help="CUDA device ID.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    num_steps = args.num_steps
    file_name = args.file_name

    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # load data
    key = jrandom.PRNGKey(22)
    rng = np.random.RandomState(22)
    Y = load_dataset.two_dimensional_data(dataset, batch_size=20000, rng=rng)
    C, _ = np.split(Y, 2, axis=1)
    X = jrandom.normal(key, shape=(20000, 1))

    # load model
    model_path = 'models/'
    to_load = data_utils.load_file(model_path + file_name + '.pickle')
    transport_model = to_load['model']


    # Plot results
    reports_path = 'reports/figures/'

    # plot 2d histogram
    conditional_pred = transport_model.transform(X, C, num_steps=10)
    test_pred = np.column_stack((C, conditional_pred))
    train_data = np.column_stack((C, X))
    figure_name = file_name + '_2d_histogram_predictions.png'
    bins = np.linspace(-4, 4, 50)
    fig = plt.figure(figsize=(12, 4))
    ax1, ax2, ax3 = visualize.plot_2d_distributions(fig, train_data, Y,
                                                test_pred, bins, bins)
    ax1.set_title('Train')
    ax2.set_title('Test')
    ax3.set_title('Predicted')
    fig.savefig(reports_path + figure_name, bbox_inches='tight', dpi=300)
    print(f'Saved figure {figure_name}')


    # plot predictions
    figure_name = file_name + '_1d_histogram_predictions.png'
    conditioned_value = [[2]]
    bins = np.linspace(-4, 4, 50)
    C_full = np.full((len(X), 1), conditioned_value)
    Y_pred = transport_model.transform(X, C_full, num_steps=10)
    Y_true = utils.find_points_in_same_bin(conditioned_value, C,
                                           Y[:, 1], bins)
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    ax = visualize.plot_conditional_density(ax, Y_true, Y_pred, labels=['True',
                                                               'Predicted'])
    fig.savefig(reports_path + figure_name, bbox_inches='tight', dpi=300)
    print(f'Saved figure {figure_name}')


    # plot loss
    train_mmd_loss = to_load['train_mmd_loss']
    test_mmd_loss = to_load['test_mmd_loss']
    rkhs_norm = to_load['rkhs_norm']
    h1_norm = to_load['h1_norm']

    epochs = np.arange(0, len(train_mmd_loss))
    figure_name = file_name + '_loss.png'
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    visualize.plot_loss(ax, epochs, train_mmd_loss, h1_norm, rkhs_norm,
                        epochs, test_mmd_loss)
    fig.savefig(reports_path + figure_name, bbox_inches='tight', dpi=300)
    print(f'Saved figure {figure_name}')



if __name__ == "__main__":
    main()
