import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import jax.random as jrandom
import numpy as np
import matplotlib.pyplot as plt

from kode.data import utils as data_utils, lotka_volterra
from kode.visualization import visualize
from kode.models import losses, kernels, utils



def parse_args():
    parser = argparse.ArgumentParser(description='Load and evaluate Kernel '
                                                 'ODE model.')
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
    num_steps = args.num_steps
    file_name = args.file_name

    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # load data
    key = jrandom.PRNGKey(22)
    Y_true = np.array([0.83194674, 0.04134147, 1.0823151, 0.03991483])
    lv_sampler = lotka_volterra.DeterministicLotkaVolterra(20)
    conditioned_value, _ = lv_sampler.sample_data(Y_true, seed=20)

    # load model
    model_path = 'models/'
    to_load = data_utils.load_file(model_path + file_name + '.pickle')
    transport_model = to_load['model']
    trajectory_scaler = to_load['trajectory_scaler']
    parameter_scaler = to_load['parameter_scaler']


    # Plot results
    reports_path = 'reports/figures/'


    # Generate data
    conditioned_value_normalized = trajectory_scaler.transform(
        conditioned_value)
    key = jrandom.PRNGKey(6)
    num_trajectories, dim = 10000, 4
    X = jrandom.normal(key, shape=(num_trajectories, dim))
    C = np.full((len(X), 18), conditioned_value_normalized)

    # Calculate posterior
    Y_pred_normalized = transport_model.transform(X, C, num_steps=num_steps)
    Y_pred = parameter_scaler.inverse_transform(Y_pred_normalized)
    Y_true_normalized = parameter_scaler.transform(Y_true.reshape(1, -11))

    # Plot posterior
    figure_name = file_name + '_posterior.png'
    symbols = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$']
    limits = [[-2, 2], [-2, 2], [-2, 2], [-2, 2]]
    fig = visualize.plot_lv_matrix(np.array(Y_pred_normalized), limits,
                            Y_true_normalized[0], symbols)
    plt.suptitle('Posterior for parameters')
    fig.savefig(reports_path + figure_name, bbox_inches='tight', dpi=300)
    print(f'Saved figure {figure_name}')


    # Plot loss curves
    train_mmd_loss = to_load['train_mmd_loss']
    test_mmd_loss = to_load['test_mmd_loss']
    rkhs_norm = to_load['rkhs_norm']
    h1_norm = to_load['h1_norm']
    batch_size = to_load['hyperparameters']['batch_size']


    # figure_name = file_name + '_loss.png'
    # num_iters_per_epoch = int(num_trajectories / batch_size)
    # train_epochs = np.arange(1, len(train_mmd_loss) + 1)
    # test_epochs = np.arange(1, len(test_mmd_loss) + 1) * (
    #     num_iters_per_epoch)
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111)
    # visualize.plot_loss(ax, train_epochs, train_mmd_loss, h1_norm,
    #                     rkhs_norm, test_epochs, test_mmd_loss)
    # fig.savefig(reports_path + figure_name, bbox_inches='tight', dpi=300)
    # print(f'Saved figure {figure_name}')


if __name__ == "__main__":
    main()
