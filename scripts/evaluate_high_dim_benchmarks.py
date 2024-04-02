import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import jax.random as jrandom
import numpy as np
import matplotlib.pyplot as plt

from kode.data import utils as data_utils, load_dataset
from kode.visualization import visualize
from kode.models import losses, kernels



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
    parser.add_argument("--cuda-device", type=str, default='2',
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
    data = load_dataset.high_dimensional_data(dataset)
    Y = data.tst.x[:10000]
    num_data, dim = Y.shape
    X = jrandom.normal(key, shape=(num_data, dim))

    # load model
    model_path = 'models/'
    to_load = data_utils.load_file(model_path + file_name + '.pickle')
    transport_model = to_load['model']

    # model predictions
    predictions = transport_model.transform(X, num_steps=num_steps)

    # Plot results
    reports_path = 'reports/figures/'

    # plot true marginals
    figure_name = file_name + '_true_marginals.png'
    fig = visualize.plot_multidim_marginals(Y, vmin=0, vmax=0.15,
                                            title=f'{dataset}: True Marginals')
    fig.savefig(reports_path + figure_name, bbox_inches='tight', dpi=300)
    print(f'Saved figure {figure_name}')

    # plot predictions
    figure_name = file_name + '_predicted_marginals.png'
    fig = visualize.plot_multidim_marginals(predictions, vmin=0, vmax=0.15,
                                            title=f'{dataset}: Predicted '
                                                  f'Marginals')
    fig.savefig(reports_path + figure_name, bbox_inches='tight', dpi=300)
    print(f'Saved figure {figure_name}')


    # Plot loss
    train_mmd_loss = to_load['train_mmd_loss']
    test_mmd_loss = to_load['test_mmd_loss']
    rkhs_norm = to_load['rkhs_norm']
    h1_norm = to_load['h1_norm']
    batch_size = to_load['hyperparameters']['batch_size']

    figure_name = file_name + '_loss.png'
    num_iters_per_epoch = int((num_data) / batch_size)
    train_epochs = np.arange(1, len(train_mmd_loss) + 1)
    test_epochs = np.arange(1, len(test_mmd_loss) + 1) * (
            num_iters_per_epoch)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    visualize.plot_loss(ax, train_epochs, train_mmd_loss, h1_norm,
                        rkhs_norm, test_epochs, test_mmd_loss)
    fig.savefig(reports_path + figure_name, bbox_inches='tight', dpi=300)
    print(f'Saved figure {figure_name}')



    # Compute normalized MMD
    params = {'length_scale': 1 / np.sqrt(2)}
    rbf_kernel = kernels.get_kernel('rbf', params)
    mmd_loss_fun = losses.compute_MMDLoss(rbf_kernel)
    mmd_X_Y = mmd_loss_fun(X, Y)
    mmd_predictions_Y = mmd_loss_fun(predictions, Y)
    normalized_mmd = mmd_predictions_Y / mmd_X_Y

    # Print metrics
    train_time = to_load['train_time']
    print(f'Dataset: {dataset}, Total training time: {train_time:0.2f}s, '
          f'Normalized MMD: {normalized_mmd:0.2e}')


if __name__ == "__main__":
    main()
