import argparse
import pprint
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import equinox as eqx
import jax.random as jrandom
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from kode.data import load_dataset
from kode.models import transporter, kernels, losses, utils, load_hyperparameters
from kode.visualization import visualize
from kode.data import utils as data_utils


def parse_args():
    """
    Parse command-line arguments
    :return:
    """

    parser = argparse.ArgumentParser(description="Train Kernel ODE.")

    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name. Use predefined hyperparameters if available.")
    parser.add_argument("--num-inducing-points", type=int, default=10,
                        help="Number of inducing points.")
    parser.add_argument("--loss-kernel-name", type=str, default="laplace",
                        help="Name of the kernel to use for the loss function.")
    parser.add_argument("--loss-kernel-multiplier", type=float, default=0.15,
                        help="Multiplier for the loss kernel's length scale.")
    parser.add_argument("--model-kernel-name", type=str, default="rbf",
                        help="Name of the kernel to use for the model.")
    parser.add_argument("--model-kernel-multiplier", type=float, default=0.1,
                        help="Multiplier for the model kernel's length scale.")
    parser.add_argument("--num-epochs", type=int, default=11,
                        help="Number of epochs for training.")
    parser.add_argument("--num-odes", type=int, default=2,
                        help="Number of discrete ODE steps in the model.")
    parser.add_argument("--num-steps", type=int, default=10,
                        help="Number of time-steps for each ODE step.")
    parser.add_argument("--rkhs-strength", type=float, default=1e-10,
                        help="Strength of the RKHS regularization.")
    parser.add_argument("--h1-strength", type=float, default=1e-10,
                        help="Strength of the H1 regularization.")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Batch size for training.")
    parser.add_argument("--save-name", type=str, default=None,
                        help="Name of the file to save.")

    args = parser.parse_args()

    # Attempt to load predefined hyperparameters when a dataset is specified
    predefined_hyperparams = load_hyperparameters.two_dimensional_data(args.dataset)
    if predefined_hyperparams is None:
        raise ValueError("Please specify a valid dataset name. Options are "
                         "pinwheel, 2spirals, moons, 8gaussians, circles, "
                         "swissroll, checkerboard.")


    user_args = {k: v for k, v in vars(args).items() if parser.get_default(
        k) != v}
    # Start with predefined hyperparams
    final_hyperparams = predefined_hyperparams.copy()
    # Update user-specified values that differ from the argument parser
    final_hyperparams.update(user_args)

    return argparse.Namespace(**final_hyperparams)



if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    save_file_name = args.save_name

    # Convert argpase.Namespace to a dictionary
    hyperparameters = vars(args)

    # print the hyperparams
    pp = pprint.PrettyPrinter(indent=2)
    print('Using hyperparameters:')
    pp.pprint(hyperparameters)

    # Unpack hyperparameters
    NUM_INDUCING_POINTS = hyperparameters['num_inducing_points']
    LOSS_KERNEL_NAME = hyperparameters['loss_kernel_name']
    LOSS_KERNEL_MULTIPLIER = hyperparameters['loss_len_scale']
    MODEL_KERNEL_NAME = hyperparameters['model_kernel_name']
    MODEL_KERNEL_MULTIPLIER = hyperparameters['model_len_scale']
    NUM_EPOCHS = hyperparameters['num_epochs']
    NUM_ODES = hyperparameters['num_odes']
    NUM_STEPS = hyperparameters['num_steps']
    RKHS_STRENGTH = hyperparameters['rkhs_strength']
    H1_STRENGTH = hyperparameters['h1_strength']
    BATCH_SIZE = hyperparameters['batch_size']


    # load data
    key = jrandom.PRNGKey(20)
    rng = np.random.RandomState(20)
    Y = load_dataset.two_dimensional_data(dataset, batch_size=20000,
                                          rng=rng)
    X = jrandom.normal(key, shape=(20000, 2))

    # split into training and testing
    X_train, X_test = train_test_split(X, train_size=BATCH_SIZE, random_state=2)
    Y_train, Y_test = train_test_split(Y, train_size=BATCH_SIZE, random_state=2)

    # calculate time to train model
    start_time = time.time()

    # find inducing points
    inducing_points, median_distance = utils.find_inducing_points(X_train,
                                                                    Y_train,
                                                                    NUM_INDUCING_POINTS,
                                                                    random_state=20,
                                                                    mode='full_batch')

    # define model
    model_params = {'length_scale': MODEL_KERNEL_MULTIPLIER * median_distance}
    key = jrandom.PRNGKey(10)
    model_kernel = kernels.get_kernel(MODEL_KERNEL_NAME, model_params)


    # define the loss function and parameters
    loss_params = {'length_scale': LOSS_KERNEL_MULTIPLIER * median_distance}
    loss_kernel  = kernels.get_kernel(LOSS_KERNEL_NAME, loss_params)
    mmd_loss_fun = losses.MMDLoss(loss_kernel)

    # initialize the optimizer
    optimizer = utils.get_adam_with_exp_decay()


    # initialize the model
    transport_model = transporter.Transporter(inducing_points, model_kernel,
                                            NUM_ODES, NUM_STEPS, key)
    gradient_mask = transport_model.get_gradient_mask()
    opt_state = optimizer.init(eqx.filter(transport_model.model, gradient_mask))

    # train loop
    h1_norm_list = []
    rkhs_norm_list = []
    train_loss_list = []
    test_loss_list = []
    for epoch in tqdm(np.arange(NUM_EPOCHS)):
        loss, transport_model.model, opt_state = transport_model.train(
            transport_model.model, X_train, Y_train, mmd_loss_fun,
            RKHS_STRENGTH, H1_STRENGTH, optimizer, opt_state, verbose=True)

        # calculate train the loss
        h1_norm = transport_model.model.h1_seminorm_mixed_norm() * H1_STRENGTH
        rkhs_norm = transport_model.model.rkhs_norm() * RKHS_STRENGTH
        train_mmd_loss = loss - h1_norm - rkhs_norm

        # calculate validation loss
        test_pred = transport_model.transform(X_test, num_steps=10)
        test_mmd_loss = mmd_loss_fun(test_pred, Y_test)

        # log the loss
        h1_norm_list.append(h1_norm)
        rkhs_norm_list.append(rkhs_norm)
        train_loss_list.append(train_mmd_loss)
        test_loss_list.append(test_mmd_loss)

    # calcualte time_take
    end_time = time.time()
    train_time = end_time - start_time
    print(f'Training completed in {train_time:.02f}s')

    # save model and log the training loss
    to_save = {'model': transport_model, 'h1_norm': h1_norm_list, 'rkhs_norm':
        rkhs_norm_list, 'train_mmd_loss': train_loss_list, 'test_mmd_loss':
        test_loss_list, 'hyperparameters': hyperparameters, 'train_time':
        train_time}


    # save the model
    print('Saving the model and metrics...')
    save_path = 'models/'
    if save_file_name is None:
        save_file_name = time.strftime("%Y-%m-%d_%H-%M") + f"_{dataset}"
    data_utils.save_file(to_save, save_path + save_file_name + '.pickle',
                         overwrite=True)


    # plot predictions and trajectory
    print('Plotting model results and saving...')
    train_pred = transport_model.transform(X_train, num_steps=10)
    test_pred, test_traj = transport_model.transform(X_test, num_steps=20,
                                                    trajectory=True)

    # Plot results
    reports_path = 'reports/figures/'

    # plot predictions
    figure_name = save_file_name + '_predictions.png'
    bins = np.linspace(-4, 4, 50)
    fig = plt.figure(figsize=(12, 4))
    ax1, ax2, ax3 = visualize.plot_2d_distributions(fig, X_test, Y_test,
                                                test_pred, bins, bins)
    fig.savefig(reports_path + figure_name, bbox_inches='tight', dpi=300)


    # plot trajectory
    figure_name = save_file_name + '_trajectories.png'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = visualize.plot_2d_trajectories(ax, test_traj, 70, seed=20)
    fig.savefig(reports_path + figure_name, bbox_inches='tight', dpi=300)

    # plot loss curves
    figure_name = save_file_name + '_loss.png'
    epochs = np.arange(0, len(train_loss_list))
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    visualize.plot_loss(ax, epochs, train_loss_list, h1_norm_list,
                        rkhs_norm_list, epochs, test_loss_list)
    fig.savefig(reports_path + figure_name, bbox_inches='tight', dpi=300)