import argparse
import pprint
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import equinox as eqx
import jax.random as jrandom
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    parser.add_argument("--num-inducing-points", type=int, default=10,
                        help="Number of inducing points.")
    parser.add_argument("--output-dim", type=int, default=1,
                        help="Dimension of the output data.")
    parser.add_argument("--loss-kernel-name", type=str, default="laplace",
                        help="Name of the kernel to use for the loss function.")
    parser.add_argument("--loss-kernel-multiplier", type=float, default=0.15,
                        help="Multiplier for the loss kernel's length scale.")
    parser.add_argument("--model-kernel-name", type=str, default="rbf",
                        help="Name of the kernel to use for the model.")
    parser.add_argument("--model-kernel-multiplier", type=float, default=0.05,
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
    predefined_hyperparams = load_hyperparameters.lotka_volterra()
    user_args = {k: v for k, v in vars(args).items() if parser.get_default(
        k) != v}
    # Start with predefined hyperparams
    final_hyperparams = predefined_hyperparams.copy()
    # Update user-specified values that differ from the argument parser
    final_hyperparams.update(user_args)
    # add save file name to the dict
    final_hyperparams["save_name"] = args.save_name

    return argparse.Namespace(**final_hyperparams)



if __name__ == "__main__":
    args = parse_args()
    save_file_name = args.save_name

    # Convert argpase.Namespace to a dictionary
    hyperparameters = vars(args)

    # print the hyperparams
    pp = pprint.PrettyPrinter(indent=2)
    print('Using hyperparameters:')
    pp.pprint(hyperparameters)

    # Unpack hyperparameters
    NUM_INDUCING_POINTS = hyperparameters['num_inducing_points']
    OUTPUT_DIM = hyperparameters['output_dim']
    LOSS_KERNEL_NAME = hyperparameters['loss_kernel_name']
    LOSS_KERNEL_MULTIPLIER = hyperparameters['loss_kernel_multiplier']
    MODEL_KERNEL_NAME = hyperparameters['model_kernel_name']
    MODEL_KERNEL_MULTIPLIER = hyperparameters['model_kernel_multiplier']
    NUM_EPOCHS = hyperparameters['num_epochs']
    NUM_ODES = hyperparameters['num_odes']
    NUM_STEPS = hyperparameters['num_steps']
    RKHS_STRENGTH = hyperparameters['rkhs_strength']
    H1_STRENGTH = hyperparameters['h1_strength']
    BATCH_SIZE = hyperparameters['batch_size']


    # load data
    key = jrandom.PRNGKey(20)
    num_trajectories, t_steps = 51000, 20
    lv_sampler, Y, C, t, Y_scaler, C_scaler = load_dataset.lotka_volterra(
        num_trajectories=num_trajectories, time_steps=t_steps)
    X = jrandom.normal(key, shape=(num_trajectories, OUTPUT_DIM))

    # concatenate
    Y_C = np.column_stack((C, Y))

    # split into training and testing
    X_train, X_test = train_test_split(X, train_size=50000,
                                       random_state=2)
    Y_train, Y_test = train_test_split(Y_C, train_size=50000,
                                       random_state=2)
    C_train, _ = np.split(Y_train, [-4], axis=1)
    C_test, _ = np.split(Y_test, [-4], axis=1)

    # calculate time to train modelr
    start_time = time.time()

    # find inducing points
    X_C_train = np.column_stack((X_train, C_train))
    inducing_points, median_distance = utils.find_inducing_points(X_C_train,
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
    transport_model = transporter.Conditional_Transporter(inducing_points,
                                              OUTPUT_DIM, model_kernel,
                                            NUM_ODES, NUM_STEPS, key)
    gradient_mask = transport_model.get_gradient_mask()
    opt_state = optimizer.init(eqx.filter(transport_model.model, gradient_mask))

    # train loop
    h1_norm_list = []
    rkhs_norm_list = []
    train_loss_list = []
    test_loss_list = []
    for epoch in tqdm(np.arange(NUM_EPOCHS)):
        for X_batch, Y_batch, C_batch in data_utils.DataLoader((X_train,
                                                                Y_train,
                                                                C_train),
                                                               batch_size=BATCH_SIZE,
                                                               seed=20):
            loss, transport_model.model, opt_state = transport_model.train(
                transport_model.model, X_batch, Y_batch, C_batch, mmd_loss_fun,
                RKHS_STRENGTH, H1_STRENGTH, optimizer, opt_state, verbose=True)

            # calculate train loss
            h1_norm = transport_model.model.h1_seminorm_mixed_norm() * H1_STRENGTH
            rkhs_norm = transport_model.model.rkhs_norm() * RKHS_STRENGTH
            train_mmd_loss = loss - h1_norm - rkhs_norm

            # log the loss
            h1_norm_list.append(h1_norm)
            rkhs_norm_list.append(rkhs_norm)
            train_loss_list.append(train_mmd_loss)

        # calculate validation loss and log
        conditional_pred = transport_model.transform(X_test, C_test,
                                                   num_steps=10)
        test_pred = np.column_stack((C_test, conditional_pred))
        test_mmd_loss = mmd_loss_fun(test_pred, Y_test)
        test_loss_list.append(test_mmd_loss)

    # calcualte time_take
    end_time = time.time()
    train_time = end_time - start_time
    print(f'Training completed in {train_time:.02f}s')

    # save model and log the training loss
    to_save = {'model': transport_model, 'h1_norm': h1_norm_list, 'rkhs_norm':
        rkhs_norm_list, 'train_mmd_loss': train_loss_list, 'test_mmd_loss':
        test_loss_list, 'hyperparameters': hyperparameters, 'train_time':
        train_time, 'trajectory_scaler': C_scaler, 'parameter_scaler':
        Y_scaler}


    # save the model
    print('Saving the model and metrics...')
    save_path = 'models/'
    if save_file_name is None:
        save_file_name = time.strftime("%Y-%m-%d_%H-%M") + (f"_conditi"
                                                            f"onal_lotka_volterra")
    data_utils.save_file(to_save, save_path + save_file_name + '.pickle',
                         overwrite=True)


    # plot prediction
    print('Plotting model results and saving...')
    predictions = transport_model.transform(X_test, C_test, num_steps=20)

    # Plot results
    reports_path = 'reports/figures/'

    # Plot posterior for an observed test point
    figure_name = save_file_name + '_posterior.png'
    Y_true = np.array([0.83194674, 0.04134147, 1.0823151, 0.03991483])
    conditioned_value, _ = lv_sampler.sample_data(Y_true, seed=20)
    conditioned_value_normalized = C_scaler.transform(conditioned_value)

    key = jrandom.PRNGKey(5)
    X_obs = jrandom.normal(key, shape=(10000, OUTPUT_DIM))
    C_obs = np.full((len(X_obs), 18), conditioned_value_normalized)


    # Calculate posterior
    Y_pred_normalized = transport_model.transform(X_obs, C_obs, num_steps=10)
    Y_pred = Y_scaler.inverse_transform(Y_pred_normalized)
    Y_true_normalized = Y_scaler.transform(Y_true.reshape(1, -1))
    symbols = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$']
    # limits = [[0.5, 1.3], [0.02, 0.07], [0.7, 1.5], [0.025, 0.065]]
    limits = [[-2, 2], [-2, 2], [-2, 2], [-2, 2]]
    fig = visualize.plot_lv_matrix(np.array(Y_pred_normalized), limits,
                            Y_true_normalized[0], symbols)
    plt.suptitle('Posterior for parameters')
    fig.savefig(reports_path + figure_name, bbox_inches='tight', dpi=300)



    # plot loss curves
    figure_name = save_file_name + '_loss.png'
    num_iters_per_epoch = int(num_trajectories / BATCH_SIZE)
    train_epochs = np.arange(1, len(train_loss_list) + 1)
    test_epochs = np.arange(1, len(test_loss_list) + 1) * (
            num_iters_per_epoch)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    visualize.plot_loss(ax, train_epochs, train_loss_list, h1_norm_list,
                        rkhs_norm_list, test_epochs, test_loss_list)
    fig.savefig(reports_path + figure_name, bbox_inches='tight', dpi=300)



