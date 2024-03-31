import argparse
import pprint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import equinox as eqx
import jax.random as jrandom
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from kode.data import load_dataset
from kode.models import transporter, kernels, losses, utils, load_hyperparameters
from kode.visualization import visualize


def parse_args():
    """
    Parse command-line arguments
    :return:
    """

    parser = argparse.ArgumentParser(description="Train Kernel ODE.")

    parser.add_argument("--dataset", type=str, default="pinwheel",
                        help="Dataset name. Use predefined hyperparameters if available.")
    parser.add_argument("--num-anchor-points", type=int, default=100,
                        help="Number of anchor points.")
    parser.add_argument("--loss-kernel-name", type=str, default="laplace",
                        help="Name of the kernel to use for the loss function.")
    parser.add_argument("--loss-kernel-multiplier", type=float, default=0.15,
                        help="Multiplier for the loss kernel's length scale.")
    parser.add_argument("--model-kernel-name", type=str, default="rbf",
                        help="Name of the kernel to use for the model.")
    parser.add_argument("--model-kernel-multiplier", type=float, default=0.1,
                        help="Multiplier for the model kernel's length scale.")
    parser.add_argument("--num-epochs", type=int, default=101,
                        help="Number of epochs for training.")
    parser.add_argument("--num-time-steps", type=int, default=5,
                        help="Number of discrete ODE steps in the model.")
    parser.add_argument("--rkhs-strength", type=float, default=1e-10,
                        help="Strength of the RKHS regularization.")
    parser.add_argument("--h1-strength", type=float, default=1e-10,
                        help="Strength of the H1 regularization.")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Batch size for training.")

    args = parser.parse_args()

    # Attempt to load predefined hyperparameters if a dataset is specified
    if args.dataset:
        predefined_hyperparams = load_hyperparameters.two_dimensional_data(args.dataset)
        # Update only those hyperparameters not explicitly set by the user
        for param, value in predefined_hyperparams.items():
            if getattr(args, param.replace('-', "_"), None) is None:
                setattr(args, param.replace('-', "_"), value)
    return args



if __name__ == "__main__":
    args = parse_args()

    # Convert argpase.Namespace to a dictionary
    hyperparameters = vars(args)

    # print the hyperparams
    pp = pprint.PrettyPrinter(indent=2)
    print('Using hyperparameters:')
    pp.pprint(hyperparameters)


    # Unpack hyperparameters
    NUM_ANCHOR_POINTS = hyperparameters['num_anchor_points']
    LOSS_KERNEL_NAME = hyperparameters['loss_kernel_name']
    LOSS_KERNEL_MULTIPLIER = hyperparameters['loss_len_scale']
    MODEL_KERNEL_NAME = hyperparameters['model_kernel_name']
    MODEL_KERNEL_MULTIPLIER = hyperparameters['model_len_scale']
    NUM_EPOCHS = hyperparameters['num_epochs']
    NUM_TIME_STEPS = hyperparameters['num_time_steps']
    RKHS_STRENGTH = hyperparameters['rkhs_strength']
    H1_STRENGTH = hyperparameters['h1_strength']
    BATCH_SIZE = hyperparameters['batch_size']


    # load data
    key = jrandom.PRNGKey(20)
    rng = np.random.RandomState(20)
    Y = load_dataset.two_dimensional_data('pinwheel', batch_size=20000,
                                          rng=rng)
    X = jrandom.normal(key, shape=(20000, 2))

    # split into training and testing
    X_train, X_test = train_test_split(X, train_size=BATCH_SIZE, random_state=2)
    Y_train, Y_test = train_test_split(Y, train_size=BATCH_SIZE, random_state=2)

    # find inducing points
    inducing_points, median_distance = utils.find_inducing_points(X_train,
                                                                    Y_train,
                                                                    NUM_ANCHOR_POINTS,
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
                                            NUM_TIME_STEPS, key)
    gradient_mask = transport_model.get_gradient_mask()
    opt_state = optimizer.init(eqx.filter(transport_model.model, gradient_mask))

    # train loop
    for epoch in np.arange(NUM_EPOCHS):
        loss, transport_model.model, opt_state = transport_model.train(
            transport_model.model, X_train, Y_train, mmd_loss_fun,
            RKHS_STRENGTH, H1_STRENGTH, optimizer, opt_state, verbose=False)

        # log the loss
        h1_norm = transport_model.h1_seminorm_mixed_norm() * H1_STRENGTH
        rkhs_norm = transport_model.rkhs_norm() * RKHS_STRENGTH
        train_mmd_loss = loss - h1_norm - rkhs_norm

        # calculate validation loss





    # save model
    to_save = {'model': transport_model, 'loss': loss, }

# # transform using the model
# Y_pred, Y_traj = transport_model.transform(X_test, mode='forward', trajectory=True)
#
# # plot the reference, target, predictions
# bins = np.linspace(-4, 4, 50)
# fig = plt.figure(figsize=(12, 4))
# ax1, ax2, ax3 = visualize.plot_2d_distributions(fig, X_test, Y_test, Y_pred,
#                                                 bins, bins)
# plt.show()