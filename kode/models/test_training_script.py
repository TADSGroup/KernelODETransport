import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import equinox as eqx
import jax.random as jrandom
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from kode.data import load_dataset
from kode.models import transporter, kernels, losses, utils
from kode.visualization import visualize

# load data
key = jrandom.PRNGKey(20)
rng = np.random.RandomState(20)
Y = load_dataset.two_dimensional_data('pinwheel', batch_size=10000, rng=rng)
X = jrandom.normal(key, shape=(10000, 2))

# split into training and testing
X_train, X_test = train_test_split(X, train_size=5000, random_state=2)
Y_train, Y_test = train_test_split(Y, train_size=5000, random_state=2)


# find inducing points and median distance
num_inducing_points = 100
inducing_points, median_distance = utils.find_inducing_points(X_train,
                                                                Y_train,
                                                                num_inducing_points,
                                                                random_state=20)

# model params
model_params = {'length_scale': [0.1 * median_distance]}
num_odes = 2
num_steps = 10
key = jrandom.PRNGKey(10)
model_kernel = kernels.get_kernel('rbf', model_params)

# define the loss function and parameters
loss_params = {'length_scale': [0.15 * median_distance]}
loss_kernel = kernels.get_kernel('laplace', loss_params)
mmd_loss_fun = losses.MMDLoss(loss_kernel)

# initialize the optimizer
optimizer = utils.get_adam_with_exp_decay()

# initialize the model
transport_model = transporter.Transporter(inducing_points, model_kernel,
                                        num_odes, num_steps, key)
gradient_mask = transport_model.get_gradient_mask()
opt_state = optimizer.init(eqx.filter(transport_model.model, gradient_mask))


# train loop
rkhs_strength = 1e-10
h1_strength = 1e-10
for epoch in np.arange(101):
    loss, transport_model.model, opt_state = transport_model.train(
        transport_model.model, X_train, Y_train, mmd_loss_fun,
        rkhs_strength, h1_strength, optimizer, opt_state, verbose=True)

#
# transporter.fit(X_train, Y_train,  101, mmd_loss_fun, rkhs_strength,
#                 h1_strength, 5000, optimizer)


# transform using the model
Y_pred, Y_traj = transport_model.transform(X_test, num_steps=1,
                                           mode='forward', trajectory=True)

# plot the reference, target, predictions
bins = np.linspace(-4, 4, 50)
fig = plt.figure(figsize=(12, 4))
ax1, ax2, ax3 = visualize.plot_2d_distributions(fig, X_test, Y_test, Y_pred,
                                                bins, bins)
plt.show()


# plot the trajectory
fig = plt.figure()
ax = fig.add_subplot(111)
visualize.plot_2d_trajectories(ax, Y_traj, 70, seed=20)
plt.show()