import os
import unittest
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

from kode.models import transporter, kernels, losses, utils, ode_models

class test_ode_models(unittest.TestCase):
    def test_trajectory(self):
        num_points, dim = 2, 2
        k1, k2, k3 = jrandom.split(jrandom.PRNGKey(10), 3)
        X = jrandom.normal(k1, shape=(num_points, dim))

        # define the model
        inducing_points = jrandom.normal(k1, shape=(3, dim))
        num_odes = 2
        model_params = {'length_scale': [0.1]}
        model_kernel = kernels.get_kernel('rbf', model_params)
        kernel_ode = ode_models.KernelODE(inducing_points, model_kernel,
                                          num_odes, key=k2)


        # get output
        num_steps = 5
        trajectory = kernel_ode(X, num_steps)
        predictions = trajectory[-1]


        assert predictions.shape == X.shape, 'Dimensions of the predictions do not match'
        assert trajectory.shape == (num_odes * num_steps + 1, num_points,
                                    dim),  ('Dimensions of the trajectories '
                                            'do not match')


    def test_transporter(self):
        ''' Check that output shapes match. If you want to predict on a finer
        mesh, the output trajectories still match.'''
        num_points, dim = 2, 2
        k1, k2, k3 = jrandom.split(jrandom.PRNGKey(10), 3)
        X = jrandom.normal(k1, shape=(num_points, dim))

        # define the model
        inducing_points = jrandom.normal(k1, shape=(3, dim))
        num_odes = 2
        num_steps = 5
        model_params = {'length_scale': [0.1]}
        model_kernel = kernels.get_kernel('rbf', model_params)
        transport_model =  transporter.Transporter(inducing_points, model_kernel,
                                          num_odes, num_steps, key=k2)

        # predict using pre-defined num steps
        predictions, trajectory = transport_model.transform(X, trajectory=True)
        assert predictions.shape == X.shape, 'Dimensions of the predictions do not match'
        assert trajectory.shape == (num_odes * num_steps + 1, num_points,
                                    dim),  ('Dimensions of the trajectories '
                                            'do not match')

        # Predict using new number of steps
        new_num_steps = 10
        predictions, trajectory = transport_model.transform(X,
                                                            new_num_steps,
                                                            trajectory=True)
        assert predictions.shape == X.shape, 'Dimensions of the predictions do not match'
        assert trajectory.shape == (num_odes * new_num_steps + 1, num_points,
                                    dim), ('Dimensions of the trajectories '
                                           'with new number of steps do not '
                                           'match')


    def test_conditional_transporter(self):
        num_points, orig_dim, conditioned_dim = 2, 2, 3
        k1, k2, k3 = jrandom.split(jrandom.PRNGKey(10), 3)

        X = jrandom.normal(k1, shape=(num_points, orig_dim))
        C = jrandom.normal(k2, shape=(num_points, conditioned_dim))

        # define the model
        inducing_points = jrandom.normal(k1, shape=(3, orig_dim + conditioned_dim))
        model_params = {'length_scale': [0.1]}
        num_odes = 2
        num_steps = 5
        model_kernel = kernels.get_kernel('rbf', model_params)
        transport_model = transporter.Conditional_Transporter(
            inducing_points, orig_dim, model_kernel, num_odes,
            num_steps, key=k2)

        # predict using pre-defined steps
        predictions, trajectory = transport_model.transform(X, C, trajectory=True)
        assert predictions.shape == X.shape, 'Dimensions of the predictions do not match'
        assert trajectory.shape == (num_odes * num_steps + 1, num_points,
                                    orig_dim),  ('Dimensions of the '
                                               'trajectories '
                                            'do not match')

        # predict using new number of steps
        new_num_steps = 10
        predictions, trajectory = transport_model.transform(X, C,
                                                            new_num_steps,
                                                            trajectory=True)
        assert predictions.shape == X.shape, ('Dimensions of the predictions '
                                              'do not match with new number of steps')
        assert trajectory.shape == (num_odes * new_num_steps + 1, num_points,
                                    orig_dim), ('Dimensions of the '
                                               'trajectories'
                                           'with new number of steps do not '
                                           'match')