import numpy as np

def two_dimensional_data(dataset_name):
    """
    Returns predefined hyperaprameters for specified datasets
    """

    predefined_hyperparams = {
        'pinwheel': {
            'num_inducing_points': 100,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.1,
            'num_epochs': 501,
            'num_odes': 5,
            'num_steps': 10,
            'rkhs_strength': 1e-6,
            'h1_strength': 1e-6,
            'batch_size': 5000,
        },
        '2spirals': {
            'num_inducing_points': 100,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.1,
            'num_epochs': 501,
            'num_odes': 5,
            'num_steps': 10,
            'rkhs_strength': 1e-9,
            'h1_strength': 1e-8,
            'batch_size': 5000,
        },
        'moons': {
            'num_inducing_points': 150,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.1,
            'num_epochs': 501,
            'num_odes': 3,
            'num_steps': 10,
            'rkhs_strength': 1e-9,
            'h1_strength': 1e-6,
            'batch_size': 5000,
        },
        '8gaussians': {
            'num_inducing_points': 150,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.1,
            'num_epochs': 701,
            'num_odes': 3,
            'num_steps': 10,
            'rkhs_strength': 1e-6,
            'h1_strength': 1e-5,
            'batch_size': 5000,
        },
        'circles': {
            'num_inducing_points': 100,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.1,
            'num_epochs': 501,
            'num_odes': 5,
            'num_steps': 10,
            'rkhs_strength': 1e-6,
            'h1_strength': 1e-7,
            'batch_size': 5000,
        },
        'swissroll': {
            'num_inducing_points': 100,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.1,
            'num_epochs': 501,
            'num_odes': 5,
            'num_steps': 10,
            'rkhs_strength': 1e-7,
            'h1_strength': 1e-6,
            'batch_size': 5000,
        },
        'checkerboard': {
            'num_inducing_points': 150,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.05,
            'num_epochs': 1001,
            'num_odes': 4,
            'num_steps': 10,
            'rkhs_strength': 1e-8,
            'h1_strength': 1e-8,
            'batch_size': 10000,
        },
    }

    return predefined_hyperparams.get(dataset_name, None)


def two_dimensional_data_conditional(dataset_name):
    """
    Returns predefined hyperaprameters for specified datasets
    """

    predefined_hyperparams = {
        'pinwheel': {
            'num_inducing_points': 100,
            'output_dim': 1,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.1,
            'num_epochs': 601,
            'num_odes': 4,
            'num_steps': 10,
            'rkhs_strength': 1e-8,
            'h1_strength': 1e-8,
            'batch_size': 5000,
        },
        '2spirals': {
            'num_inducing_points': 100,
            'output_dim': 1,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.1,
            'num_epochs': 501,
            'num_odes': 5,
            'num_steps': 10,
            'rkhs_strength': 1e-9,
            'h1_strength': 1e-8,
            'batch_size': 5000,
        },
        'moons': {
            'num_inducing_points': 150,
            'output_dim': 1,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.1,
            'num_epochs': 501,
            'num_odes': 3,
            'num_steps': 10,
            'rkhs_strength': 1e-9,
            'h1_strength': 1e-6,
            'batch_size': 5000,
        },
        '8gaussians': {
            'num_inducing_points': 150,
            'output_dim': 1,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.1,
            'num_epochs': 701,
            'num_odes': 3,
            'num_steps': 10,
            'rkhs_strength': 1e-6,
            'h1_strength': 1e-5,
            'batch_size': 5000,
        },
        'circles': {
            'num_inducing_points': 100,
            'output_dim': 1,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.1,
            'num_epochs': 501,
            'num_odes': 5,
            'num_steps': 10,
            'rkhs_strength': 1e-6,
            'h1_strength': 1e-7,
            'batch_size': 5000,
        },
        'swissroll': {
            'num_inducing_points': 100,
            'output_dim': 1,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.1,
            'num_epochs': 601,
            'num_odes': 4,
            'num_steps': 10,
            'rkhs_strength': 1e-8,
            'h1_strength': 1e-8,
            'batch_size': 5000,
        },
        'checkerboard': {
            'num_inducing_points': 150,
            'output_dim': 1,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.15,
            'model_len_scale': 0.05,
            'num_epochs': 1001,
            'num_odes': 4,
            'num_steps': 10,
            'rkhs_strength': 1e-8,
            'h1_strength': 1e-8,
            'batch_size': 10000,
        },
    }

    return predefined_hyperparams.get(dataset_name, None)


def high_dimensional_data(dataset_name):
    """
    Returns predefined hyperaprameters for specified datasets
    """

    predefined_hyperparams = {
        'power': {
            'num_inducing_points': 1000,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.1,
            'model_len_scale': 0.1,
            'num_epochs': 11,
            'num_odes': 2,
            'num_steps': 10,
            'rkhs_strength': 1e-11,
            'h1_strength': 1e-11,
            'batch_size': 2048,
        },
        'gas': {
            'num_inducing_points': 2000,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.1,
            'model_len_scale': 0.1,
            'num_epochs': 11,
            'num_odes': 4,
            'num_steps': 10,
            'rkhs_strength': 1e-11,
            'h1_strength': 1e-11,
            'batch_size': 2048,
        },
        'hepmass': {
            'num_inducing_points': 2000,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.1,
            'model_len_scale': 0.05,
            'num_epochs': 20,
            'num_odes': 2,
            'num_steps': 10,
            'rkhs_strength': 1e-11,
            'h1_strength': 1e-11,
            'batch_size': 1024,
        },
        'miniboone': {
            'num_inducing_points': 1000,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.1,
            'model_len_scale': 0.05,
            'num_epochs': 20,
            'num_odes': 2,
            'num_steps': 10,
            'rkhs_strength': 1e-11,
            'h1_strength': 1e-11,
            'batch_size': 1024,
        },
        'bsds300': {
            'num_inducing_points': 2000,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'rbf',
            'loss_len_scale': 0.1,
            'model_len_scale': 0.05,
            'num_epochs': 7,
            'num_odes': 2,
            'num_steps': 10,
            'rkhs_strength': 1e-11,
            'h1_strength': 1e-11,
            'batch_size': 512,
        }
    }

    return predefined_hyperparams.get(dataset_name, None)


def lotka_volterra():
    """
    Returns predefined hyperaprameters for specified datasets
    """

    predefined_hyperparams = {
        'lotka_volterra': {
            'num_inducing_points': 1000,
            'output_dim': 4,
            'loss_kernel_name': 'laplace',
            'model_kernel_name': 'laplace',
            'loss_kernel_multiplier': 0.1,
            'model_kernel_multiplier': 0.05,
            'num_epochs': 401,
            'num_odes': 1,
            'num_steps': 10,
            'rkhs_strength': 1e-11,
            'h1_strength': 1e-11,
            'batch_size': 2048,
        }
    }

    return predefined_hyperparams.get('lotka_volterra', None)