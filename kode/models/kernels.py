import jax
import jax.numpy as jnp
import equinox as eqx
from copy import deepcopy


class Kernel(eqx.Module):
    """ Base Kernel class.
    """
    kernel_fun: callable
    params: dict

    def __init__(self, kernel_fun, params):
        """
        :param kernel_fun: A kernel function with the signature f(X, Y,
        parameters)
        :type kernel_fun: function
        :param params: Kernel specific parameters.
        :type params: dict
        """
        self.params = params
        self.kernel_fun = kernel_fun

    def __call__(self, x, y):
        """
        Compute the gram matrix in blocks of 5000 for memory efficiency.
        :param x: Input 1
        :type x: array
        :param y: Input 2
        :type y: array
        :return: Gram matrix between x and y with the specified kernel
        :rtype: array
        """
        K = jnp.zeros((len(x), len(y)))
        start_x, end_x = 0, 5000
        while start_x <= len(x):
            start_y, end_y = 0, 5000
            while start_y <= len(y):
                K_block = jax.vmap(
                    lambda x1: jax.vmap(lambda y1: self.kernel_fun(x1, y1,
                                                              self.params)) \
                        (y[start_y:end_y]))(x[start_x:end_x])
                K = K.at[start_x:end_x, start_y:end_y].set(K_block)

                # update y indices
                start_y += 5000
                end_y += 5000
            # update x indices
            start_x += 5000
            end_x += 5000
        return K


def sqeuclidean_distances(x, y):
    """
    Compute squared euclidean distance between two vectors

    :param x: Input 1
    :type x: array
    :param y: Input 2
    :type y: array
    :return: squared euclidean distance
    :rtype: float
    """
    return jnp.sum((x - y) ** 2)


def rbf_kernel(x, y, params):
    """
    Standard RBF Kernel

    :param x: Input 1
    :type x: array
    :param y: Input 2
    :type y: array
    :param params: Parameters specific to the kernel
    :type params: dict
    :return: Kernel distance
    :rtype: float
    """
    dim = x.shape[-1]
    length_scale = params['length_scale']
    term = jnp.exp(-sqeuclidean_distances(x, y) / (2 * length_scale ** 2))
    norm_factor = jnp.sqrt(2 * jnp.pi * length_scale ** 2) ** dim
    return term / norm_factor


def rbf_mixture(x, y, params):
    dim = x.shape[-1]
    length_scale = params['length_scale']
    dist = sqeuclidean_distances(x, y)
    kvals = jnp.array([jnp.exp(-dist / (2 * l ** 2)) for l in length_scale])
    norm_factor = jnp.array([jnp.sqrt(2 * jnp.pi * l ** 2) ** dim
                             for l in length_scale])
    return jnp.sum(kvals / norm_factor)

def rbf_mixture_unnormalized(x, y, params):
    length_scale = params['length_scale']
    dist = sqeuclidean_distances(x, y)
    kvals = jnp.array([jnp.exp(-dist / (2 * l ** 2)) for l in length_scale])
    return jnp.sum(kvals)


def rq_kernel(x, y, params):
    """
    The Rational Quadratic Kernel.
    K(x, y) = (1 + (x - y) ** 2 / (2 * alpha * l * 2))^-alpha

    :param x: Input 1
    :type x: array
    :param y: Input 2
    :type y: array
    :param params: Parameters specific to the kernel
    :type params: dict
    :return: Kernel distance
    :rtype: float
    """
    scale_mixture = params['scale_mixture']
    length_scale = params['length_scale']
    t1 = sqeuclidean_distances(x, y)
    t2 = (2 * scale_mixture * length_scale ** 2)
    return (1 + (t1 / t2)) ** (-scale_mixture)


def rq_mixture(x, y, params):
    scale_mixture = params['scale_mixture']
    length_scale = params['length_scale']
    t1 = sqeuclidean_distances(x, y)
    t2 = jnp.array([2 * s * l ** 2 for (s, l) in zip(scale_mixture,
                                                   length_scale)])
    kvals = jnp.array([(1 + (t1/t)) ** (-s) for (t, s) in zip(t2,
                                                             scale_mixture)])
    return jnp.sum(kvals)


def laplace_kernel(x, y, params):
    '''
    The laplace kernel. Can be used as a mixture.

    :param params: length scale of laplace kernel
    :type params: dict
    :param x: Input 1
    :param y: Input 2
    :return: Kernel distance
    :type: float
    '''
    length_scale = params['length_scale']
    dist = jnp.sum(jnp.abs(x - y))
    kvals = jnp.array([jnp.exp(-dist / (2 * l ** 2)) for l in length_scale])
    return jnp.sum(kvals)


def matern_kernel(x, y, params):
    '''
    The matern kernel.
    :param x: Input 1
    :param y: Input 2
    :param params: length scale and nu
    :type params: dict
    :return: Kernel distance
    :type: float
    '''
    length_scale = params['length_scale']
    nu = params['nu']
    dist = jnp.sqrt(jnp.sum((x - y) ** 2)) / length_scale
    if nu == 0.5:
        K = jnp.exp(-dist)
    elif nu == 1.5:
        K = dist * jnp.sqrt(3)
        K = (1.0 + K) * jnp.exp(-K)
    elif nu == 2.5:
        K = dist * jnp.sqrt(5)
        K = (1.0 + K + K ** 2 / 3.0) * jnp.exp(-K)
    elif nu == jnp.inf:
        K = jnp.exp(-(dist ** 2) / 2.0)

    else:
        raise NotImplementedError("The matern kernel when nu is not [0.5, "
                                  "1.5, 2.5, jnp.inf] is not implemented")
    return K


def poly_kernel(x, y, params):
    '''
    Computes a mixture of degree d polynomials with the same offset
    :param x:
    :param y:
    :param params: degree and offset
    :return: Kernel distance
    :type: float
    '''
    degree = params['degree']
    offset = params['offset']
    kvals = jnp.array([(x.dot(y) + offset) ** d for d in degree])
    return jnp.sum(kvals)


def ARD_kernel(x, y, params):
    return None


def adjust_kernel_length_scale(kernel, median_dist):
    adjusted_kernel = deepcopy(kernel)
    # adjust length scale based on median distance
    if 'length_scale' in adjusted_kernel.params.keys():
        length_scales = kernel.params['length_scale']
        adjusted_kernel.params['length_scale'] = [ls * median_dist for
                                              ls in length_scales]
    return adjusted_kernel