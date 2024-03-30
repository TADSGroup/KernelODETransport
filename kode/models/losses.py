import equinox as eqx
import jax.numpy as jnp
import optax


# define the MMDLoss class
class MMDLoss(eqx.Module):
    """ Biased empirical estimate of MMD from Gretton et. al. eqn (5)"""
    kernel: eqx.Module

    def __init__(self, kernel, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel

    def __call__(self, x, y):
        XX = self.kernel(x, x)
        YY = self.kernel(y, y)
        XY = self.kernel(x, y) * 2
        return jnp.mean(XX) + jnp.mean(YY) - jnp.mean(XY)


class compute_MMDLoss(eqx.Module):
    """ For computing biased MMD between huge matrices fast. NOT suitable for
    autograd."""
    kernel: eqx.Module

    def __init__(self, kernel, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel

    def __call__(self, x, y, biased=True):
        nX, nY = len(x), len(y)
        XTerm = 1 / nX ** 2
        YTerm = 1 / nY ** 2
        XYTerm = 2 / (nX * nY)

        num_batches_x = jnp.ceil(nX / 10000).astype('int')
        num_batches_y = jnp.ceil(nY / 10000).astype('int')
        kxx, kyy, kxy = 0, 0, 0
        for i in range(num_batches_x):
            start_i, end_i = 10000 * i, 10000 * (i + 1)
            for j in range(num_batches_x):
                start_j, end_j = 10000 * j, 10000 * (j + 1)
                kxx += jnp.sum(self.kernel(x[start_i:end_i], x[start_j:end_j]))

        for i in range(num_batches_y):
            start_i, end_i = 10000 * i, 10000 * (i + 1)
            for j in range(num_batches_y):
                start_j, end_j = 10000 * j, 10000 * (j + 1)
                kyy += jnp.sum(self.kernel(y[start_i:end_i], y[start_j:end_j]))

        for i in range(num_batches_x):
            start_i, end_i = 10000 * i, 10000 * (i + 1)
            for j in range(num_batches_y):
                start_j, end_j = 10000 * j, 10000 * (j + 1)
                kxy += jnp.sum(self.kernel(x[start_i:end_i], y[start_j:end_j]))

        return kxx * XTerm + kyy * YTerm - kxy * XYTerm


def mse_loss(output, target):
    return jnp.mean(jnp.sum((output - target) ** 2, axis=1))