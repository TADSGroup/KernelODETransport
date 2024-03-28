from sklearn.cluster import MiniBatchKMeans, KMeans
from scipy.spatial.distance import pdist
import numpy as np
import equinox as eqx
import optax
import jax.numpy as jnp

def find_inducing_points(X, Y, num_inducing_points, median_dist=True,
                         random_state=None, mode='mini_batch'):

    # use k-means to find inducing points
    if mode == 'mini_batch':
        kmeans = MiniBatchKMeans(n_clusters=num_inducing_points,
                                 batch_size=2048, n_init=1, max_iter=100,
                                 verbose=0, max_no_improvement=20,
                                 random_state=random_state)
    elif mode == 'full_batch':
        kmeans = KMeans(n_clusters=num_inducing_points,
                        random_state=random_state)
    kmeans.fit(np.vstack((X, Y)))
    inducing_points = jnp.array(kmeans.cluster_centers_)
    if median_dist is True:
        dist = find_median_distance(inducing_points)
        return inducing_points, dist
    else:
        return inducing_points


def find_median_distance(X):
    # find median distance based on L1 distance between inducing points
    distances = pdist(X, 'minkowski', p=1)
    median_dist = np.median(distances)
    return median_dist



def train(model, X, Y, grad_loss_fun, loss_fun, rkhs_strength, h1_strength,
          optimizer, opt_state, verbose=True):
    loss, grads = grad_loss_fun(model, X, Y, loss_fun,
                                rkhs_strength, h1_strength)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    if verbose:
        rkhs_norm = model.rkhs_norm() * rkhs_strength
        h1_norm = model.h1_seminorm_mixed_norm() * h1_strength
        print(f'MMD loss: {loss - rkhs_norm - h1_norm}, RKHS penalty:'
              f' {rkhs_norm}, '
              f'H1 penalty: {h1_norm}')
    return loss, opt_state


def get_adam_with_exp_decay():
    exponential_decay_scheduler = optax.exponential_decay(init_value=0.1,
                                                          transition_steps=5001,
                                                          decay_rate=0.1,
                                                          transition_begin=int(
                                                              4001 * 0.1),
                                                          staircase=False)
    optim = optax.adam(learning_rate=exponential_decay_scheduler)
    return optim