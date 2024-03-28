from sklearn.base import BaseEstimator, TransformerMixin
import equinox as eqx
import jax.tree_util as jtu
import functools as ft
import jax.numpy as jnp
from copy import deepcopy

from kode.models.ode_models import KernelODE, Conditional_KernelODE
from kode.data import utils


class Transporter(BaseEstimator, TransformerMixin):
    def __init__(self, inducing_points, kernel, num_odes, key):
        self.inducing_points = inducing_points
        self.num_odes = num_odes
        self.kernel = deepcopy(kernel)
        self.key = key
        self.model = KernelODE(self.inducing_points,
                                             self.kernel,
                                             self.num_odes,
                                             key=self.key)
        # Note that we are time-independent in num_odes == 1
        if self.num_odes == 1:
            print('Note: We are in the time-independent regime. For '
                  'time-dependent regime, num_odes > 1.')

    def fit(self, X, Y, num_epochs, loss_fun, rkhs_strength,  h1_strength,
            batch_size, optimizer, verbose=True):
        model = deepcopy(self.model)
        opt_state = optimizer.init(eqx.filter(model,
                                              self.get_gradient_mask()))
        for i in range(num_epochs):
            for (x, y) in utils.DataLoader((X, Y), batch_size=batch_size,
                                           seed=20):
                loss, model, opt_state = self.train(model, x, y,
                                                         loss_fun,
                                                         rkhs_strength,
                                                         h1_strength,
                                                         optimizer,
                                                         opt_state,
                                                         verbose=False)

            self.model = model
            if verbose:
                rkhs_norm = self.model.rkhs_norm() * rkhs_strength
                h1_norm = self.model.h1_seminorm_mixed_norm() * h1_strength
                mmd_loss = loss - rkhs_norm - h1_norm
                print(f'Epoch: {i},'
                      f'MMD loss: {mmd_loss}, '
                      f'RKHS penalty: {rkhs_norm}, '
                      f'H1 penalty: {h1_norm}')


    def get_gradient_mask(self):
        gradient_mask = jtu.tree_map(lambda _: False, self.model)
        gradient_mask = eqx.tree_at(lambda tree: [func.weights for func
                                                in tree.funcs], gradient_mask,
                                  replace=[True for _ in
                                           range(self.num_odes)])
        return gradient_mask

    def initialize_optimizer(self, optimizer):
        gradient_mask = self.get_gradient_mask()
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(eqx.filter(self.model,
                                                        gradient_mask))
        return 'Optimizer initialized!'

    @eqx.filter_jit
    def get_gradients(self, model, X, Y, loss_fun, rkhs_strength, \
                                    h1_strength):
        gradient_mask = self.get_gradient_mask()
        @ft.partial(eqx.filter_value_and_grad, arg=gradient_mask)
        def regularized_loss(model, X, Y, loss_fun, rkhs_strength, \
                                  h1_strength):
            pred = model(X)[-1]
            loss = loss_fun(pred, Y)

            # norms
            rkhs_norm = model.rkhs_norm()
            h1_norm = model.h1_seminorm_mixed_norm()

            # total loss
            total_loss = loss + rkhs_strength * rkhs_norm + h1_strength * h1_norm
            return total_loss

        # get loss and gradients
        loss, grads = regularized_loss(model, X, Y, loss_fun,
                                       rkhs_strength, h1_strength)
        return loss, grads

    def train(self, model, X, Y, loss_fun, rkhs_strength,
              h1_strength, optimizer, opt_state, verbose=True):
        loss, grads = self.get_gradients(model, X, Y, loss_fun,
                                         rkhs_strength, h1_strength)
        updates, updated_opt_state = optimizer.update(grads, opt_state)
        updated_model = eqx.apply_updates(model, updates)

        if verbose:
            rkhs_norm = model.rkhs_norm() * rkhs_strength
            h1_norm = model.h1_seminorm_mixed_norm() * h1_strength
            print(f'MMD loss: {loss - rkhs_norm - h1_norm}, RKHS penalty:'
                  f' {rkhs_norm}, '
                  f'H1 penalty: {h1_norm}')
        return loss, updated_model, updated_opt_state

    def transform(self, X, mode='forward', trajectory=False):
        X_trajectory = self.model(X, mode=mode)
        X_transformed = X_trajectory[-1]
        if trajectory:
            return X_transformed, X_trajectory
        else:
            return X_transformed



class Conditional_Transporter(BaseEstimator, TransformerMixin):
    def __init__(self, inducing_points, conditional_dim, kernel, num_odes,
                 key):
        self.inducing_points = inducing_points
        self.conditional_dim = conditional_dim
        self.num_odes = num_odes
        self.kernel = deepcopy(kernel)
        self.key = key
        self.model = Conditional_KernelODE(self.inducing_points,
                                                          self.kernel,
                                                          self.conditional_dim,
                                                          self.num_odes,
                                                          key=self.key)
        # Note that we are time-independent in num_odes == 1
        if self.num_odes == 1:
            print('Note: We are in the time-independent regime. For '
                  'time-dependent regime, num_odes > 1.')

    def fit(self, X, Y, C, num_epochs, loss_fun, rkhs_strength,
            h1_strength, batch_size, optimizer, verbose=True):
        model = deepcopy(self.model)
        opt_state = optimizer.init(eqx.filter(model,
                                              self.get_gradient_mask()))
        for i in range(num_epochs):
            for (x, y, c) in utils.DataLoader((X, Y, C), batch_size=batch_size,
                                              seed=20):
                loss, model, opt_state = self.train(model, x, y, c,
                                                    loss_fun,
                                                    rkhs_strength,
                                                    h1_strength,
                                                    optimizer,
                                                    opt_state,
                                                    verbose=False)

            self.model = model
            if verbose:
                rkhs_norm = self.model.rkhs_norm() * rkhs_strength
                h1_norm = self.model.h1_seminorm_mixed_norm() * h1_strength
                mmd_loss = loss - rkhs_norm - h1_norm
                print(f'Epoch: {i},'
                      f'MMD loss: {mmd_loss}, '
                      f'RKHS penalty: {rkhs_norm}, '
                      f'H1 penalty: {h1_norm}')


    def get_gradient_mask(self):
        gradient_mask = jtu.tree_map(lambda _: False, self.model)
        gradient_mask = eqx.tree_at(lambda tree: [func.weights for func
                                                in tree.funcs], gradient_mask,
                                  replace=[True for _ in
                                           range(self.num_odes)])
        return gradient_mask

    def initialize_optimizer(self, optimizer):
        gradient_mask = self.get_gradient_mask()
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(eqx.filter(self.model,
                                                        gradient_mask))
        return 'Optimizer initialized!'

    @eqx.filter_jit
    def get_gradients(self, model, X, Y, C, loss_fun, rkhs_strength, \
                                    h1_strength):
        gradient_mask = self.get_gradient_mask()
        @ft.partial(eqx.filter_value_and_grad, arg=gradient_mask)
        def regularized_loss(model, X, Y, C, loss_fun, rkhs_strength, \
                                  h1_strength):
            pred = model(X, C)[-1]

            # concatenate the predictions with the concatenated example
            pred_cond = jnp.column_stack((C, pred))
            loss = loss_fun(pred_cond, Y)

            # norms
            rkhs_norm = model.rkhs_norm()
            h1_norm = model.h1_seminorm_mixed_norm()

            # total loss
            total_loss = loss + rkhs_strength * rkhs_norm + h1_strength * h1_norm
            return total_loss

        # get loss and gradients
        loss, grads = regularized_loss(model, X, Y, C, loss_fun,
                                       rkhs_strength, h1_strength)
        return loss, grads

    def train(self, model, X, Y, C, loss_fun, rkhs_strength,
              h1_strength, optimizer, opt_state, verbose=True):
        loss, grads = self.get_gradients(model, X, Y, C, loss_fun,
                                         rkhs_strength, h1_strength)
        updates, updated_opt_state = optimizer.update(grads, opt_state)
        updated_model = eqx.apply_updates(model, updates)

        if verbose:
            rkhs_norm = model.rkhs_norm() * rkhs_strength
            h1_norm = model.h1_seminorm_mixed_norm() * h1_strength
            print(f'MMD loss: {loss - rkhs_norm - h1_norm}, RKHS penalty:'
                  f' {rkhs_norm}, '
                  f'H1 penalty: {h1_norm}')
        return loss, updated_model, updated_opt_state

    def transform(self, X, C, mode='forward', trajectory=False):
        X_trajectory = self.model(X, C, mode=mode)
        X_transformed = X_trajectory[-1]
        if trajectory:
            return X_transformed, X_trajectory
        else:
            return X_transformed








