import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import jax.random as random
from typing import List


class TimeIndependentFunc(eqx.Module):
    """ Time Independent RHS of the Kernel ODE."""
    anchor_points: jnp.ndarray
    kernel: eqx.Module
    weights: jnp.ndarray

    def __init__(self, anchor_points, kernel, *, key, **kwargs):
        super().__init__(**kwargs)
        self.weights = random.normal(key, shape=anchor_points.shape)
        self.kernel = kernel
        self.anchor_points = anchor_points

    def __call__(self, ts, x, args=None):
        gram_matrix = self.kernel(x, self.anchor_points)
        output = jnp.dot(gram_matrix, self.weights)
        return output

    def l2_norm(self):
        norm = jnp.sum(self.weights ** 2)
        return norm

    def rkhs_norm(self):
        """ Computes the RKHS norm of the function. |f|_H = sum_{i=1}^d
        c_i^T K(a, a^T) c_i"""

        cov = self.kernel(self.anchor_points, self.anchor_points)
        # rkhs_norm = sum([self.weights[:, i] @ cov @ self.weights[:, i] for i
        #                  in range(self.weights.shape[1])])
        rkhs_norm = (jnp.matmul(self.weights.T, cov) * self.weights.T).sum()
        return rkhs_norm


class KernelODE(eqx.Module):
    """ Time dependent Kernel Class with various H1 penalties"""
    funcs: List[TimeIndependentFunc]
    t0: int
    t1: int
    num_odes: int
    dt0: float
    anchor_points: jnp.ndarray
    kernel: eqx.Module

    def __init__(self, anchor_points, kernel, num_odes, *, key, **kwargs):
        super().__init__(**kwargs)
        self.t0 = 0.
        self.t1 = 1.
        self.num_odes = num_odes
        self.dt0 = self.t1 / self.num_odes
        self.anchor_points = anchor_points
        self.kernel = kernel

        keys = random.split(key, self.num_odes)
        self.funcs = [
            TimeIndependentFunc(self.anchor_points, self.kernel, key=k) for k
            in keys]

    def __call__(self, y0, num_steps=10, mode='forward'):
        solver = dfx.Heun()
        out = jnp.zeros(((num_steps * self.num_odes) + 1, len(y0),
                         y0.shape[1]))
        out = out.at[0].set(y0)

        if mode == 'forward':
            t_steps = jnp.linspace(self.t0, self.t1, self.num_odes + 1)
            func_list = self.funcs
        elif mode == 'backward':
            t_steps = jnp.linspace(self.t1, self.t0, self.num_odes + 1)
            func_list = self.funcs[::-1]

        for i in range(1, self.num_odes + 1):
            t_start = t_steps[i - 1]
            t_end = t_steps[i]
            dt0 = (t_end - t_start) / num_steps
            save_at = jnp.linspace(t_start, t_end, num_steps + 1)[1:]

            func = func_list[i - 1]
            term = dfx.ODETerm(func)
            sol = dfx.diffeqsolve(term, solver, t0=t_start, t1=t_end,
                                  dt0=dt0, y0=y0,
                                  saveat=dfx.SaveAt(ts=save_at))
            ys = sol.ys
            out = out.at[num_steps * (i-1) + 1: num_steps * (i) + 1].set(ys)
            y0 = ys[-1]
        return out.squeeze()

    def rkhs_norm(self):
        norm_over_time = jnp.array([func.rkhs_norm() for func in self.funcs])
        norm = norm_over_time * self.dt0
        return jnp.sum(norm)

    def l2_norm_of_weights(self):
        norm = jnp.sum(jnp.array([func.l2_norm() for func in self.funcs]))
        norm *= self.dt0
        return norm


    def h1_seminorm_mixed_norm(self):
        assert self.num_odes > 0, 'The num time steps should be positive'
        if self.num_odes == 1:
            return 0
        else:
            weights = jnp.array([func.weights for func in self.funcs])
            dweights_dt = jnp.gradient(weights, self.dt0, axis=0)
            K_xx = self.kernel(self.anchor_points, self.anchor_points)
            norm_over_time = jnp.array(
                [(jnp.sum(jnp.matmul(dweights_dt[i].T, K_xx) *
                              dweights_dt[i].T)) for i in range(len(weights))])
            norm = jnp.sum(norm_over_time) * self.dt0
            return norm



class Conditional_TimeIndependentFunc(eqx.Module):
    """ Time Independent RHS of the Kernel ODE."""
    anchor_points: jnp.ndarray
    kernel: eqx.Module
    weights: jnp.ndarray
    conditioning_dim: int

    def __init__(self, anchor_points, conditioning_dim, kernel, *, key,
    **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.anchor_points = anchor_points
        self.weights = random.normal(key, shape=(len(anchor_points),
                                                 conditioning_dim))
        self.conditioning_dim = conditioning_dim

    def __call__(self, ts, x, conditioning_data):
        concat = jnp.column_stack((x, conditioning_data))
        gram_matrix = self.kernel(concat, self.anchor_points)
        output = jnp.dot(gram_matrix, self.weights)
        return output

    def l2_norm(self):
        norm = jnp.sum(self.weights ** 2)
        return norm

    def rkhs_norm(self):
        """ Computes the RKHS norm of the function.
        |f|_H = \sum_{i=1}^d c_i^T K(a, a^T) c_i"""

        cov = self.kernel(self.anchor_points, self.anchor_points)
        rkhs_norm = (jnp.matmul(self.weights.T, cov) * self.weights.T).sum()
        return rkhs_norm


class Conditional_KernelODE(eqx.Module):
    """ Time dependent Kernel Class with various H1 penalties"""
    funcs: List[Conditional_TimeIndependentFunc]
    t0: int
    t1: int
    num_odes: int
    conditioning_dim: int
    dt0: float
    anchor_points: jnp.ndarray
    kernel: eqx.Module

    def __init__(self, anchor_points, kernel, conditioning_dim, num_odes,
                 *, key, **kwargs):
        super().__init__(**kwargs)
        self.t0 = 0.
        self.t1 = 1.
        self.num_odes = num_odes
        self.dt0 = self.t1 / self.num_odes
        self.anchor_points = anchor_points
        self.conditioning_dim = conditioning_dim
        self.kernel = kernel

        keys = random.split(key, self.num_odes)
        self.funcs = [
            Conditional_TimeIndependentFunc(self.anchor_points,
                                             self.conditioning_dim,
                                             self.kernel, key=k) for k in keys]

    def __call__(self, y0, conditioning_data, num_steps=10, mode='forward'):
        solver = dfx.Heun()
        out = jnp.zeros(((num_steps * self.num_odes) + 1, len(y0),
                         y0.shape[1]))
        out = out.at[0].set(y0)

        if mode == 'forward':
            t_steps = jnp.linspace(self.t0, self.t1, self.num_odes + 1)
            func_list = self.funcs
        elif mode == 'backward':
            t_steps = jnp.linspace(self.t1, self.t0, self.num_odes + 1)
            func_list = self.funcs[::-1]

        for i in range(1, self.num_odes + 1):
            t_start = t_steps[i - 1]
            t_end = t_steps[i]
            dt0 = (t_end - t_start) / num_steps
            save_at = jnp.linspace(t_start, t_end, num_steps + 1)[1:]

            func = func_list[i - 1]
            term = dfx.ODETerm(func)
            sol = dfx.diffeqsolve(term, solver, t0=t_start, t1=t_end,
                                  dt0=dt0, y0=y0, args=conditioning_data,
                                  saveat=dfx.SaveAt(ts=save_at))
            ys = sol.ys
            out = out.at[num_steps * (i-1) + 1: num_steps * (i) + 1].set(ys)
            y0 = ys[-1]
        return out.squeeze()

    def rkhs_norm(self):
        norm_over_time = jnp.array([func.rkhs_norm() for func in self.funcs])
        norm = norm_over_time * self.dt0
        return jnp.sum(norm)

    def l2_norm_of_weights(self):
        norm = jnp.sum(jnp.array([func.l2_norm() for func in self.funcs]))
        norm *= self.dt0
        return norm


    def h1_seminorm_mixed_norm(self):
        assert self.num_odes > 0, 'The num time steps should be positive'
        if self.num_odes == 1:
            return 0
        else:
            weights = jnp.array([func.weights for func in self.funcs])
            dweights_dt = jnp.gradient(weights, self.dt0, axis=0)
            K_xx = self.kernel(self.anchor_points, self.anchor_points)
            norm_over_time = jnp.array(
                [(jnp.sum(jnp.matmul(dweights_dt[i].T, K_xx) *
                              dweights_dt[i].T)) for i in range(len(weights))])
            norm = jnp.sum(norm_over_time) * self.dt0
            return norm