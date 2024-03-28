import os
import sys
import unittest
import jax.numpy as jnp
import torch
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt

REQUIRED_PYTHON = "python3"

class TestEnvironment(unittest.TestCase):

    def test_python_version(self):
        """Test if the current Python version matches the required version."""
        required_major = 3 if REQUIRED_PYTHON == "python3" else 2
        system_major = sys.version_info.major
        self.assertEqual(system_major, required_major,
                         f"This project requires Python {required_major}. "
                         f"Found: Pythin {system_major}")
    def test_jax_array_creation_and_sum(self):
        """Test that JAX can create an array and compute its sum."""
        arr = jnp.array([1, 2, 3])
        arr_sum = jnp.sum(arr)
        self.assertEqual(arr_sum, 6, "JAX array sum did not match expected "
                                     "value")

    def test_torch_tensor_creation_and_multiplication(self):
        """Test that PyTorch can create a tensor and compute its
        multiplication."""
        tensor = torch.tensor([1, 2, 3])
        tensor_mul = tensor * 2
        # Convert tensor to list for comparison
        expected_result = [2, 4, 6]
        self.assertListEqual(tensor_mul.tolist(), expected_result, "Pytorch "
                                                                   "tensor "
                                                                   "multiplication did not match expected values.")

    def test_diffrax_ode_solver(self):
        """Test solving a simple ODE using Diffrax"""

        def f(t, y, args):
            # A simple ODE: dy/dt = -y
            return -y

        # Initial Conditions
        t0 = 0.0
        t1 = 1.0
        y0 = jnp.array([1.0])

        # Solve the ODE
        try:
            term = ODETerm(f)
            solver = Dopri5()
            saveat = SaveAt(ts=[1.0])
            solution = diffeqsolve(term, solver, t0=t0, t1=1, dt0=0.1, y0=y0,
                                   saveat=saveat)
            # Evaluate the solution at the final time
            y_final = solution.ys[0]
            # Check if the solution at t=1 is close to the expected value
            self.assertTrue(jnp.isclose(y_final, jnp.exp(-1.0), rtol=1e-3),
                        "Diffrax ODE solve did not match expected value. ")
        except ValueError as e:
            # If running on an unsupported platform, ensure the error is
            # raised due to Jax support
            if "METAL backend" in str(e):
                raise AssertionError("JAX GPU support is experimental on "
                                     "Apple M1 Chips." + str(e)) from e
            else:
                raise


if __name__ == '__main__':
    unittest.main()