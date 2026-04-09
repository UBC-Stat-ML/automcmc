import unittest

from jax import numpy as jnp
from automcmc import utils

class TestUtils(unittest.TestCase):

    def test_round_arithmetic(self):
        for n_rounds in jnp.arange(1,30):
            # print(f"n_rounds={n_rounds}")
            n_warmup, n_keep = utils.split_n_rounds(n_rounds)
            n_samples = n_warmup + n_keep
            self.assertEqual(n_rounds, utils.n_warmup_to_adapt_rounds(n_warmup) + 1)
            self.assertEqual(n_samples, 2**(n_rounds+1)-2)
            self.assertEqual(n_rounds, utils.current_round(n_samples))

    def test_newton(self):
        # example from 
        # https://github.com/jax-ml/jax/discussions/17975#discussion-5707669
        def f(x):
            x, y, z = x
            f1 = x**2 + y**2 + z**2 - 3
            f2 = x**2 + y**2 - z    - 1
            f3 = x    + y    + z    - 3
            return jnp.array([f1, f2, f3])

        x_true = jnp.ones(3)

        for mode in ("direct", "gmres"):
            # nice initial point
            x0 = jnp.array([0.2, 0.3, 0.5])
            x, n, val, err, d_err, is_satisfied = utils.newton(f, x0, mode=mode)
            self.assertTrue(is_satisfied)
            self.assertLess(err, 1e-3)
            self.assertLess(d_err, 0)
            self.assertTrue(jnp.allclose(x, x_true, rtol=0.05))

            # unstable initial point => divergence
            x0 = jnp.array([ 0.36057416,  1.2849895 , -0.73873436])
            x, n, val, err, d_err, is_satisfied = utils.newton(f, x0, mode=mode)
            self.assertFalse(is_satisfied)
            self.assertEqual(n, 4) # divergence caught
            self.assertGreater(d_err, 0)

            # initial point with rank-deficient Jacobian => inf/nans at n=1
            x0 = jnp.zeros_like(x_true)
            x, n, val, err, d_err, is_satisfied = utils.newton(f, x0, mode=mode)
            self.assertFalse(is_satisfied)
            self.assertEqual(n, 1) # nans caught
            self.assertTrue(jnp.isnan(d_err)) # nans caught


if __name__ == '__main__':
    unittest.main()