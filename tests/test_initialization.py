from tests import utils as testutils

import unittest

from jax import random
from jax import numpy as jnp

from automcmc import initialization
from automcmc import tempering

class TestInitialization(unittest.TestCase):

    def test_MAP(self):
        rng_key = random.key(90)
        model, model_args, model_kwargs = testutils.make_eight_schools()
        MAP_estimate_unconstrained = initialization.MAP(
            model,
            rng_key,
            model_args,
            model_kwargs,
            optimize_fun_settings={"L-BFGS": {"n_iter": 64, "solver_params": {}}}
        )

        # transform to constrained space
        exec_trace = tempering.trace_from_unconst_samples(
            model, 
            model_args, 
            model_kwargs,
            MAP_estimate_unconstrained, 
        )
        MAP_estimate = {
            name: site["value"] for name, site in exec_trace.items() 
            if site["type"] == "sample" and not site["is_observed"]
        }
        self.assertTrue(jnp.allclose(MAP_estimate['mu'], MAP_estimate['theta']))
        self.assertLess(MAP_estimate['tau'], 1e-5)

if __name__ == '__main__':
    unittest.main()
