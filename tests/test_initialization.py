from tests import utils as testutils

import unittest

from jax import random
from jax import numpy as jnp

from numpyro.infer import MCMC
from automcmc import utils,autorwmh,initialization,tempering

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

    def test_optim_init_params(self):
        run_key = random.key(5)
        kernel = autorwmh.AutoRWMH(
            potential_fn=testutils.gaussian_potential,
            optimize_init_params = True
        )
        n_warmup, n_keep = utils.split_n_rounds(10)
        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep)
        mcmc.run(run_key, init_params=jnp.full(100,100,dtype=jnp.float32))
        self.assertAlmostEqual(mcmc.get_samples().mean(), 2, delta=0.1)
        
if __name__ == '__main__':
    unittest.main()
