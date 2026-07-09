from tests import utils as testutils

import unittest
from jax import random
from jax import numpy as jnp

from numpyro.infer import MCMC

from automcmc import autohmc
from automcmc import selectors
from automcmc import utils

class TestSelectors(unittest.TestCase):

    def test_selectors(self):
        asym_sel = selectors.AsymmetricSelector()
        sym_sel = selectors.SymmetricSelector()
        det_asym_sel = selectors.DeterministicAsymmetricSelector()
        det_sym_sel = selectors.DeterministicSymmetricSelector()
        params = jnp.array([-1.5, -0.5])

        # sym and asym should give different decisions here
        log_diff = jnp.float32(0.8)
        self.assertTrue(asym_sel.should_grow(params, log_diff))
        self.assertTrue(det_asym_sel.should_grow(params, log_diff))
        self.assertFalse(sym_sel.should_grow(params, log_diff))
        self.assertFalse(det_sym_sel.should_grow(params, log_diff))
        self.assertFalse(asym_sel.should_shrink(params, log_diff))
        self.assertFalse(det_asym_sel.should_shrink(params, log_diff))
        self.assertFalse(sym_sel.should_shrink(params, log_diff))
        self.assertFalse(det_sym_sel.should_shrink(params, log_diff))

        # sym and asym should agree in the following 2 cases
        log_diff = jnp.float32(-2)
        self.assertFalse(asym_sel.should_grow(params, log_diff))
        self.assertFalse(det_asym_sel.should_grow(params, log_diff))
        self.assertFalse(sym_sel.should_grow(params, log_diff))
        self.assertFalse(det_sym_sel.should_grow(params, log_diff))
        self.assertTrue(asym_sel.should_shrink(params, log_diff))
        self.assertTrue(det_asym_sel.should_shrink(params, log_diff))
        self.assertTrue(sym_sel.should_shrink(params, log_diff))
        self.assertTrue(det_sym_sel.should_shrink(params, log_diff))

        log_diff = jnp.float32(0)
        self.assertTrue(asym_sel.should_grow(params, log_diff))
        self.assertTrue(det_asym_sel.should_grow(params, log_diff))
        self.assertTrue(sym_sel.should_grow(params, log_diff))
        self.assertTrue(det_sym_sel.should_grow(params, log_diff))
        self.assertFalse(asym_sel.should_shrink(params, log_diff))
        self.assertFalse(det_asym_sel.should_shrink(params, log_diff))
        self.assertFalse(sym_sel.should_shrink(params, log_diff))
        self.assertFalse(det_sym_sel.should_shrink(params, log_diff))

    def test_max_n_iter_respected(self):
        kernel_class = autohmc.AutoMALA
        rng_key = random.key(90)
        for max_n_iter in range(1,4):
            rng_key, run_key = random.split(rng_key)
            kernel = kernel_class(
                testutils.toy_unid,
                selector=selectors.DeterministicSymmetricSelector(max_n_iter=max_n_iter),
                init_base_step_size = 100.0 # big number
            )

            # just need this to init the state
            mcmc = MCMC(kernel, num_warmup=0, num_samples=1, progress_bar=False)
            mcmc.run(run_key, 100, n_heads=50)
            kernel = mcmc.sampler
            state = mcmc.last_state
            precond_state = state.base_precond_state

            # check n iters are bounded
            state = kernel.update_log_joint(state, precond_state) # should be ok but just in case
            exponent = kernel.auto_step_size(
                state, (-2.0, -1.0), precond_state # any log bounds should work
            )[-1]
            self.assertLessEqual(jnp.abs(exponent), max_n_iter)

    # fixed selector preserves the initial base step size
    def test_fixed_step_size_preserved(self):
        init_val = jnp.array([1., 2.])
        step_key, run_key = random.split(random.key(54))
        init_base_step_size = random.exponential(step_key)
        kernel = autohmc.AutoMALA(
            potential_fn=testutils.gaussian_potential,
            init_base_step_size = init_base_step_size,
            selector=selectors.FixedStepSizeSelector()
        )
        n_warmup, n_keep = utils.split_n_rounds(10)
        mcmc = MCMC(
            kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False
        )
        mcmc.run(run_key, init_params=init_val)
        self.assertTrue(mcmc.last_state.base_step_size == init_base_step_size)

if __name__ == '__main__':
    unittest.main()