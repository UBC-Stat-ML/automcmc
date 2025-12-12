from tests import utils as testutils

import unittest

import jax
from jax import random
from jax import numpy as jnp

from numpyro.infer import MCMC

from automcmc import autohmc
from automcmc import autorwmh
from automcmc import slicer
from automcmc import utils
from automcmc import tempering

class TestTempering(unittest.TestCase):

    TESTED_KERNELS = (
        autorwmh.AutoRWMH,
        autohmc.AutoMALA,
        autohmc.AutoHMC,
        slicer.HitAndRunSliceSampler,
        slicer.DeterministicScanSliceSampler
    )
    
    def test_no_nan_at_zero(self):
        p = tempering.tempered_potential_from_logprior_and_loglik(
            jnp.float32(0.2), jnp.float32(jnp.inf), jnp.float32(0)
        )
        self.assertFalse(jnp.isnan(p))

    def test_autohmc_is_inv_temp_aware(self):
        # warmup
        rng_key = random.key(9)
        n_rounds = 10
        n_warmup, n_keep = utils.split_n_rounds(n_rounds) # translate rounds to warmup/keep
        model, model_args, model_kwargs = testutils.make_eight_schools()
        kernel = autohmc.AutoHMC(model)
        mcmc = MCMC(
            kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False
        )
        mcmc.run(rng_key, *model_args, **model_kwargs)
        
        # check the default is using no inv temp
        init_state = mcmc.last_state
        self.assertIsNone(init_state.inv_temp)

        # check that the result of the (deterministic) involution changes with
        # the inverse temperature of the target
        def vmap_fn(inv_temp):
            state = init_state._replace(inv_temp = inv_temp)
            state = kernel.involution_main(
                state.base_step_size, 
                state, 
                state.base_precond_state
            )
            return state.p_flat[0]

        signatures = jax.vmap(vmap_fn)(jnp.linspace(0,1,10))
        self.assertGreater(signatures.std(), jnp.finfo(signatures.dtype).eps)

    def test_tempered_moments(self):
        n_rounds = 14
        n_warmup, n_keep = utils.split_n_rounds(n_rounds) # translate rounds to warmup/keep
        model, model_args, model_kwargs = testutils.toy_conjugate_normal()
        rng_key = random.key(321453)
        for inv_temp in jnp.array([0., 0.25, 0.75, 1.0]):
            true_var = jnp.reciprocal(inv_temp + jnp.reciprocal(jnp.square(model_args[0])))
            true_sd = jnp.sqrt(true_var)
            true_mean = inv_temp * model_args[1][0] * true_var
            for kernel_type in self.TESTED_KERNELS:
                with self.subTest(inv_temp=inv_temp, kernel_type=kernel_type):
                    rng_key, mcmc_key = random.split(rng_key) 
                    kernel = kernel_type(model, init_inv_temp=inv_temp)
                    mcmc = MCMC(
                        kernel, 
                        num_warmup=n_warmup, 
                        num_samples=n_keep, 
                        progress_bar=False
                    )
                    mcmc.run(mcmc_key, *model_args, **model_kwargs)
                    adapt_stats=mcmc.last_state.stats.adapt_stats
                    self.assertTrue(
                        jnp.allclose(adapt_stats.sample_mean, true_mean, atol=0.3, rtol=0.1),
                        msg=f"sample_mean={adapt_stats.sample_mean} but true_mean={true_mean}"
                    )
                    sample_sd = jnp.sqrt(adapt_stats.sample_var)
                    self.assertTrue(
                        jnp.allclose(sample_sd, true_sd, atol=0.3, rtol=0.15),
                        msg=f"sample_sd={sample_sd} but true_sd={true_sd}"
                    )

if __name__ == '__main__':
    unittest.main()
