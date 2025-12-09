from tests import utils as testutils

from functools import partial
import unittest

from scipy import stats

import jax
from jax import lax
from jax import random
from jax import numpy as jnp

from numpyro.infer import MCMC

from automcmc import autohmc,autorwmh,slicer
from automcmc import preconditioning
from automcmc import selectors
from automcmc import statistics
from automcmc import utils


# sequentially sample multiple times, discard intermediate states
def loop_sample(kernel, n_refresh, kernel_state):
    return lax.scan(
        lambda state, _: (kernel.sample(state,(),{}), None),
        kernel_state,
        length=n_refresh
    )[0]

# tune a sampler starting from an iid sample from the target
def init_and_tune_kernel(
        kernel, 
        dim, 
        true_mean, 
        true_sd, 
        n_warmup,
        rng_key
    ):
    run_key, init_key = random.split(rng_key)

    # draw an iid sample from the target
    init_val = true_mean + true_sd*random.normal(init_key, dim)

    # initialize a kernel state with warmup to tune parameters
    # note: num_samples = 0 fails.
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=1, progress_bar=False)
    mcmc.run(run_key, init_params=init_val)
    kernel_state = mcmc.last_state

    # reset the initial state to the sample from the target 
    kernel_state = kernel_state._replace(x=init_val)
    return kernel, kernel_state

    
class TestKernels(unittest.TestCase):

    TESTED_KERNELS = (
        autorwmh.AutoRWMH,
        autohmc.AutoMALA,
        autohmc.AutoHMC,
        slicer.HitAndRunSliceSampler,
    )

    TESTED_PRECONDITIONERS = (
        preconditioning.IdentityDiagonalPreconditioner(),
        preconditioning.FixedDiagonalPreconditioner(),
        preconditioning.FixedDensePreconditioner()
    )

    TESTED_SELECTORS = (
        selectors.AsymmetricSelector, 
        selectors.SymmetricSelector,
        selectors.DeterministicSymmetricSelector,
        selectors.MaxEJDSelector,
    )
    
    def test_involution(self):
        tol = 1e-5
        init_val = jnp.array([1., 2.])
        n_warmup = utils.split_n_rounds(10)[0]
        rng_key = random.key(321)
        for kernel_class in self.TESTED_KERNELS:
            if kernel_class is slicer.HitAndRunSliceSampler: # not involutive
                continue
            for prec in self.TESTED_PRECONDITIONERS:
                for sel in self.TESTED_SELECTORS:
                    with self.subTest(kernel_class=kernel_class, prec_type=type(prec), sel_type=sel):
                        print(f"kernel_class={kernel_class}, prec_type={type(prec)}, sel_type={sel}")
                        rng_key, run_key = random.split(rng_key)
                        kernel = kernel_class(
                            potential_fn=testutils.gaussian_potential,
                            selector=sel(),
                            preconditioner = prec
                        )
                        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=1, progress_bar=False)
                        mcmc.run(run_key, init_params=init_val)
                        s = mcmc.last_state
                        precond_state = s.base_precond_state
                        step_size = s.base_step_size
                        s_half = kernel.involution_main(step_size, s, precond_state)
                        s_one = kernel.involution_aux(s_half)
                        s_onehalf = kernel.involution_main(step_size, s_one, precond_state)
                        s_two = kernel.involution_aux(s_onehalf)
                        self.assertTrue(
                            jnp.allclose(s_two.x, s.x, atol=tol, rtol=tol),
                            msg=f"s.x={s.x} but s_two.x={s_two.x}"
                        )
                        self.assertTrue(
                            jnp.allclose(s_two.p_flat, s.p_flat, atol=tol, rtol=tol),
                            msg=f"s.p_flat={s.p_flat} but s_two.p_flat={s_two.p_flat}"
                        )

    # invariance test on an isotropic multivariate normal
    def test_invariance(self):
        dim = 4
        true_mean = 2.
        true_var = 0.5
        true_sd = jnp.sqrt(true_var)
        true_cdf = partial(stats.norm.cdf, loc=true_mean, scale=true_sd)
        n_warmup = utils.split_n_rounds(10)[0]
        n_refresh = 1024
        n_samples = 1024
        pval_threshold = 0.001
        rng_key = random.key(2)
        max_n_iters = (1, 2**20)
        
        for kernel_class in self.TESTED_KERNELS:
            for prec in self.TESTED_PRECONDITIONERS:
                for sel in self.TESTED_SELECTORS:
                    if kernel_class is slicer.HitAndRunSliceSampler and (sel is not selectors.DeterministicSymmetricSelector):
                        continue
                    for max_n_iter in max_n_iters:
                        with self.subTest(
                            kernel_class=kernel_class, prec_type=type(prec), sel_type=sel, max_n_iter=max_n_iter
                        ):
                            print(f"kernel_class={kernel_class}, prec_type={type(prec)}, sel_type={sel}, max_n_iter={max_n_iter}")
                            rng_key, exp_key = random.split(rng_key)
                            def run_fn(init_key):
                                kernel = kernel_class(
                                    potential_fn=testutils.gaussian_potential, 
                                    selector=sel(max_n_iter=max_n_iter),
                                    preconditioner = prec
                                )
                                kernel, kernel_state = init_and_tune_kernel(
                                    kernel, 
                                    dim, 
                                    true_mean, 
                                    true_sd, 
                                    n_warmup,
                                    init_key
                                )
                                return loop_sample(kernel, n_refresh, kernel_state).x[0]
                            vmap_fn = jax.vmap(run_fn)
                            run_keys = random.split(exp_key, n_samples)
                            mcmc_samples = vmap_fn(run_keys)
                            ks_res = stats.ks_1samp(mcmc_samples, true_cdf)
                            self.assertGreater(ks_res.pvalue, pval_threshold)

    # test online stats
    # note: currently only tests kernels using the diagonal precond, but the
    # `test_preconditioning` file runs tests using both
    def test_online_stats(self):
        init_val = jnp.array([1., 2.])
        rng_key = random.key(23)
        true_mean = 2.
        true_var = 0.5
        true_sd = jnp.sqrt(true_var)
        n_rounds = 14
        n_warmup, n_keep = utils.split_n_rounds(n_rounds)
        tol = 0.2
        for kernel_class in self.TESTED_KERNELS:
            for sel in self.TESTED_SELECTORS:
                if kernel_class is slicer.HitAndRunSliceSampler and (sel is not selectors.DeterministicSymmetricSelector):
                        continue
                with self.subTest(kernel_class=kernel_class, sel_type=sel):
                    print(f"kernel_class={kernel_class}, sel_type={sel}")
                    rng_key, run_key = random.split(rng_key)
                    kernel = kernel_class(
                        potential_fn=testutils.gaussian_potential, 
                        selector=sel()
                    )
                    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False)
                    mcmc.run(run_key, init_params=init_val)
                    state = mcmc.last_state
                    stats = state.stats
                    adapt_stats = stats.adapt_stats
                    self.assertEqual(stats.n_samples, n_warmup+n_keep)
                    self.assertEqual(adapt_stats.sample_idx, n_keep)
                    self.assertEqual(n_keep, jnp.shape(mcmc.get_samples())[0])
                    var_tril_factor = state.base_precond_state.var_tril_factor
                    self.assertTrue(
                        jnp.allclose(var_tril_factor, true_sd, atol=tol, rtol=tol),
                        msg=f"var_tril_factor={var_tril_factor} but true_sd={true_sd}"
                    )
                    self.assertTrue(
                        jnp.allclose(adapt_stats.sample_mean, true_mean, rtol=tol),
                        msg=f"sample_mean={adapt_stats.sample_mean} but true_mean={true_mean}"
                    )
                    sample_sd = jnp.sqrt(adapt_stats.sample_var)
                    self.assertTrue(
                        jnp.allclose(sample_sd, true_sd, rtol=tol, atol=tol),
                        msg=f"sample_sd={sample_sd} but true_sd={true_sd}"
                    )
    

    def test_numpyro_model(self):
        rng_key = random.key(9)
        tol = 0.1
        n_rounds = 14
        n_warmup, n_keep = utils.split_n_rounds(n_rounds)
        for kernel_class in self.TESTED_KERNELS:
            for prec in self.TESTED_PRECONDITIONERS:
                if isinstance(prec, preconditioning.IdentityDiagonalPreconditioner):
                    # doesnt work
                    continue
                for sel in self.TESTED_SELECTORS:
                    if kernel_class is slicer.HitAndRunSliceSampler and (sel is not selectors.DeterministicSymmetricSelector):
                        continue
                    if sel is selectors.AsymmetricSelector:
                        # doesnt work with automala here
                        continue
                    with self.subTest(kernel_class=kernel_class, prec_type=type(prec), sel_type=sel):
                        print(f"kernel_class={kernel_class}, prec_type={type(prec)}, sel_type={sel}")
                        rng_key, run_key = random.split(rng_key)
                        kernel = kernel_class(
                            testutils.toy_unid, 
                            selector=sel(),
                            preconditioner = prec
                        )
                        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep, progress_bar=False)
                        mcmc.run(run_key, 100, n_heads=50)
                        samples = mcmc.get_samples()
                        self.assertLess(testutils.extremal_diagnostics(mcmc)[0], 1+tol)
                        self.assertAlmostEqual(samples["p1"].mean(), 0.71, delta=tol)
                        self.assertAlmostEqual(samples["p2"].mean(), 0.71, delta=tol)
                        mean_p_prod = (samples["p1"] * samples["p2"]).mean()
                        self.assertAlmostEqual(mean_p_prod, 0.5, delta=tol)
                

if __name__ == '__main__':
    unittest.main()
