from tests import utils as testutils

import unittest

from jax import random
import jax.numpy as jnp

from numpyro.infer import MCMC

from automcmc.slicer import HitAndRunSliceSampler
from automcmc.autohmc import AutoMALA
from automcmc import utils, preconditioning, selectors

class TestPreconditioning(unittest.TestCase):

    def test_preconditioning(self):
        dim = 3
        rho = 0.9
        tol = 0.2
        n_rounds = 14
        rng_key = random.key(2349895454)
        init_vals = jnp.ones(dim)
        S = testutils.make_const_off_diag_corr_mat(dim, rho)
        diag_S = jnp.diag(S)
        pot_fn = testutils.make_correlated_Gaussian_potential(S)
        n_warmup, n_keep = utils.split_n_rounds(n_rounds) # translate rounds to warmup/keep
        precs = (
            preconditioning.IdentityDiagonalPreconditioner(),
            preconditioning.FixedDiagonalPreconditioner(),
            preconditioning.FixedDensePreconditioner(),
        )
        sels = (
            selectors.DeterministicSymmetricSelector(),
            selectors.MaxEJDSelector()
        )
        for kernel_class in (AutoMALA, HitAndRunSliceSampler):
            for sel in sels:
                if kernel_class is HitAndRunSliceSampler and sel != sels[0]:
                    # Slicer doesnt actually use the selector
                    continue
                all_min_ess = []
                for p in precs:
                    subtest_config={
                        'kernel_class': kernel_class, 
                        'sel_class': type(sel),
                        'prec_class': type(p)
                    }
                    with self.subTest(**subtest_config):
                        kernel = kernel_class(
                            potential_fn = pot_fn, preconditioner = p
                        )
                        mcmc = MCMC(
                            kernel, 
                            num_warmup=n_warmup, 
                            num_samples=n_keep, 
                            progress_bar=False
                        )
                        rng_key, mcmc_key = random.split(rng_key)
                        mcmc.run(mcmc_key, init_params=init_vals)
                        last_base_precond_state = mcmc.last_state.base_precond_state
                        min_ess = testutils.extremal_diagnostics(mcmc)[1]
                        all_min_ess.append(min_ess)
                        print(f"{subtest_config}: min_ess={min_ess}")
                        self.assertGreater(min_ess, 100)

                        # check factors
                        L = last_base_precond_state.var_tril_factor
                        U = last_base_precond_state.inv_var_triu_factor
                        last_round_used_var = last_base_precond_state.var
                        if jnp.ndim(L) == 2:
                            v = L@L.T
                            I1, I2 = (U.T@L, jnp.identity(L.shape[-1]))
                        else:
                            v = L*L
                            I1, I2 = (U*L, jnp.ones_like(L))
                        self.assertTrue(jnp.allclose(v, last_round_used_var, rtol=tol))
                        self.assertTrue(jnp.allclose(I1, I2, atol=tol)) # use atol for off-diag zeros

                        # check preconditioner state variance and online estimator against true var
                        if preconditioning.is_dense(p):
                            last_round_estimate_var = mcmc.last_state.stats.adapt_stats.sample_var
                            self.assertTrue(jnp.allclose(S, last_round_used_var, rtol=tol))
                            self.assertTrue(jnp.allclose(S, last_round_estimate_var, rtol=tol))
                        else:
                            last_round_estimate_var = mcmc.last_state.stats.adapt_stats.sample_var
                            self.assertTrue(
                                jnp.allclose(last_round_estimate_var, diag_S, rtol=tol),
                                f"last_round_estimate_var={last_round_estimate_var}"
                            )
                            if not isinstance(p, preconditioning.IdentityDiagonalPreconditioner):
                                self.assertTrue(
                                    jnp.allclose(last_round_used_var, diag_S, rtol=tol),
                                    f"last_round_used_var={last_round_used_var}"
                                )

                # check that Dense gives highest ess
                self.assertEqual(all_min_ess[-1], max(all_min_ess))

if __name__ == '__main__':
    unittest.main()
