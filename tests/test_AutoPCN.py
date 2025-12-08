from tests import utils as testutils
import unittest
import jax
from jax import random
from jax import numpy as jnp

from automcmc import autopcn, preconditioning

class TestAutoPCN(unittest.TestCase):

    def test_autopcn(self):
        # make a correlated Gaussian target
        S = testutils.make_const_off_diag_corr_mat(3, 0.99)
        L = jax.lax.linalg.cholesky(S)
        U = jax.lax.linalg.triangular_solve(
            L, 
            jnp.identity(S.shape[-1]), 
            transpose_a=True, 
            lower=True
        )
        def corr_normal(x):
            x_std = jnp.dot(x, U) # == U.T @ x
            return 0.5*jnp.dot(x_std,x_std)
        
        def std_normal(x):
            return 0.5*jnp.dot(x,x)

        rng_key = random.key(1)
        for p in (std_normal, corr_normal):
            rng_key, refresh_key, init_key, mcmc_key = random.split(rng_key,4)
            init_params = random.normal(init_key, S.shape[0])
            if p is corr_normal:
                prec = preconditioning.FixedDensePreconditioner()
            else:
                prec = preconditioning.FixedDiagonalPreconditioner()
            kernel = autopcn.AutoPCN(potential_fn=p, preconditioner=prec)
            init_state = kernel.init(mcmc_key, 0, init_params, (), {})
            if p is corr_normal:
                precond_state = preconditioning.PreconditionerState(
                    mean = jnp.zeros_like(init_params),
                    var = S,
                    var_tril_factor = L,
                    inv_var_triu_factor= U
                )
            else:
                precond_state = init_state.base_precond_state
            s = kernel.update_log_joint(
                kernel.refresh_aux_vars(refresh_key,init_state,precond_state),
                precond_state
            )

            # check step size lattice respects the limits
            step_sizes = kernel.step_size(1.0, jnp.arange(-20,21))
            self.assertTrue(
                jnp.all(jnp.logical_and(step_sizes>0, step_sizes<jnp.pi))
            )

            # test invariance of the log joint under rotations when target is 
            # Gaussian and the preconditioner matches exactly
            def vmap_fn(eps):
                sn = kernel.update_log_joint(
                    kernel.involution_main(eps, s, precond_state),
                    precond_state
                )
                return jnp.array(
                    (
                        sn.log_prior,
                        -kernel.kinetic_energy(sn,precond_state),
                        sn.log_joint
                    )
                )
            logprobs = jax.vmap(vmap_fn)(step_sizes)
            self.assertTrue(
                jnp.allclose(logprobs[:,0]+logprobs[:,1], logprobs[:,-1])
            )
            self.assertLess(logprobs[:,-1].std(), 0.01)

if __name__ == '__main__':
    unittest.main()
    