import unittest
import jax
from jax import random
from jax import numpy as jnp

from automcmc import autopcn

class TestAutoPCN(unittest.TestCase):

    def test_autopcn(self):
        def std_normal(x):
            return 0.5*jnp.square(x).sum()

        rng_key = random.key(1)
        refresh_key, init_key, mcmc_key = random.split(rng_key,3)
        init_params = random.normal(init_key, (2,))
        kernel = autopcn.AutoPCN(potential_fn=std_normal)
        init_state = kernel.init(mcmc_key, 0, init_params, (), {})
        precond_state = init_state.base_precond_state
        s = kernel.update_log_joint(
            kernel.refresh_aux_vars(refresh_key,init_state,precond_state),
            precond_state
        )

        # check step size lattice respects the limits
        step_sizes = kernel.step_size(1.0, jnp.arange(-20,21))
        self.assertTrue(jnp.all(jnp.logical_and(step_sizes>0, step_sizes<jnp.pi)))

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
        self.assertTrue(jnp.allclose(logprobs[:,0]+logprobs[:,1], logprobs[:,-1]))
        self.assertLess(logprobs[:,-1].std(), jnp.finfo(logprobs.dtype).eps)

if __name__ == '__main__':
    unittest.main()
    