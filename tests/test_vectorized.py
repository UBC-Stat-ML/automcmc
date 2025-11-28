from tests import utils as testutils

import unittest

from jax import random

from numpyro.infer import MCMC

from automcmc import autohmc
from automcmc import autorwmh
from automcmc import slicer
from automcmc import preconditioning
from automcmc import utils
    
class TestVectorized(unittest.TestCase):

    TESTED_KERNELS = (
        autorwmh.AutoRWMH,
        autohmc.AutoMALA,
        autohmc.AutoHMC,
        slicer.DeterministicScanSliceSampler,
        slicer.HitAndRunSliceSampler
    )

    TESTED_PRECONDITIONERS = (
        preconditioning.FixedDiagonalPreconditioner(),
        preconditioning.FixedDensePreconditioner()
    )

    def test_vectorized(self):
        rng_key = random.key(9)
        n_chains = 4
        n_rounds = 4
        n_warmup, n_keep = utils.split_n_rounds(n_rounds)
        model, model_args, model_kwargs = testutils.make_eight_schools()
        for kernel_class in self.TESTED_KERNELS:
            for prec in self.TESTED_PRECONDITIONERS:
                with self.subTest(kernel_class=kernel_class, prec_type=type(prec)):
                    print(f"kernel_class={kernel_class}, prec_type={type(prec)}")
                    rng_key, run_key = random.split(rng_key)
                    kernel = kernel_class(model, preconditioner = prec)
                    mcmc = MCMC(
                        kernel, 
                        num_warmup=n_warmup, 
                        num_samples=n_keep,
                        num_chains=n_chains,
                        chain_method="vectorized", 
                        progress_bar=False
                    )
                    mcmc.run(run_key, *model_args, **model_kwargs)
                    self.assertEqual(mcmc.last_state.x['mu'].shape, (n_chains,))
                    self.assertEqual(mcmc.last_state.p_flat.shape[0], n_chains)

        with self.assertRaises(NotImplementedError):
            rng_key, run_key = random.split(rng_key)
            kernel=kernel_class(
                model, preconditioner = prec, optimize_init_params=True
            )
            mcmc = MCMC(
                kernel, 
                num_warmup=n_warmup, 
                num_samples=n_keep,
                num_chains=n_chains,
                chain_method="vectorized", 
                progress_bar=False
            )
            mcmc.run(run_key, *model_args, **model_kwargs)

                

if __name__ == '__main__':
    unittest.main()
