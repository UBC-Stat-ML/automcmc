from tests import utils as testutils

from functools import partial
import unittest

import numpy as np
from scipy import stats, integrate

import jax
from jax import numpy as jnp

from automcmc import constrained,utils,preconditioning,selectors
from numpyro.infer import MCMC

class TestConstrained(unittest.TestCase):

    def test_invalid_inputs(self):
        with self.assertRaises(AssertionError):
            constrained.AutoConstrainedRWMH(
                potential_fn=lambda x: 0.0,
                preconditioner=preconditioning.FixedDensePreconditioner(),
            )

    def test_Jacobian_algebra(self):
        m = 6
        n = 145
        J_key, v_key = jax.random.split(jax.random.key(1),2)
        v = jax.random.normal(v_key, (n,))
        J = jax.random.normal(J_key, (m,n))
        tol = jnp.sqrt(jnp.finfo(J.dtype).eps)
        cs = constrained.make_constraint_state(True,J) # err is irrelevant here
        self.assertAlmostEqual(1, jnp.linalg.cond(cs.Q),delta=tol)
        self.assertGreater(jnp.linalg.cond(J.T), 1.1)
        self.assertLess(
            utils.newton_fn_value_err(cs.Q.T@cs.Q - jnp.identity(m)), tol
        )
        self.assertTrue(jnp.isclose(
            cs.log_abs_det,
            -0.5*jnp.log(jnp.abs(jnp.linalg.det(jnp.inner(J,J)))),
            rtol = 0.01
        ))
        _,PTv = constrained.proj_normal_tangent(cs, v)
        self.assertTrue(jnp.abs(J@PTv).max() < tol) # PTv should be orthogonal to every row of J

    def test_constrained_involution(self):
        # std normal prior on R^n
        # constrain to unit circle
        potential_fn = lambda x: 0.5*jnp.dot(x,x)
        constraint_fn = lambda x: (x*x).sum(keepdims=True)-1
        rng_key, randn_key = jax.random.split(jax.random.key(1))
        n_dim = 30
        step_size = 0.5/jnp.sqrt(n_dim) # step should prob decrease with dim (assume same rate as std rwmh)
        all_init_params = {
            "randn": jax.random.normal(randn_key,n_dim),
            "ones": jnp.ones(n_dim),
            "arange":jnp.arange(n_dim,dtype=float)
        }
        for init_params_type, init_params in all_init_params.items():
            for mode in ("direct","gmres"):
                with self.subTest(mode=mode,init_params_type=init_params_type):
                    kernel = constrained.AutoConstrainedRWMH(
                        potential_fn=potential_fn,
                        constraint_fn=constraint_fn,
                        solver_options={'mode': mode}
                    )

                    # test initialization to feasible set
                    rng_key, init_key, refresh_key = jax.random.split(rng_key, 3)
                    state = kernel.init(init_key, 0, init_params, (), {})
                    tol = kernel.solver_options['tol']
                    self.assertTrue(state.idiosyncratic.is_satisfied)
                    self.assertLess(
                        utils.newton_fn_value_err(constraint_fn(state.x)), tol
                    )

                    # test log joint and velocity refreshment
                    precond_state = kernel.preconditioner.maybe_alter_precond_state(
                        state.base_precond_state, 0
                    )
                    state = kernel.update_log_joint(
                        kernel.refresh_aux_vars(refresh_key, state, precond_state),
                        precond_state
                    )
                    self.assertAlmostEqual(
                        state.log_prior,
                        state.idiosyncratic.log_abs_det - potential_fn(state.x),
                        delta=tol
                    )
                    self.assertAlmostEqual(
                        jnp.abs(jnp.dot(state.p_flat,state.idiosyncratic.Q))[0],
                        0,
                        delta=tol
                    )

                    # test involutive property
                    state_half = kernel.involution_main(
                        step_size, state, precond_state
                    )
                    self.assertTrue(state_half.idiosyncratic.is_satisfied)
                    self.assertLess(
                        utils.newton_fn_value_err(constraint_fn(state_half.x)), tol
                    )
                    self.assertFalse(kernel.close_in_ambient_space(
                            state_half.x, state.x
                    ))
                    self.assertAlmostEqual(
                        jnp.abs(jnp.dot(state_half.p_flat,state_half.idiosyncratic.Q))[0],
                        0,
                        delta=tol
                    )
                    self.assertAlmostEqual(
                        # due to nature of this problem, velocities are also
                        # rotating around, and therefore its density (std
                        # normal) should be preserved
                        kernel.kinetic_energy(state_half, precond_state),
                        kernel.kinetic_energy(state, precond_state),
                        delta = n_dim*tol
                    )
                    state_one = kernel.involution_aux(state_half)
                    state_onehalf = kernel.involution_main(
                        step_size, state_one, precond_state
                    )
                    self.assertTrue(state_onehalf.idiosyncratic.is_satisfied)
                    self.assertLess(
                        utils.newton_fn_value_err(constraint_fn(state_onehalf.x)),
                        tol
                    )
                    state_two = kernel.involution_aux(state_onehalf)
                    self.assertTrue(kernel.close_in_ambient_space(
                        state_two.x, state.x
                    ))
                    self.assertTrue(kernel.close_in_ambient_space(
                        state_two.p_flat, state.p_flat
                    ))

    def test_sampling_torus(self):
        # T^2 torus embedded in R^3
        # Example 1 in Zappa & Holmes-Cerfon (2018)
        # Note: this example satisfies JJ^T = constant, which for the
        # particular choices of (R=1,r=1/2) used here give JJ^T = 1.
        # Hence, the Lebesgue measure induces the uniform dist on the surface

        # define problem params
        R, r = 1.0, 0.5
        rng_key = jax.random.key(1)

        # Setup 1-sample KS tests on known (phi, theta) marginals
        theta_cdf_fn = partial(stats.uniform.cdf, scale=2*np.pi)
        # phi_pdf_fn = lambda x: (1+(r/R)*np.cos(x)) / (2*np.pi)
        phi_cdf_fn = lambda q: (q+(r/R)*jnp.sin(q)) / (2*jnp.pi)

        # check problem functions are correct
        constraint_fn = partial(testutils.torus_constraint, R, r)
        self.assertGreater(
            jnp.abs(constraint_fn(jnp.zeros(3))[0]), 0.1
        )
        param_fn = partial(testutils.torus_param, R, r)
        inv_param_fn = partial(testutils.inv_torus_param, R, r)
        rng_key, angles_key, mcmc_key = jax.random.split(rng_key, 3)
        theta, phi = 2*jnp.pi*jax.random.uniform(angles_key, (2,2**10))
        theta2, phi2 = inv_param_fn(*param_fn(theta, phi))
        self.assertTrue(jnp.allclose(theta, theta2))
        self.assertTrue(jnp.allclose(phi, phi2))
        self.assertLess(
            utils.newton_fn_value_err(
                jax.vmap(constraint_fn,in_axes=(1,))(param_fn(theta, phi))
            ),
            1e-6
        )

        # mcmc sampling
        # use thinning to approximate independent sampling so KS test
        # assumption is "less broken"
        mode = "direct"
        potential_fn = lambda x: jnp.zeros_like(x,shape=()) # uniform
        n_warm, n_keep = utils.split_n_rounds(15)
        thinning=32 # %ESS ~ 1/32
        init_params = jnp.ones(3) # init outside level set on purpose
        extra_fields = ('idiosyncratic.log_abs_det',)
        for sel in (
            selectors.FixedStepSizeSelector(),
            selectors.DeterministicSymmetricSelector(),
        ):
            rng_key, mcmc_key = jax.random.split(rng_key)
            kernel = constrained.AutoConstrainedRWMH(
                potential_fn=potential_fn,
                constraint_fn=constraint_fn,
                solver_options={'mode': mode},
                init_base_step_size = 0.5, # step used in paper
                selector = sel
            )
            mcmc = MCMC(
                kernel,
                num_warmup=n_warm,
                num_samples=n_keep,
                thinning=thinning,
                progress_bar=False
            )
            mcmc.run(mcmc_key,init_params=init_params, extra_fields=extra_fields)
            log_abs_det = next(iter((mcmc.get_extra_fields().values())))
            self.assertLess(jnp.abs(log_abs_det).max(), kernel.x_tols['atol']) # check that they are all ~0
            samples = mcmc.get_samples()
            self.assertLessEqual(
                utils.newton_fn_value_err(jax.vmap(constraint_fn)(samples)),
                kernel.solver_options['tol']
            )
            theta, phi = inv_param_fn(*samples.T)
            self.assertAlmostEqual(theta.mean(), jnp.pi, delta=0.15)
            self.assertAlmostEqual(phi.mean(), jnp.pi, delta=0.15)

            # KS tests
            self.assertGreater(stats.ks_1samp(theta, theta_cdf_fn).pvalue, 0.01)
            self.assertGreater(stats.ks_1samp(phi, phi_cdf_fn).pvalue, 0.01)


    def test_sampling_cone(self):
        # cone embedded in R^3
        # Example 2 in Zappa & Holmes-Cerfon (2018)
        # Note: this example satisfies JJ^T = constant, so the Lebesgue measure
        # induces the uniform dist on the surface

        # mcmc sampling
        rng_key = jax.random.key(67534)
        init_params=jnp.full((3,), 0.5) # init in interior of cone
        n_warm, n_keep = utils.split_n_rounds(14)
        thinning=16 # %ESS ~ 1/16
        extra_fields = ('idiosyncratic.log_abs_det',)
        for sel in (
            selectors.FixedStepSizeSelector(),
            selectors.DeterministicSymmetricSelector(),
        ):
            rng_key, mcmc_key = jax.random.split(rng_key)
            kernel = constrained.AutoConstrainedRWMH(
                potential_fn=testutils.cone_potential,
                constraint_fn=testutils.cone_constraint,
                solver_options={'mode': 'direct'},
                init_base_step_size = 0.9, # same as in paper
                selector = sel
            )
            mcmc = MCMC(
                kernel,
                num_warmup=n_warm,
                num_samples=n_keep,
                thinning=thinning,
                progress_bar=False
            )
            mcmc.run(
                mcmc_key, init_params=init_params, extra_fields=extra_fields
            )
            log_abs_det = next(iter((mcmc.get_extra_fields().values())))
            self.assertLess(
                jnp.abs(log_abs_det+jnp.log(2)/2).max(), 1e-5 # check it matches the known constant
            )

            # ks tests
            xs,ys,zs = mcmc.get_samples().T
            xy_cdf = lambda q: (q*jnp.sqrt(1-q*q) + jnp.arcsin(q))/jnp.pi + 0.5 # thx 2 symbolic integration
            z_cdf = np.square
            self.assertGreater(stats.ks_1samp(xs, xy_cdf).pvalue, 0.01)
            self.assertGreater(stats.ks_1samp(ys, xy_cdf).pvalue, 0.01)
            self.assertGreater(stats.ks_1samp(zs, z_cdf).pvalue, 0.01)



if __name__ == '__main__':
    unittest.main()
