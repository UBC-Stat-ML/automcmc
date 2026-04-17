from tests import utils as testutils

from functools import partial
import unittest

import numpy as np
from scipy import stats

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

    def test_proj_normal_tangent(self):
        d = 3
        n = d*d
        n_sim = 10
        rng_key = jax.random.key(1)
        for _ in range(n_sim):
            X = jnp.array(stats.ortho_group.rvs(dim=d,)) # iid sample SO(3)
            constraint_fn = testutils.orthonormal_constraint
            tol = utils.newton_default_tol(X)
            cs = constrained.make_constraint_state(constraint_fn,X.flatten(),tol)
            self.assertTrue(cs.is_satisfied)
            m = cs.chol.shape[0]
            self.assertEqual(m, d*(d+1)//2)
            J = jax.jacobian(constraint_fn)(cs.x_base)
            gram_mat = jnp.inner(J,J)
            self.assertLess(
                jnp.linalg.norm(gram_mat - jnp.inner(cs.chol,cs.chol)), tol
            )
            self.assertAlmostEqual(
                cs.log_abs_det,
                -0.5*jnp.log(jnp.abs(jnp.linalg.det(gram_mat))),
                delta = 0.001
            )
            rng_key, v_key = jax.random.split(rng_key)
            v = jax.random.normal(v_key, (n,))
            PNv,PTv = constrained.proj_normal_tangent(constraint_fn,cs,v)
            self.assertTrue(jnp.allclose(v, PNv+PTv))
            jvp_val = jax.jvp(constraint_fn, (cs.x_base,), (PTv,))[-1] # PTv \perp to every row of Jac
            self.assertLess(jnp.abs(jvp_val).max(), 0.01)
            jvp_val_dumb = J@PTv
            self.assertTrue(
                jnp.allclose(jvp_val, jvp_val_dumb, atol=tol, rtol=0.01),
                f"jvp_val={jvp_val} but J@PTv={jvp_val_dumb}"
            )


    def test_constrained_involution(self):
        potential_fn = lambda x: jnp.zeros_like(x,shape=()) # uniform
        rng_key = jax.random.key(1)
        n_dim = 121
        step_size = 0.5/jnp.sqrt(n_dim) # step should prob decrease with dim (assume same rate as std rwmh)
        constraint_fn_types= {
            'circle': lambda x: (x*x).sum(keepdims=True)-1,
            'orthonormal': testutils.orthonormal_constraint
        }
        n_reps = 3
        for constraint_fn_type, constraint_fn in constraint_fn_types.items():
            for n_rep in range(n_reps):
                with self.subTest(
                    constraint_fn_type=constraint_fn_type,
                    n_rep=n_rep,
                ):
                    # test initialization to feasible set
                    kernel = constrained.AutoConstrainedRWMH(
                        potential_fn=potential_fn,
                        constraint_fn=constraint_fn,
                    )
                    (
                        rng_key,
                        randn_key,
                        init_key,
                        refresh_key
                    ) = jax.random.split(rng_key, 4)
                    init_params = jax.random.normal(randn_key, (n_dim,))
                    state = kernel.init(init_key, 0, init_params, (), {})
                    tol = kernel.solver_options['tol']
                    self.assertTrue(state.idiosyncratic.is_satisfied)
                    self.assertLess(
                        utils.newton_fn_value_err(constraint_fn(state.x)), tol
                    )

                    # test log joint and velocity refreshment
                    preconditioner = kernel.preconditioner
                    precond_state = preconditioner.maybe_alter_precond_state(
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

                    # test involutive property
                    state_half = kernel.involution_main(
                        step_size, state, precond_state
                    )
                    self.assertTrue(state_half.idiosyncratic.is_satisfied)
                    self.assertLess(
                        utils.newton_fn_value_err(constraint_fn(state_half.x)),
                        tol
                    )
                    self.assertFalse(kernel.close_in_ambient_space(
                            state_half.x, state.x
                    ))
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
                        utils.newton_fn_value_err(
                            constraint_fn(state_onehalf.x)
                        ),
                        tol
                    )
                    state_two = kernel.involution_aux(state_onehalf)
                    self.assertTrue(
                        kernel.close_in_ambient_space(state_two.x, state.x),
                        f"|diff|/|x0| = {jnp.linalg.norm(state_two.x- state.x)/jnp.linalg.norm(state.x)}"
                    )
                    self.assertTrue(
                        kernel.close_in_ambient_space(
                            state_two.p_flat, state.p_flat
                        ),
                    )

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
            with self.subTest(sel=sel):
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
