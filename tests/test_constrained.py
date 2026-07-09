from tests import utils as testutils

from functools import partial
import unittest

import numpy as np
from scipy import stats

import jax
from jax import numpy as jnp

from automcmc import (
    constrained,
    utils,
    preconditioning,
    selectors,
    optimization
)
from automcmc.constrained import LevelSetHandlerCholesky, LevelSetHandlerQR
from numpyro.infer import MCMC

TESTED_LEVELSET_HANDLERS = (LevelSetHandlerCholesky(), LevelSetHandlerQR())

class TestConstrained(unittest.TestCase):

    def test_invalid_inputs(self):
        pot = lambda x: 0.0
        with self.assertRaisesRegex(ValueError, "You must supply one"):
            constrained.AutoConstrainedRWMH(potential_fn=pot)

        with self.assertRaisesRegex(AssertionError,"This method is justified"):
            constrained.AutoConstrainedRWMH(
                potential_fn=pot,
                preconditioner=preconditioning.FixedDensePreconditioner(),
            )

        # check unfeasible problem is caught
        init_params = jnp.zeros(5)
        kernel = constrained.AutoConstrainedRWMH(
            potential_fn=pot,
            fwd_model=lambda x: jnp.ones_like(x, shape=(2,)),
            init_obs_output=jnp.array([-3,3]), # not in range of fwd mod
        )
        with self.assertRaisesRegex(AssertionError, "Cannot find feasible"):
            kernel.init(jax.random.key(1), 0, init_params, (), {})

        # check non full rank jacobian is caught
        init_params = jnp.array([1.0,0.0,0.0])
        kernel = constrained.AutoConstrainedRWMH(
            potential_fn=pot,
            fwd_model=lambda x: jnp.stack(2*(jnp.linalg.norm(x),)),
            init_obs_output=jnp.array([1.,1.]), # not in range of fwd mod
        )
        with self.assertRaisesRegex(AssertionError, "The Jacobian may not be"):
            kernel.init(jax.random.key(1), 0, init_params, (), {})


    def test_proj_normal_tangent(self):
        d = 3
        n = d*d
        m = (n+d)//2 # d(d+1)//2
        n_sim = 10
        rng_key = jax.random.key(1)
        for levelset_handler in TESTED_LEVELSET_HANDLERS:
            for n_rep in range(n_sim):
                with self.subTest(
                    levelset_handler_type=type(levelset_handler), n_rep=n_rep,
                ):
                    X = jnp.array(stats.ortho_group.rvs(dim=d,)) # iid sample SO(3)
                    constraint_fn = testutils.orthonormal_constraint
                    tol = utils.newton_default_tol(X)
                    lss = levelset_handler.make_levelset_state(
                        constraint_fn, 0, X.flatten(), tol
                    )
                    self.assertTrue(lss.is_satisfied)
                    J = jax.jacobian(constraint_fn)(lss.x_base)
                    gram_mat = jnp.inner(J,J)
                    if isinstance(levelset_handler, LevelSetHandlerCholesky):
                        self.assertEqual((m,m), lss.mat.shape)
                        self.assertLess(
                            jnp.linalg.norm(gram_mat - jnp.inner(lss.mat,lss.mat)),
                            tol
                        )
                    else:
                        self.assertEqual((n,m), lss.mat.shape)

                    self.assertAlmostEqual(
                        lss.log_abs_det,
                        -0.5*jnp.log(jnp.abs(jnp.linalg.det(gram_mat))),
                        delta = 0.002
                    )
                    rng_key, v_key = jax.random.split(rng_key)
                    v = jax.random.normal(v_key, (n,))
                    PNv, PTv = levelset_handler.proj_normal_tangent(
                        constraint_fn, lss, v
                    )
                    self.assertTrue(jnp.allclose(v, PNv+PTv))
                    jvp_val = jax.jvp(constraint_fn, (lss.x_base,), (PTv,))[-1] # PTv \perp to every row of Jac
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
                    state_one = kernel.involution(
                        step_size, state, precond_state
                    )
                    self.assertTrue(state_one.idiosyncratic.is_satisfied)
                    self.assertLess(
                        utils.newton_fn_value_err(constraint_fn(state_one.x)),
                        tol
                    )
                    self.assertFalse(kernel.close_in_ambient_space(
                            state_one.x, state.x
                    ))
                    self.assertAlmostEqual(
                        # due to nature of this problem, velocities are also
                        # rotating around, and therefore its density (std
                        # normal) should be preserved
                        kernel.kinetic_energy(state_one, precond_state),
                        kernel.kinetic_energy(state, precond_state),
                        delta = n_dim*tol
                    )
                    state_two = kernel.involution(
                        step_size, state_one, precond_state
                    )
                    self.assertTrue(state_two.idiosyncratic.is_satisfied)
                    self.assertLess(
                        utils.newton_fn_value_err(
                            constraint_fn(state_two.x)
                        ),
                        tol
                    )
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
            self.assertLess(jnp.abs(log_abs_det).max(), 2*kernel.x_tols['atol']) # check that they are all ~0
            samples = mcmc.get_samples()
            self.assertLessEqual(
                utils.newton_fn_value_err(jax.vmap(constraint_fn)(samples)),
                kernel.solver_options['tol']
            )
            theta, phi = inv_param_fn(*samples.T)
            self.assertAlmostEqual(theta.mean(), jnp.pi, delta=0.15)
            self.assertAlmostEqual(phi.mean(), jnp.pi, delta=0.15)

            # KS tests
            self.assertGreater(stats.ks_1samp(theta, theta_cdf_fn).pvalue, 0.001)
            self.assertGreater(stats.ks_1samp(phi, phi_cdf_fn).pvalue, 0.001)


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
                self.assertGreater(stats.ks_1samp(xs, xy_cdf).pvalue, 0.001)
                self.assertGreater(stats.ks_1samp(ys, xy_cdf).pvalue, 0.001)
                self.assertGreater(stats.ks_1samp(zs, z_cdf).pvalue, 0.001)


    def test_sample_orthonormal(self):
        # matrices with orthonormal rows
        # Example 3 in Zappa & Holmes-Cerfon (2018)
        constraint_fn=testutils.orthonormal_constraint
        potential_fn = lambda x: jnp.zeros_like(x,shape=()) # uniform
        d = 11
        n_dim = d*d

        for levelset_handler in TESTED_LEVELSET_HANDLERS:
            with self.subTest(levelset_handler_type=type(levelset_handler)):
                # mcmc sampling
                rng_key, init_key = jax.random.split(jax.random.key(53))
                init_params=jax.random.normal(init_key, (n_dim,)) # init in interior of cone
                n_warm, n_keep = utils.split_n_rounds(13)
                thinning=2**6 # %ESS ~ 1/64
                extra_fields = ('idiosyncratic.log_abs_det',)
                rng_key, mcmc_key = jax.random.split(rng_key)
                kernel = constrained.AutoConstrainedRWMH(
                    potential_fn=potential_fn,
                    constraint_fn=constraint_fn,
                    solver_options={'mode': 'direct'},
                    init_base_step_size = 0.28, # in paper
                    selector = selectors.FixedStepSizeSelector(),
                    levelset_handler=levelset_handler
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
                self.assertLess(log_abs_det.std(), 0.05)

                # ks test for normality of traces of the matrices
                traces = jax.vmap(
                    lambda v: jnp.diag(v.reshape((d,d))).sum()
                )(mcmc.get_samples())
                self.assertGreater(
                    stats.ks_1samp(traces, stats.norm.cdf).pvalue, 0.01
                )

    def test_constrained_vectorized(self):
        # params
        n_chains = 64
        n_dim = 2
        rng_key = jax.random.key(9)
        n_warm, n_keep = utils.split_n_rounds(13)
        thinning=2**3

        # define an equally spaced grid in [0,1]
        # check the Riemann integral recovers the truth (2/3) for expected radius
        init_obs_output = jnp.arange(1, n_chains+1).reshape((n_chains,1))/n_chains
        dr = init_obs_output[1,0]-init_obs_output[0,0]
        self.assertAlmostEqual(
            2*dr*jnp.square(init_obs_output).sum(), 2/3, delta=0.02
        )

        # define sampler
        potential_fn=lambda x: jnp.zeros((), x.dtype)
        fwd_model=lambda x: jnp.array([jnp.linalg.norm(x)])
        mcmc_key, params_key = jax.random.split(rng_key)
        init_params = jax.random.normal(params_key,(n_chains, n_dim))
        kernel = constrained.AutoConstrainedRWMH(
            potential_fn=potential_fn,
            fwd_model=fwd_model,
            init_obs_output=init_obs_output
        )

        # run mcmc and get samples
        mcmc = MCMC(
            kernel,
            num_warmup=n_warm,
            num_samples=n_keep,
            thinning=thinning,
            num_chains=n_chains,
            chain_method="vectorized",
            progress_bar=False
        )
        mcmc.run(mcmc_key,init_params=init_params)
        samples = mcmc.get_samples(True)

        # make sure we didn't request too small radii
        self.assertGreater(init_obs_output[0,0], kernel.solver_options['tol'])

        # check that the fwd model is respected along chains at every sample step
        fwd_vals = jax.vmap(jax.vmap(fwd_model))(samples)
        self.assertTrue(jnp.all(
            jax.vmap(
                partial(jnp.allclose, rtol=0, atol=kernel.solver_options['tol']),
                in_axes=(1,None),
            )(fwd_vals, init_obs_output)
        ))

        # but also check that the submanifolds are explored by ensuring the
        # angles follow a Unif(-pi,pi) distribution
        # use a simple histogram check instead of KS test (too sensitive at
        # this number of samples, which are not really indep anyway)
        # impose grid to avoid being fooled by constant data
        angles = jnp.arctan2(samples[:,:,0],samples[:,:,1]).flatten()
        hist = jnp.histogram(angles,bins=jnp.linspace(-jnp.pi, jnp.pi, 11))
        self.assertTrue(
            jnp.allclose(hist[0], angles.size/10, rtol=0.15)
        )

    def test_mrna(self):
        def prior_potential(x):
            assert x.shape == (4,)
            lt0, lkm0, lbeta, ldelta = x
            ok = jnp.logical_and(lt0 > -2, lt0 < 1)
            ok = jnp.logical_and(ok, jnp.logical_and(lkm0 > -5, lkm0 < 5))
            ok = jnp.logical_and(ok, jnp.logical_and(lbeta > -5, lbeta < 5))
            ok = jnp.logical_and(ok, jnp.logical_and(ldelta > -5, ldelta < 5))
            return jnp.where(ok, 0, jnp.inf)

        def ode_solution(km0, beta, delta, dt):
            abs_diff = jnp.abs(delta-beta)
            exp_prod = jnp.exp(-jnp.minimum(delta,beta)*dt)*jnp.expm1(-abs_diff*dt)
            return -(km0/abs_diff)*exp_prod

        def make_fwd_model(measured_times):
            assert len(measured_times) < 4
            def fwd_model(x):
                assert x.shape == (4,)
                t0, km0, beta, delta = 10**x
                rel_ts = jax.nn.relu(measured_times - t0)
                return ode_solution(km0, beta, delta, rel_ts)

            return fwd_model

        # define settings
        rng_key = jax.random.key(6)
        n_rounds = 14
        thinning = 2**4
        n_warm, n_keep = utils.split_n_rounds(n_rounds)
        measured_times = jnp.array((4.0,8.0,16.0))
        fwd_model = make_fwd_model(measured_times)

        # simulate an observation by using a given point in input space
        true_lambda = jnp.array(
            [0.18858075,  1.1060505, -2.875724, -0.7061024]
        )
        init_obs_output = fwd_model(true_lambda)

        # define initial parameters and sampler
        rng_key, init_key, mcmc_key = jax.random.split(rng_key,3)
        init_params = jax.random.uniform(init_key,(4,),minval=-1)
        levelset_finder_settings=(
            optimization.DEFAULT_OPTIMIZE_FUN_SETTINGS['NADAMW'].copy()
        )
        levelset_finder_settings['n_iter'] = 2**11
        kernel = constrained.AutoConstrainedRWMH(
            potential_fn=prior_potential,
            fwd_model=fwd_model,
            init_obs_output=init_obs_output,
            levelset_finder_settings=levelset_finder_settings
        )
        mcmc = MCMC(
            kernel,
            num_warmup=n_warm,
            num_samples=n_keep,
            thinning=thinning,
            progress_bar=False
        )
        mcmc.run(mcmc_key, init_params=init_params)
        assert True

    def test_double_torus_unif(self):
        # Sample the uniform distribution on the double torus by turning off
        # the co-area factor. This is necessary because the factor is not
        # constant on the zero level set.
        rng_key = jax.random.key(6)
        n_rounds = 10
        n_warm, n_keep = utils.split_n_rounds(n_rounds)
        init_key, mcmc_key = jax.random.split(rng_key)
        init_params = jax.random.uniform(init_key,(3,),minval=-1)
        extra_fields = ('idiosyncratic.log_abs_det', 'log_prior', 'log_lik')
        kernel = constrained.AutoConstrainedRWMH(
            potential_fn = lambda x: jnp.zeros_like(x,shape=()),
            constraint_fn = testutils.double_torus_constraint,
            init_base_step_size = 0.5,
            selector = selectors.DeterministicSymmetricSelector(p_hi=0.9),
            add_coarea_factor= False
        )
        mcmc = MCMC(
            kernel,
            num_warmup=n_warm,
            num_samples=n_keep,
            progress_bar=False
        )
        mcmc.run(mcmc_key, init_params=init_params,extra_fields = extra_fields)
        extra_fields_samples = mcmc.get_extra_fields()

        # check the factor changes along the surface
        self.assertGreater(
            extra_fields_samples['idiosyncratic.log_abs_det'].std(),
            0.1
        )

        # check the total log_prior is exactly zero (i.e., that the co-area
        # factor was not added to it)
        self.assertTrue(jnp.allclose(extra_fields_samples['log_prior'], 0))

        # check the log_lik is also zero (this is always true for constrained
        # sampling via `potential_fn`)
        self.assertTrue(jnp.allclose(extra_fields_samples['log_lik'], 0))



if __name__ == '__main__':
    unittest.main()
