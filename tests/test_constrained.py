import unittest
import jax
from jax import numpy as jnp

from automcmc import constrained,utils,preconditioning

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
        self.assertTrue(jnp.abs(cs.Q.T@cs.Q - jnp.identity(m)).max() < tol)
        self.assertTrue(jnp.isclose(
            cs.log_abs_det,
            -0.5*jnp.log(jnp.abs(jnp.linalg.det(jnp.inner(J,J)))),
            rtol = 0.01
        ))
        _,PTv = constrained.proj_normal_tangent(cs, v)
        self.assertTrue(jnp.abs(J@PTv).max() < tol) # PTv should be orthogonal to every row of J

    def test_involution(self):
        # std normal prior on R^n
        # constrain to unit circle
        potential_fn = lambda x: 0.5*jnp.dot(x,x)
        constraint_fn = lambda x: (x*x).sum(keepdims=True)-1
        rng_key = jax.random.key(1)
        n_dim = 30

        for mode in ("direct","gmres"):
            kernel = constrained.AutoConstrainedRWMH(
                potential_fn=potential_fn,
                constraint_fn=constraint_fn,
                solver_options={
                    'tol': 10*jnp.finfo(jnp.float32).eps,
                    'mode': mode
                }
            )

            # test initialization to feasible set
            rng_key, init_key, refresh_key = jax.random.split(rng_key, 3)
            init_params = jnp.ones(n_dim)
            state = kernel.init(init_key, 0, init_params, (), {})
            tol = kernel.solver_options['tol']
            self.assertTrue(
                utils.newton_fn_value_err(constraint_fn(state.x)) < tol and
                state.idiosyncratic.is_satisfied
            )
            self.assertTrue(jnp.allclose(
                # due to the characteristics of the problem, solution must
                # coincide with the orthogonal projection
                state.x, init_params / jnp.sqrt(n_dim), rtol=0.01
            ))

            # test log joint and velocity refreshment
            precond_state = kernel.preconditioner.maybe_alter_precond_state(
                state.base_precond_state, 0
            )
            state = kernel.update_log_joint(
                kernel.refresh_aux_vars(refresh_key, state, precond_state), precond_state
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
            step_size = jax.lax.rsqrt(n_dim+0.0) # step should prob decrease with dim (assume same rate as std rwmh)
            state_half = kernel.involution_main(step_size, state, precond_state)
            self.assertTrue(
                utils.newton_fn_value_err(constraint_fn(state_half.x)) < tol and
                state_half.idiosyncratic.is_satisfied
            )
            self.assertFalse(jnp.allclose(state_half.x, state.x, atol=10*tol, rtol=0.01))
            self.assertAlmostEqual(
                jnp.abs(jnp.dot(state_half.p_flat,state_half.idiosyncratic.Q))[0],
                0,
                delta=tol
            )
            self.assertAlmostEqual(
                # due to nature of this problem, velocities are also rotating around, and
                # therefore its density (std normal) should be preserved
                kernel.kinetic_energy(state_half, precond_state),
                kernel.kinetic_energy(state, precond_state),
                delta = n_dim*tol
            )
            state_one = kernel.involution_aux(state_half)
            state_onehalf = kernel.involution_main(step_size, state_one, precond_state)
            self.assertTrue(
                utils.newton_fn_value_err(constraint_fn(state_onehalf.x)) < tol and
                state_onehalf.idiosyncratic.is_satisfied
            )
            state_two = kernel.involution_aux(state_onehalf)
            self.assertTrue(
                jnp.allclose(state_two.x, state.x, atol=10*tol, rtol=0.01)
            )
            self.assertTrue(
                jnp.allclose(state_two.p_flat, state.p_flat, atol=10*tol, rtol=0.01)
            )



if __name__ == '__main__':
    unittest.main()