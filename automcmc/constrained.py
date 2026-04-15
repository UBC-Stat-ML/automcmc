from typing import NamedTuple

import jax
from jax.typing import ArrayLike
from jax import flatten_util, random, numpy as jnp

from automcmc import utils, autostep, preconditioning
from automcmc.automcmc import AutoMCMCState

class ConstraintState(NamedTuple):
    """
    A :class:`NamedTuple` type for storing constraint-related quantities
    evaluated at a specific point in space.

    Attributes:
        is_satisfied: true if the constraint is satisfied up to tolerance.
        Q: Q factor of the QR decomposition of :math:`J_x^T` where :math:`J_x`
            is the Jacobian evaluated at :math:`x`. Used for projection onto
            normal space.
        log_abs_det: log of :math:`|(J_xJ_x^T)|^{-1/2}` computed using the
            R factor of the QR decomposition.
    """
    is_satisfied: ArrayLike
    Q: ArrayLike
    log_abs_det: ArrayLike

def make_constraint_state(is_satisfied: ArrayLike, J: ArrayLike):
    """
    Build :class:`ConstraintState`.

    :param is_satisfied: is the constraint satisfied?
    :param J: Jacobian matrix of shape `(m,n)` with `m<n`.
    """
    m, n = jnp.shape(J)
    assert m < n
    Q,R = jnp.linalg.qr(J.T)
    log_abs_det = -jnp.log(jnp.abs(jnp.diag(R))).sum()
    return ConstraintState(is_satisfied, Q, log_abs_det)

# project onto normal (N) space
# P[N]v = J^T(JJ^T)^{-1}Jv = Q(Q^Tv)
# Cost: O(mn^2 + nm^2)
def proj_normal(cs: ConstraintState, v: ArrayLike) -> jax.Array:
    """
    Project a vector onto the normal space at `x`.

    :param v: vector of length `n`.
    :return: normal component of `v`.
    """
    return cs.Q @ jnp.dot(v, cs.Q)

# project onto normal (N) and tangent (T) spaces
# P[T] = v - P[N]v
# Cost: O(mn^2 + nm^2)
def proj_normal_tangent(
        cs: ConstraintState,
        v: ArrayLike
    ) -> tuple[jax.Array, jax.Array]:
    """
    Project a vector onto the normal and tangent spaces at `x`.

    :param v: vector of length `n`.
    :return: normal and tangent components of `v`.
    """
    PNv = proj_normal(cs, v)
    return (PNv, v - PNv)


class AutoConstrainedRWMH(autostep.AutoStep):
    """
    Implementation of constrained random walk Metropolis-Hastings as described
    in [1,2], within an AutoStep framework for automatic selection of the
    proposal step size.

    .. rubric:: References

    [1] Zappa, E., Holmes-Cerfon, M. & Goodman, J. (2018).
    Monte Carlo on manifolds: Sampling densities and integrating functions.
    *Comm. Pure Appl. Math.*, 71, 2609-2647.

    [2] Xu, K., & Holmes-Cerfon, M. (2024). Monte Carlo on manifolds in high
    dimensions. *Journal of Computational Physics*, 506, 112939.
    """

    def __init__(
            self,
            *args,
            preconditioner=preconditioning.IdentityDiagonalPreconditioner(), # we need rotational symmetry
            constraint_fn=None, # aim is to target `constraint_fn=0`. Input var is x_flat.
            solver_options = {},
            x_tols = {},
            **kwargs
        ):
        super().__init__(*args,preconditioner=preconditioner,**kwargs)
        assert isinstance(
            self.preconditioner, preconditioning.IdentityDiagonalPreconditioner
        ), "This method is justified only for `IdentityDiagonalPreconditioner`" \
          f" but I got instead {type(self.preconditioner)}"
        self.constraint_fn = constraint_fn
        self.solver_options = solver_options
        self.x_tols = x_tols

    # TODO: using vjp for J^Tz=(z^TJ)^T instead of Qz would be more memory
    # efficient **if** we can get rid of Q in projection to tangent space
    def proj_level_set(
            self,
            x_flat: ArrayLike,
            cs: ConstraintState
        ) -> tuple:
        """
        Project a point on the constraint set

        :param ArrayLike x_flat: a vector.
        :param ConstraintState cs: a :class:`ConstraintState`.
        :return tuple: projected vector, updated :class:`ConstraintState`, and
            solver diagnostics.
        """
        # project the initial point to the feasible set
        #   solve for z: 0 = f(x + Qz) => set x' = x + Qz
        z, *diagnostics, is_satisfied = utils.newton(
            lambda z: self.constraint_fn(x_flat + cs.Q @ z),
            jnp.zeros(cs.Q.shape[-1], cs.Q.dtype),
            **self.solver_options
        )
        x_flat += cs.Q @ z

        # update the constraint state and return
        cs = make_constraint_state(
            is_satisfied, jax.jacrev(self.constraint_fn)(x_flat)
        )
        return (x_flat, cs, diagnostics)

    def init_extras(self, state):
        # TODO: extract constraint_fn for numpyro models (via a deterministic
        # stmtnt+model_kwargs or a Delta dist (problem: breaks log_prob))

        # set the constraint tolerance
        x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
        if 'tol' not in self.solver_options:
            self.solver_options['tol'] = utils.newton_default_tol(x_flat)

        # set the ambient space tolerances (`allclose` parameters)
        # prefer using only rtol, but need to have atol>0 if the origin
        # satisfies the constraint
        if 'rtol' not in self.x_tols:
            self.x_tols['rtol'] = jnp.sqrt(jnp.finfo(x_flat.dtype).eps)
        if 'atol' not in self.x_tols:
            self.x_tols['atol'] = len(x_flat)*self.solver_options['tol'] # recommended in Xu&Holmes-Cerfon(2024)

        # initialize the constraint state
        cs = make_constraint_state(
            False, jax.jacrev(self.constraint_fn)(x_flat)
        )

        # project to zero level set
        x_flat, cs, diagnostics = self.proj_level_set(x_flat, cs)
        if not cs.is_satisfied:
            raise ValueError(
                "Cannot find feasible starting point. " \
                f"Solver output:\n{diagnostics}"
            )

        # return updated state
        return state._replace(x = unravel_fn(x_flat), idiosyncratic = cs)

    def kinetic_energy(self, state, precond_state):
        return 0.5*jnp.dot(state.p_flat, state.p_flat)

    # sample and project to tangent space
    def refresh_aux_vars(self, rng_key, state, precond_state):
        return state._replace(
            p_flat = proj_normal_tangent(
                state.idiosyncratic,
                random.normal(rng_key, jnp.shape(state.p_flat))
            )[-1]
        )

    # need to
    #   - add the log of the co-area formula factor
    #   - make the likelihood 0 when constraint is not satisfied, so that
    #     solver failures are handled gracefully by the autostep executors
    def postprocess_logprior_and_loglik(self, state, log_prior, log_lik):
        cs = state.idiosyncratic
        log_prior += cs.log_abs_det
        log_lik = jnp.where(cs.is_satisfied, log_lik, -jnp.inf)
        return log_prior, log_lik

    def close_in_ambient_space(self, x, y):
        return jnp.allclose(
            x, y, atol=self.x_tols['atol'], rtol=self.x_tols['rtol']
        )

    def maybe_build_roundtrip_state(
            self,
            bwd_step_size: ArrayLike,
            prop_state_flip: AutoMCMCState,
            precond_state: preconditioning.PreconditionerState
        ):
        return self.involution_aux(
            self.involution_main(bwd_step_size, prop_state_flip, precond_state)
        )

    def reversibility_check(
            self,
            fwd_exponent: int,
            bwd_exponent: int,
            initial_state: AutoMCMCState,
            proposed_state: AutoMCMCState,
            roundtrip_state: AutoMCMCState
        ) -> bool:
        passed = fwd_exponent == bwd_exponent
        passed = jnp.logical_and(
            passed, proposed_state.idiosyncratic.is_satisfied
        )
        passed = jnp.logical_and(
            passed, roundtrip_state.idiosyncratic.is_satisfied
        )
        passed = jnp.logical_and(
            passed,
            self.close_in_ambient_space(
                flatten_util.ravel_pytree(initial_state.x)[0],
                flatten_util.ravel_pytree(roundtrip_state.x)[0]
            )
        )
        return jnp.logical_and(
            passed,
            self.close_in_ambient_space(
                initial_state.p_flat, roundtrip_state.p_flat
            )
        )


    # Both position and velocity are affected by this transform
    #   - position is updated by moving along velocity and projecting
    #   - velocity is updated by projecting the displacement vector
    def involution_main(self, step_size, state, precond_state):
        # move outside the level set: x <- x+s*v
        x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
        v_flat = state.p_flat
        x_flat_out = x_flat + step_size * v_flat

        # project back to level set
        x_flat_new,cs,_ = self.proj_level_set(x_flat_out, state.idiosyncratic)

        # transport the velocity by projecting the displacement vector x'-x
        # onto the tangent space at the new point
        # note: the displacement is on scale step_size * v, so we must divide
        # by the step size to get the right scale
        v_flat = proj_normal_tangent(cs, x_flat_new - x_flat)[-1] / step_size

        # update state and return
        return state._replace(
            x = unravel_fn(x_flat_new),
            p_flat = v_flat,
            idiosyncratic = cs
        )

    # only need to flip sign of the velocity, which was already transported
    # to the new tangent space in `involution_main`
    def involution_aux(self, state):
        return state._replace(p_flat = -state.p_flat)
