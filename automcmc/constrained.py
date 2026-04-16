from typing import NamedTuple, Callable

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
        is_satisfied: true if `f(x_base)=0` up to tolerance.
        x_base: point that dictates the level set :math:`f^{-1}(\\{x\\})` and
            the tangent and normal spaces that determine the projections onto
            said level set. We assume `x_base` is flattened.
        chol: Cholesky decomposition of :math:`J_xJ_x^T` where :math:`J_x`
            is the Jacobian of the constraint evaluated at `x_base`.
        log_abs_det: log of :math:`|(J_xJ_x^T)|^{-1/2}` computed using the
            Cholesky decomposition.
    """
    is_satisfied: ArrayLike
    x_base: ArrayLike
    chol: jax.Array
    log_abs_det: jax.Array

def make_constraint_state(
        constraint_fn: Callable[[ArrayLike], jax.Array],
        x_base: ArrayLike,
        tol: ArrayLike
    ) -> ConstraintState:
    """
    Build :class:`ConstraintState`.

    :param constraint_fn: the constraint function.
    :param x_base: base point; see :class:`ConstraintState`.
    :param tol: tolerance for constraint satisfiability.
    :return: a :class:`ConstraintState` corresponding to `x_base`.
    """
    J, f_val = jax.jacrev(lambda x: 2*(constraint_fn(x),),has_aux=True)(x_base) # get Jacobian and value in one pass
    is_satisfied = utils.newton_fn_value_err(f_val) < tol
    m, n = jnp.shape(J)
    assert m < n
    chol = jax.lax.linalg.cholesky(jnp.inner(J,J), symmetrize_input=False) # (JJ^T)^T=JJ^T -> no need to force it
    log_abs_det = -jnp.log(jnp.abs(jnp.diag(chol))).sum()
    return ConstraintState(is_satisfied, x_base, chol, log_abs_det)

# project onto normal (N) space
#   P[N]v = J^T(JJ^T)^{-1}Jv
# Following Xu&Holmes-Cerfon(2024), we can do this in steps
#   1) u = Jv --> single jvp, O(1 func eval). For norm-like f, this is O(nm)
#   2) solve for w: LL^Tw=u <=> w = L^{-T}[L^{-1}u]
#                           <=> t=L^{-1}u and w=L^{-T}t
#       a) O(m^2) triangular solve for t: Lt = u
#       b) O(m^2) triangular solve for w: L^Tw=t
#   3) P[N]v = J^Tw = (w^TJ)^T --> single vjp, O(1 func eval)
# Cost: O(nm + m^2)
def proj_normal(
        constraint_fn: Callable[[ArrayLike], jax.Array],
        cs: ConstraintState,
        v: ArrayLike
    ) -> jax.Array:
    """
    Project a vector onto the normal space at `cs.x_base`.

    :param constraint_fn: the constraint function.
    :param cs: a :class:`ConstraintState` dictating the normal space.
    :param v: vector to project.
    :return: normal component of `v`.
    """
    u = jax.jvp(constraint_fn, (cs.x_base,), (v,))[-1]
    t = jax.lax.linalg.triangular_solve(cs.chol, u, lower=True) # == jnp.linalg.solve(cs.chol, u)
    w = jax.lax.linalg.triangular_solve(
        cs.chol, t, lower=True, transpose_a=True # == jnp.linalg.solve(cs.chol.T, t)
    )
    f_vjp = jax.vjp(constraint_fn, cs.x_base)[-1]
    return f_vjp(w)[0]

# project onto normal (N) and tangent (T) spaces
# same as cost of just doing the normal part
def proj_normal_tangent(
        constraint_fn: Callable[[ArrayLike], jax.Array],
        cs: ConstraintState,
        v: ArrayLike
    ) -> tuple[jax.Array, jax.Array]:
    """
    Project a vector onto the normal and tangent spaces at `cs.x_base`.

    :param constraint_fn: the constraint function.
    :param cs: a :class:`ConstraintState` dictating the normal space.
    :param v: vector to project.
    :return: normal and tangent components of `v`.
    """
    PNv = proj_normal(constraint_fn, cs, v)
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

    def proj_level_set(
            self,
            x_flat: ArrayLike,
            cs: ConstraintState
        ) -> tuple:
        """
        Project a point on the constraint level set along the normal space
        determined by a :class:`ConstraintState`.

        :param ArrayLike x_flat: a vector.
        :param ConstraintState cs: a :class:`ConstraintState` corresponding to
            the base point which dictates the normal spaces.
        :return tuple: projected vector, updated :class:`ConstraintState`, and
            solver diagnostics.
        """
        # project to the feasible set in the normal directions at cs.x_base
        #   solve for z: 0 = f(x + J^Tz) => set x' = x + J^Tz
        # since J^Tz = (z^TJ)^T, we can use vector-Jacobian prods (vjp)
        f_val, f_vjp = jax.vjp(self.constraint_fn, cs.x_base)
        z, *diagnostics, _ = utils.newton(
            lambda z: self.constraint_fn(x_flat + f_vjp(z)[0]),
            jnp.zeros_like(f_val),
            **self.solver_options
        )
        x_flat += f_vjp(z)[0]

        # update the constraint state and return
        cs = make_constraint_state(
            self.constraint_fn, x_flat, self.solver_options['tol']
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
            self.constraint_fn, x_flat, self.solver_options['tol']
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
                self.constraint_fn,
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
        v_flat = proj_normal_tangent(
            self.constraint_fn, cs, x_flat_new - x_flat
        )[-1] / step_size

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
