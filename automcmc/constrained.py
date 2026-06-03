from abc import ABCMeta, abstractmethod
from typing import NamedTuple, Callable, Any, Optional

import jax
from jax.typing import ArrayLike
from jax import flatten_util, random, numpy as jnp

from automcmc import utils, autostep, preconditioning, optimization
from automcmc.automcmc import AutoMCMCState
from automcmc.preconditioning import (
    Preconditioner,
    IdentityDiagonalPreconditioner
)

DEBUG_CONSTRAINED_SAMPLING = False

###############################################################################
# Classes that handle the projection onto the normal and tangent spaces
###############################################################################

class LevelSetState(NamedTuple):
    """
    A :class:`NamedTuple` describing the forward model around a base point, and
    its agreement with an observed model output.

    Attributes:
        x_base: point that dictates the level set :math:`f^{-1}(\\{f(x)\\})`
            We assume `x_base` is one-dimensional (i.e., a vector).
        obs_output: an output of the forward model :math:`y`, dictating the
            level set :math:`f^{-1}(y)`.
        is_satisfied: true if `f(x_base)=y` (up to tolerance); i.e., if the two
            level sets described above are the same.
        log_abs_det: logarithm of :math:`|J_xJ_x^T|^{-1/2}` at `x_base`.
        mat: a matrix used for projection onto the constraint set. Depends on
            the handler used.
    """
    x_base: jax.Array
    obs_output: jax.Array
    is_satisfied: ArrayLike
    log_abs_det: jax.Array
    mat: jax.Array


class LevelSetHandler(metaclass=ABCMeta):

    @staticmethod
    def get_jacobian_and_check_output(fwd_model, obs_output, x_base, tol):
        # get Jacobian and value in one pass
        # use reverse-mode because we work in the m<n world
        J, f_val = jax.jacrev(
            lambda x: 2*(fwd_model(x),), has_aux=True
        )(x_base)
        is_satisfied = utils.newton_fn_value_err(f_val-obs_output) < tol
        m, n = jnp.shape(J)
        assert m < n
        return J, is_satisfied

    @abstractmethod
    def make_levelset_state(
        self,
        fwd_model: Callable[[ArrayLike], jax.Array],
        obs_output: ArrayLike,
        x_base: ArrayLike,
        tol: ArrayLike
    ) -> LevelSetState:
        """
        Build :class:`LevelSetState` from a forward model function and an
        observed output.

        :param fwd_model: the forward model.
        :param obs_output: an observed output of the forward model.
        :param x_base: base point; see :class:`LevelSetState`.
        :param tol: tolerance for constraint satisfiability.
        :return: a :class:`LevelSetState` corresponding to `x_base`.
        """
        pass

    @staticmethod
    @abstractmethod
    def proj_normal(
        fwd_model: Callable[[ArrayLike], jax.Array],
        lss: LevelSetState,
        v: ArrayLike
    ) -> jax.Array:
        """
        Project a vector onto the normal space dictated by a
        :class:`LevelSetState`.

        :param fwd_model: the forward model.
        :param lss: a :class:`LevelSetState` dictating the normal space.
        :param v: vector to project.
        :return: normal component of `v`.
        """
        pass

    # project onto normal (N) and tangent (T) spaces
    # same as cost of just doing the normal part
    def proj_normal_tangent(
            self,
            fwd_model: Callable[[ArrayLike], jax.Array],
            lss: LevelSetState,
            v: ArrayLike
        ) -> tuple[jax.Array, jax.Array]:
        """
        Project a vector onto the normal and tangent spaces dictated by a
        :class:`LevelSetState`.

        :param fwd_model: the forward model.
        :param lss: a :class:`LevelSetState`.
        :param v: vector to project.
        :return: normal and tangent components of `v`.
        """
        PNv = self.proj_normal(fwd_model, lss, v)
        return (PNv, v - PNv)


class LevelSetHandlerCholesky(LevelSetHandler):
    """
    Handles the projection onto the normal space using an approach based on
    `jvp`, `vjp`, and the Cholesky decomposition of :math:`J_xJ_x^T` [1].
    Its main benefit is that each operation has complexity at most linear
    in `n` while only using :math:`O(n+m^2)` memory, making it ideal for
    high-dimensional settings. However, the method exhibits low accuracy in the
    presence of high curvature.

    .. rubric:: References

    [1] Xu, K., & Holmes-Cerfon, M. (2024). Monte Carlo on manifolds in high
    dimensions. *Journal of Computational Physics*, 506, 112939.
    """
    def make_levelset_state(
            self,
            fwd_model: Callable[[ArrayLike], jax.Array],
            obs_output: ArrayLike,
            x_base: ArrayLike,
            tol: ArrayLike
        ) -> LevelSetState:
        J, is_satisfied = self.get_jacobian_and_check_output(
            fwd_model, obs_output, x_base, tol
        )
        chol = jax.lax.linalg.cholesky(jnp.inner(J,J))
        log_abs_det = -jnp.log(jnp.abs(jnp.diag(chol))).sum()
        return LevelSetState(
            x_base, obs_output, is_satisfied, log_abs_det, chol
        )

    # project onto normal (N) space
    #   P[N]v = J^T(JJ^T)^{-1}Jv
    # Following Xu&Holmes-Cerfon(2024), we can do this in steps
    #   1) u = Jv --> single jvp, O(1 func eval). For norm-like f, this is O(nm)
    #   2) solve for w: LL^Tw=u <=> w = L^{-T}[L^{-1}u]
    #                           <=> t=L^{-1}u and w=L^{-T}t
    #       a) O(m^2) triangular solve for t: Lt = u
    #       b) O(m^2) triangular solve for w: L^Tw=t
    #   3) P[N]v = J^Tw = (w^TJ)^T --> single vjp, O(1 func eval)
    # Cost: O(2nm + m^2)
    @staticmethod
    def proj_normal(
            fwd_model: Callable[[ArrayLike], jax.Array],
            lss: LevelSetState,
            v: ArrayLike
        ) -> jax.Array:
        chol = lss.mat
        u = jax.jvp(fwd_model, (lss.x_base,), (v,))[-1]
        t = jax.lax.linalg.triangular_solve(chol, u, lower=True) # == jnp.linalg.solve(chol, u)
        w = jax.lax.linalg.triangular_solve(
            chol, t, lower=True, transpose_a=True # == jnp.linalg.solve(chol.T, t)
        )
        f_vjp = jax.vjp(fwd_model, lss.x_base)[-1]
        return f_vjp(w)[0]


class LevelSetHandlerQR(LevelSetHandler):
    """
    Handles the projection onto the normal space using the QR decomposition
    of :math:`J_x^T` [1]. It is simple and remarkably accurate even in high
    curvature scenarios. However, its cost is :math:`O(nm^2+n^2m)` and uses
    space :math:`O(nm)`, making it expensive in the big `n` scenario.

    .. rubric:: References

    [1] Zappa, E., Holmes-Cerfon, M. & Goodman, J. (2018).
    Monte Carlo on manifolds: Sampling densities and integrating functions.
    *Comm. Pure Appl. Math.*, 71, 2609-2647.
    """
    def make_levelset_state(
            self,
            fwd_model: Callable[[ArrayLike], jax.Array],
            obs_output: ArrayLike,
            x_base: ArrayLike,
            tol: ArrayLike
        ) -> LevelSetState:
        J, is_satisfied = self.get_jacobian_and_check_output(
            fwd_model, obs_output, x_base, tol
        )
        Q,R = jnp.linalg.qr(J.T)
        log_abs_det = -jnp.log(jnp.abs(jnp.diag(R))).sum()
        return LevelSetState(
            x_base, obs_output, is_satisfied, log_abs_det, Q
        )

    # project onto normal (N) space
    #   P[N]v = J^T(JJ^T)^{-1}Jv
    # If J^T = QR, then JJ^T = R^TR and
    #   P[N]v = [QR(R^TR)^{-1}R^TQ^T]v = QQ^Tv
    # Cost: O(mn)
    @staticmethod
    def proj_normal(
            _: Callable[[ArrayLike], jax.Array],
            lss: LevelSetState,
            v: ArrayLike
        ) -> jax.Array:
        Q = lss.mat
        return Q @ jnp.dot(v,Q) # dot(v,Q) === Q.T@v


###############################################################################
# Sampler definition
###############################################################################

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
            preconditioner: Preconditioner = IdentityDiagonalPreconditioner(), # we need rotational symmetry
            constraint_fn: Optional[Callable[[ArrayLike], jax.Array]] = None,
            fwd_model: Optional[Callable[[ArrayLike], jax.Array]] = None,
            init_obs_output: Optional[ArrayLike] = None,
            levelset_handler: LevelSetHandler = LevelSetHandlerQR(),
            solver_options: dict = {},
            x_tols: dict = {},
            levelset_finder_settings: Optional[bool | dict] = None,
            **kwargs
        ) -> None:
        """Instantiate an :class:`AutoConstrainedRWMH` sampler.

        :param args: Passed to the :class:`AutoStep` constructor.
        :param Preconditioner preconditioner: only allows instances of class
            :class:`IdentityDiagonalPreconditioner`.
        :param Optional[Callable[[ArrayLike], jax.Array]] constraint_fn: user
            provided function to constrain the sampler to the function's zero
            levelset. This is equivalent to providing the same function as
            `fwd_model` and additionally passing `init_obs_output=0`.
        :param Optional[Callable[[ArrayLike], jax.Array]] fwd_model: constrain
            the sampler to a levelset of this function, parametrized via the
            `init_obs_output` argument.
        :param Optional[ArrayLike] init_obs_output: output of `fwd_model` on
            whose levelset we wish to constain the sampler. Under NumPyro's
            vectorized sampling, it is possible to batch `num_chains` different
            output values in order to simultaneously sample multiple levelsets.
        :param LevelSetHandler levelset_handler: defines the method to project
            velocities to tangent spaces along the level set. Defaults to the
            QR-based approach described in [1] (:class:`LevelSetHandlerQR`). In
            high-dimensional settings, it may be beneficial to switch to the
            approach suggested in [2], based on the Cholesky decomposition
            (:class:`LevelSetHandlerCholesky`), at the cost of less robustness
            to high curvature models.
        :param dict solver_options: passed to :func:`utils.newton`. Default
            values are established based on the float type of the arguments and
            the dimension.
        :param dict x_tols: optional `dict` with `atol` and `rtol` values used
            to detect difference in ambient space during reversibility check.
            Default values are established based on the float type of the
            arguments and the dimension.
        :param Optional[bool | dict] levelset_finder_settings: if truthy,
            an initial robust gradient descent (NADAMW) phase is employed to
            find the desired level set. This is useful when the initial
            parameters are far from the level set, since the Newton solver
            struggles in this setting. Defaults to `None`; i.e., not used.
            For additional control, the user may pass a `dict` with the same
            structure as
            `automcmc.optimization.DEFAULT_OPTIMIZE_FUN_SETTINGS['NADAMW']`.
        :param kwargs: Passed to the :class:`AutoStep` constructor.


        .. rubric:: References

        [1] Zappa, E., Holmes-Cerfon, M. & Goodman, J. (2018).
        Monte Carlo on manifolds: Sampling densities and integrating functions.
        *Comm. Pure Appl. Math.*, 71, 2609-2647.

        [2] Xu, K., & Holmes-Cerfon, M. (2024). Monte Carlo on manifolds in high
        dimensions. *Journal of Computational Physics*, 506, 112939.
        """
        super().__init__(*args,preconditioner=preconditioner,**kwargs)
        assert isinstance(
            self.preconditioner, IdentityDiagonalPreconditioner
        ), "This method is justified only for `IdentityDiagonalPreconditioner`" \
          f" but I got instead {type(self.preconditioner)}"

        if fwd_model is None:
            assert init_obs_output is None or jnp.allclose(
                init_obs_output, jnp.zeros_like(init_obs_output)
            ), "You supplied a non-zero `init_obs_output` but no `fwd_model`"
            if constraint_fn is not None:
                fwd_model = constraint_fn
            else:
                raise ValueError(
                    "You must supply one of `fwd_model` or `constraint_fn`"
                )

        self.fwd_model = fwd_model
        self.init_obs_output = 0 if init_obs_output is None else init_obs_output
        self.levelset_handler = levelset_handler
        self.solver_options = solver_options
        self.x_tols = x_tols
        if (
            levelset_finder_settings and
            (not isinstance(levelset_finder_settings, dict))
            ):
            # shallow copy is enough since we only change `n_iter`
            levelset_finder_settings = (
                optimization.DEFAULT_OPTIMIZE_FUN_SETTINGS['NADAMW'].copy()
            )
            levelset_finder_settings['n_iter'] = 256

        self.levelset_finder_settings = levelset_finder_settings

    def proj_level_set(
            self,
            x_flat: ArrayLike,
            lss: LevelSetState
        ) -> tuple[Any, ...]:
        """
        Project a point on the level set determined by a
        :class:`LevelSetState`.

        :param ArrayLike x_flat: a vector.
        :param LevelSetState lss: determines the level set that is being
            targeted.
        :return tuple[Any, ...]: projected vector, updated
            :class:`LevelSetState`, and solver diagnostics.
        """
        # project to the feasible set in the normal directions at x_base
        #   solve for z: y = f(x + J^Tz) => set x' = x + J^Tz
        # since J^Tz = (z^TJ)^T, we can use vector-Jacobian prods (vjp)
        f_val, f_vjp = jax.vjp(self.fwd_model, lss.x_base) # Jac[fwd-mod] == Jac[constr]
        z, *diagnostics, _ = utils.newton(
            lambda z: self.fwd_model(x_flat + f_vjp(z)[0]) - lss.obs_output,
            jnp.zeros_like(f_val),
            **self.solver_options
        )
        x_flat += f_vjp(z)[0]

        # update the constraint state and return
        lss = self.levelset_handler.make_levelset_state(
            self.fwd_model,
            lss.obs_output,
            x_flat,
            self.solver_options['tol']
        )
        return (x_flat, lss, diagnostics)

    # preliminary pass with gradient descent on least-squares loss to get close to
    # levelset, as Newton fails when initialized too far away
    def find_levelset(self, x_flat, init_obs_output):
        if not isinstance(self.levelset_finder_settings, dict):
            return x_flat
        n_iter = self.levelset_finder_settings['n_iter']
        target_fun = lambda x: jnp.square(
            self.fwd_model(x) - init_obs_output
        ).sum()
        solver, step_fn = optimization.make_nadamw_solver(
            target_fun, self.levelset_finder_settings['solver_params'], False
        )
        opt_state = solver.init(x_flat)
        return jax.lax.scan(
            lambda t,_: (step_fn(t[0], t[1])[:2], None),
            (x_flat, opt_state),
            length=n_iter
        )[0][0]

    def init_lss_and_project(self, x, init_obs_output):
        # Home in on the levelset using gradient descent (currently NADAMW)
        # This is required because Newton's method fails when started too far
        # from the levelset
        x_flat, unravel_fn = flatten_util.ravel_pytree(x)
        x_flat = self.find_levelset(x_flat, init_obs_output)

        # set the constraint tolerance
        # Note: even though this function may run inside vmap, tolerances are
        # always singletons. Moreover, they don't depend on the value of `x`,
        # just their shape. Furthermore, vmap is smart and reinterprets `.shape`
        # (and `.size`, etc) so as to return the un-vmapped value. Hence, tols
        # are the same scalars regardless of vectorization
        if 'tol' not in self.solver_options:
            self.solver_options['tol'] = utils.newton_default_tol(x_flat)

        # set the ambient space tolerances (`allclose` parameters)
        # prefer using only rtol, but need to have atol>0 if the origin
        # satisfies the constraint
        if 'rtol' not in self.x_tols:
            self.x_tols['rtol'] = jnp.finfo(x_flat.dtype).eps**0.25
        if 'atol' not in self.x_tols:
            self.x_tols['atol'] = self.solver_options['tol'] # this val already scales with size of problem so no need to scale again

        # initialize the forward model state
        lss = self.levelset_handler.make_levelset_state(
            self.fwd_model,
            init_obs_output,
            x_flat,
            self.solver_options['tol']
        )

        # project to zero level set
        x_flat, lss, diagnostics = self.proj_level_set(x_flat, lss)
        return unravel_fn(x_flat), lss, diagnostics


    # TODO: extract `fwd_model` for numpyro models (via a deterministic
    # stmtnt+model_kwargs or a Delta dist (problem: breaks log_prob))
    def init_extras(self, state):
        rng_key_shape = jnp.shape(state.rng_key)
        if rng_key_shape == ():
            x, lss, diagnostics = self.init_lss_and_project(
                state.x, self.init_obs_output
            )
        else:
            assert (
                jnp.ndim(self.init_obs_output) > 0 and
                rng_key_shape[0] == self.init_obs_output.shape[0]
            ), "`init_obs_output` must have a leading dimension equal to " \
               f"the number of chains requested ({rng_key_shape[0]})"
            x, lss, diagnostics = jax.vmap(self.init_lss_and_project)(
                state.x, self.init_obs_output
            )

        assert jnp.all(lss.is_satisfied), \
            f"Cannot find feasible starting point. Solver output:\n{diagnostics}"

        # return updated state
        return state._replace(x = x, idiosyncratic = lss)

    def kinetic_energy(self, state, precond_state):
        return 0.5*jnp.dot(state.p_flat, state.p_flat)

    # sample and project to tangent space
    def refresh_aux_vars(self, rng_key, state, precond_state):
        return state._replace(
            p_flat = self.levelset_handler.proj_normal_tangent(
                self.fwd_model, # Jac[fwd-mod] == Jac[constraint]
                state.idiosyncratic,
                random.normal(rng_key, jnp.shape(state.p_flat))
            )[-1]
        )

    # need to
    #   - add the log of the co-area formula factor
    #   - make the likelihood 0 when constraint is not satisfied, so that
    #     solver failures are handled gracefully by the autostep executors
    def postprocess_logprior_and_loglik(self, state, log_prior, log_lik):
        lss = state.idiosyncratic
        log_prior += lss.log_abs_det
        log_lik = jnp.where(lss.is_satisfied, log_lik, -jnp.inf)
        return log_prior, log_lik

    # check closeness of ambient space vectors by using a symmetric check
    # (i.e., invariant to flipping x<->y) based on L^2-norms
    # better than jnp.allclose which is == all(jnp.isclose)
    def close_in_ambient_space(self, x, y):
        return utils.close_in_norm(
            x, y, self.x_tols['rtol'], self.x_tols['atol']
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
        DEBUG_CONSTRAINED_SAMPLING and jax.debug.print(
            "exponents match: {}", passed, ordered=True
        )
        passed = jnp.logical_and(
            passed, proposed_state.idiosyncratic.is_satisfied
        )
        DEBUG_CONSTRAINED_SAMPLING and jax.debug.print(
            "and prop state is feasible: {}", passed, ordered=True
        )
        passed = jnp.logical_and(
            passed, roundtrip_state.idiosyncratic.is_satisfied
        )
        DEBUG_CONSTRAINED_SAMPLING and jax.debug.print(
            "and rt state is feasible: {}", passed, ordered=True
        )
        passed = jnp.logical_and(
            passed,
            self.close_in_ambient_space(
                flatten_util.ravel_pytree(initial_state.x)[0],
                flatten_util.ravel_pytree(roundtrip_state.x)[0]
            )
        )
        DEBUG_CONSTRAINED_SAMPLING and jax.debug.print(
            "and rt state is close to init: {}", passed, ordered=True
        )
        passed = jnp.logical_and(
            passed,
            self.close_in_ambient_space(
                initial_state.p_flat, roundtrip_state.p_flat
            )
        )
        DEBUG_CONSTRAINED_SAMPLING and jax.debug.print(
            "and rt vel is close to init: {}", passed, ordered=True
        )
        DEBUG_CONSTRAINED_SAMPLING and jax.debug.print(
            "\ninit_st={}\n\nprop_st={}\n\nrt_st={}",
            initial_state, proposed_state, roundtrip_state,
            ordered=True
        )
        return passed


    # Both position and velocity are affected by this transform
    #   - position is updated by moving along velocity and projecting
    #   - velocity is updated by projecting the displacement vector
    def involution_main(self, step_size, state, precond_state):
        # move outside the level set: x <- x+s*v
        x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
        v_flat = state.p_flat
        x_flat_out = x_flat + step_size * v_flat

        # project back to level set
        x_flat_new, lss, _ = self.proj_level_set(
            x_flat_out, state.idiosyncratic
        )

        # transport the velocity by projecting the displacement vector x'-x
        # onto the tangent space at the new point
        # note: the displacement is on scale step_size * v, so we must divide
        # by the step size to get the right scale
        v_flat = self.levelset_handler.proj_normal_tangent(
            self.fwd_model, # Jac[fwd-mod] == Jac[constraint]
            lss,
            x_flat_new - x_flat
        )[-1] / step_size

        # update state and return
        return state._replace(
            x = unravel_fn(x_flat_new),
            p_flat = v_flat,
            idiosyncratic = lss
        )

    # only need to flip sign of the velocity, which was already transported
    # to the new tangent space in `involution_main`
    def involution_aux(self, state):
        return state._replace(p_flat = -state.p_flat)
