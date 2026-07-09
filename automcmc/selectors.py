from abc import ABC, ABCMeta, abstractmethod

import jax
from jax import random
from jax import lax
import jax.numpy as jnp

from automcmc import utils

DEBUG_EXECUTOR = False

def _draw_log_unif_bounds(rng_key):
    return lax.sort(random.exponential(rng_key, (2,)) * (-1))

class StepSizeSelector(ABC):
    """
    Abstract class for defining criteria for automatically selecting step sizes
    at each Markov step.
    """

    def draw_parameters(self, rng_key):
        """
        Draw the random parameters used (if any) by the selector. By default
        this returns `None`.

        :param rng_key: Random number generator key.
        :return: Random instance of the parameters used by the selector.
        """
        return None

    @staticmethod
    def adapt_base_step_size(base_step_size, mean_step_size, n_samples_in_round):
        """
        Criterion for adapting the base step size after a round of sampling.

        :param base_step_size: Previous base step size.
        :param mean_step_size: Average step size used in the round.
        :param n_samples_in_round: Length of the round.
        :return: An updated base step size.
        """
        return mean_step_size

#######################################
# acceptance probability bracketing
#######################################

class AcceptProbBracketingSelector(StepSizeSelector, metaclass=ABCMeta):
    """
    Class of selectors that adjust the step size by bracketing the acceptance
    probability in a possibly randomized interval (Biron-Lattes et al. 2024,
    Liu et al. 2025)

    :param max_n_iter: Maximum number of step size doubling/halvings.
    :param bounds_sampler: A function that takes a PRNG key and samples a pair
        of endpoints used in the step-size selection loop. Defaults to ordered
        log-uniform random variables.
    """

    def __init__(
            self,
            max_n_iter=14, # 2**14~1.6e4 => step size changes +/- 4 orders of mag
            bounds_sampler=_draw_log_unif_bounds
        ):
        self.max_n_iter = max_n_iter
        self.bounds_sampler = bounds_sampler

    def draw_parameters(self, rng_key):
        """
        Draw the random parameters used (if any) by the selector.

        :param rng_key: Random number generator key.
        :return: Random instance of the parameters used by the selector.
        """
        return self.bounds_sampler(rng_key)

    @staticmethod
    @abstractmethod
    def should_grow(parameters, log_diff):
        """
        Decide if step size should grow based on `parameters` and current log joint
        difference `log_diff`.

        :param parameters: selector parameters.
        :param log_diff: log joint difference.
        :return: `True` if step size should grow; `False` otherwise.
        """
        pass

    @staticmethod
    @abstractmethod
    def should_shrink(parameters, log_diff):
        """
        Decide if step size should shrink based on `parameters` and current log joint
        difference `log_diff`.

        :param parameters: selector parameters.
        :param log_diff: log joint difference.
        :return: `True` if step size should shrink; `False` otherwise.
        """
        pass

    def direction(self, parameters, log_diff):
        """
        Compute direction of movement for the step size exponent.

        :param parameters: selector parameters.
        :param log_diff: log joint difference.
        :return: `+1` for growing, `-1` for shrinking, and `0` otherwise.
        """
        return (
            1*self.should_grow(parameters, log_diff) -
            1*self.should_shrink(parameters, log_diff)
        )

    def gen_executor(self, kernel):
        """
        Generates a function that implements the logic for automatically
        selecting a step size by bracketing the acceptance probability, as
        described in Algorithm 2 of Biron-Lattes et al. (2024) and
        Liu et al. (2025).

        :param kernel: An instance of :class:`autostep.AutoStep`.
        :return: A function.
        """
        assert self is kernel.selector # sanity check

        def loop_body(carry):
            (
                # the following three dont change inside the loop
                state,
                selector_params,
                precond_state,
                # these do change
                exponent,
                direction,
                _, # keep_going
            ) = carry

            # update exponent and step size
            # note: in the first iteration exponent=direction=0, so this yields
            # the base step size
            exponent += direction
            step_size = kernel.step_size(state.base_step_size, exponent)

            # involution + update log joint density
            next_state = kernel.update_log_joint(
                kernel.involution(step_size, state, precond_state),
                precond_state
            )

            # roundtrip involution (no need to update log joint)
            # this is a no-op if sampler uses an involution that is guaranteed
            # to succeed
            roundtrip_state = kernel.maybe_get_roundtrip_state(
                step_size, next_state, precond_state
            )

            # check reversibility of the underlying involution
            # if it failed, set next log joint diff to -inf
            reversibility_passed = kernel.reversibility_check(
                state,
                next_state,
                roundtrip_state
            )
            next_log_joint_diff = jnp.where(
                reversibility_passed,
                # `numerically_safe_diff` is used to avoid corner cases where
                # the step size is 0 already but, because of floating point
                # arithmetic, the log joint at this fictituous "next" point is
                # exactly equal to the next float of `init_log_joint`
                utils.numerically_safe_diff(
                    state.log_joint, next_state.log_joint
                ),
                -jnp.inf
            )

            # compute new direction of movement for the exponent
            # set actual direction only in first pass
            new_direction = self.direction(selector_params, next_log_joint_diff)
            direction = jnp.where(exponent==0, new_direction, direction)

            # check termination
            keep_going = jnp.logical_and(
                # still within budget of iterations
                jnp.abs(exponent) < self.max_n_iter,
                jnp.logical_and(
                    # we decided to change the exponent in the first iteration
                    jnp.logical_or(exponent > 0, jnp.abs(direction) > 0),
                    # we can only keep going in the original direction
                    jnp.logical_or(exponent == 0, new_direction == direction)
                )
            )

            # maybe print debug info
            if DEBUG_EXECUTOR:
                jax.debug.print(
                    "dir: {d: d}: base: {bs:.2e} + exp: {e:>2} = ss: {s:.2e} "
                    "| (L0, L1, DL, NDL): ({l0: .2e},{l1: .2e},{dl: .2e},"
                    "{ndl: .2e}) | bounds: ({a: .2e},{b: .2e})",
                    ordered=True,
                    d=direction,
                    bs=state.base_step_size,
                    e=exponent,
                    s=step_size,
                    l0=state.log_joint,
                    l1=next_state.log_joint,
                    dl=next_state.log_joint-state.log_joint,
                    ndl=next_log_joint_diff,
                    a=selector_params[0],
                    b=selector_params[1]
                )

            # return updated carry
            return (
                state,
                selector_params,
                precond_state,
                exponent,
                direction,
                keep_going
            )

        # TODO: can we spit out `next_state` too so that we don't have to
        # recompute it in a way that doesn't involve too much memory? We would
        # need to handle the fact that growing requires backtracking, but not
        # shrinking.
        # TODO: right now, `state` is never updated because we don't have stats
        # that change during the loop. If we reactivate the counter for target
        # evaluations, this needs to change
        def auto_step_size_fn(state, selector_params, precond_state):
            # run the loop, with first iteration corresponding to exponent=0
            # (i.e., base step size), where the direction of improvement is
            # decided (+1,-1).
            exponent = jax.lax.while_loop(
                lambda carry: carry[-1], # itemgetter(-1) does not work (`TypeError: cannot create weak reference to 'operator.itemgetter' object`),
                loop_body,
                (state, selector_params, precond_state, 0, 0, True)
            )[3]

            # deduct 1 step to avoid cliffs when increasing exponent, but only
            # if we didn't go over the max number of iterations (so that no
            # cliff was actually seen)
            # Note: this adjustment is necessary not only for the correctness
            # of AsymmetricSelectors, but also because more complicated
            # involutions (HMC with many steps, constrained mcmc) actually face
            # cliffs (think divergences with HMC)
            exponent = jnp.where(
                jnp.logical_and(exponent > 0, exponent < self.max_n_iter),
                exponent-1,
                exponent
            )

            return state, exponent

        return auto_step_size_fn


class AsymmetricSelector(AcceptProbBracketingSelector):
    """
    Asymmetric selector.
    """

    @staticmethod
    def should_grow(bounds, log_diff):
        return log_diff > bounds[1]

    @staticmethod
    def should_shrink(bounds, log_diff):
        return jnp.logical_or(
            jnp.logical_not(lax.is_finite(log_diff)),
            log_diff < bounds[0]
        )

def make_deterministic_bounds_sampler(p_lo, p_hi):
    assert p_lo < p_hi and 0 < p_lo and p_hi <= 1
    fixed_bounds = jnp.log(jnp.array([p_lo, p_hi]))
    return (lambda _: fixed_bounds)

def DeterministicAsymmetricSelector(p_lo=0.0001, p_hi=0.9999, *args, **kwargs):
    """
    Asymmetric selector with fixed deterministic endpoints.

    :param p_lo: Left endpoint in [0,1].
    :param p_hi: Right endpoint in [0,1].
    :param *args: Additional arguments for `AsymmetricSelector`.
    :param **kwargs: Additional keyword arguments for `AsymmetricSelector`.
    """
    return AsymmetricSelector(
        *args,
        bounds_sampler = make_deterministic_bounds_sampler(p_lo, p_hi),
        **kwargs
    )

class SymmetricSelector(AcceptProbBracketingSelector):
    """
    Symmetric selector.
    """

    @staticmethod
    def should_grow(bounds, log_diff):
        return lax.abs(log_diff) + bounds[1] < 0

    @staticmethod
    def should_shrink(bounds, log_diff):
        return jnp.logical_or(
            jnp.logical_not(lax.is_finite(log_diff)),
            lax.abs(log_diff) + bounds[0] > 0
        )

def DeterministicSymmetricSelector(p_lo=0.0001, p_hi=0.9999, *args, **kwargs):
    """
    Symmetric selector with fixed deterministic endpoints.

    :param p_lo: Left endpoint in [0,1].
    :param p_hi: Right endpoint in [0,1].
    :param *args: Additional arguments for `SymmetricSelector`.
    :param **kwargs: Additional keyword arguments for `SymmetricSelector`.
    """
    return SymmetricSelector(
        *args,
        bounds_sampler = make_deterministic_bounds_sampler(p_lo, p_hi),
        **kwargs
    )

class FixedStepSizeSelector(AcceptProbBracketingSelector):
    """
    A dummy selector that never adjusts the step size.
    """
    def __init__(self):
        super().__init__(
            max_n_iter = 0,
            bounds_sampler = make_deterministic_bounds_sampler(0.4, 0.6) # bounds are irrelevant
        )

    @staticmethod
    def should_grow(bounds, log_diff):
        return False

    @staticmethod
    def should_shrink(bounds, log_diff):
        return False

    @staticmethod
    def adapt_base_step_size(base_step_size, mean_step_size, n_samples_in_round):
        return base_step_size


#######################################
# maximum expected jump distance
#######################################

class MaxEJDSelector(StepSizeSelector):

    # just discard any arguments passed to keep it compatible with others
    def __init__(self, *args, **kwargs):
        pass

    # TODO: try to simplify this code
    def gen_executor(self, kernel):

        def expected_jump_dist(eps, state, precond_state):
            # take the step
            next_state = kernel.update_log_joint(
                kernel.involution(eps, state, precond_state),
                precond_state
            )

            # compute the Mahalanobis distance using the given preconditioner
            x_flat = jax.flatten_util.ravel_pytree(state.x)[0]
            next_x_flat = jax.flatten_util.ravel_pytree(next_state.x)[0]
            dx = next_x_flat - x_flat
            U = precond_state.inv_var_triu_factor
            dx_std = U.T @ dx if jnp.ndim(U) == 2 else U * dx
            dist = jnp.linalg.norm(dx_std)

            # compute min(fwd,bwd) acc prob and return the worst-case EJD
            log_joint_diff = next_state.log_joint-state.log_joint
            min_acc_prob = jnp.exp(-jnp.abs(log_joint_diff))
            return state, min_acc_prob*dist

        def gen_optim_ejd_funcs(kernel, direction):
            assert direction == 1 or direction == -1
            def cond_fn(carry):
                e, old_ejd, new_ejd, *_ = carry
                return jnp.logical_and(jnp.sign(e) == direction, new_ejd>old_ejd)

            def body_fn(carry):
                e, _, new_ejd, state, precond_state = carry
                e = e + direction
                old_ejd = new_ejd
                eps = kernel.step_size(state.base_step_size, e)
                state, new_ejd = expected_jump_dist(eps, state, precond_state)
                if DEBUG_EXECUTOR:
                    jax.debug.print(
                        "e={}, old_ejd={}, new_ejd={}", e, old_ejd, new_ejd, ordered=True
                    )
                return (e, old_ejd, new_ejd, state, precond_state)

            return cond_fn, body_fn

        inc_cond_fn, inc_body_fn = gen_optim_ejd_funcs(kernel, 1)
        dec_cond_fn, dec_body_fn = gen_optim_ejd_funcs(kernel, -1)

        def auto_step_size_fn(state, _, precond_state):
            # check EJD for staying put, doubling, and halving
            base_eps = kernel.step_size(state.base_step_size, 0)
            state, base_eps_ejd = expected_jump_dist(base_eps, state, precond_state)
            inc_eps = kernel.step_size(state.base_step_size, 1)
            state, inc_eps_ejd = expected_jump_dist(inc_eps, state, precond_state)
            dec_eps = kernel.step_size(state.base_step_size, -1)
            state, dec_eps_ejd = expected_jump_dist(dec_eps, state, precond_state)

            # check which direction gives the best improvement
            # Note: KEY IMPLEMENTATION DETAIL
            # argmax defaults to first elem when they are all equal. In particular,
            # when all(all_ejd==0), the decrease direction is selected. This is
            # intentional! Doing this allows the algorithm to decrease the step
            # size since it is clearly too agressive.
            all_ejd = jnp.array([dec_eps_ejd,base_eps_ejd,inc_eps_ejd])
            imax = all_ejd.argmax()
            new_ejd = all_ejd[imax]
            exponent = jnp.array([-1,0,1])[imax]
            if DEBUG_EXECUTOR:
                jax.debug.print(
                    "init: base_eps_ejd={}, inc_eps_ejd={}, dec_eps_ejd={}"
                    " -> e={}", base_eps_ejd, inc_eps_ejd, dec_eps_ejd, exponent,
                    ordered=True
                )

            # maybe greedily increase if doubling gave the highest EJD
            # at the end, undo last doubling that caused a marginal decrease in EJD
            exponent, _, new_ejd, state, precond_state = jax.lax.while_loop(
                inc_cond_fn,
                inc_body_fn,
                (exponent, -1, new_ejd, state, precond_state)
            )
            exponent = jnp.where(exponent>0,exponent-1,exponent) # old_ejd < 0 so we always enter the loop if exponent>0

            # maybe greedily decrease if halving gave the highest EJD
            # at the end, undo last halving that caused a marginal decrease in EJD
            exponent, _, new_ejd, state, precond_state = jax.lax.while_loop(
                dec_cond_fn,
                dec_body_fn,
                (exponent, -1, new_ejd, state, precond_state) # old_ejd < 0 so we always enter the loop if exponent<0
            )
            exponent = jnp.where(exponent<0,exponent+1,exponent)

            if DEBUG_EXECUTOR:
                jax.debug.print("final: e={}", exponent, ordered=True)

            return state, exponent

        return auto_step_size_fn
