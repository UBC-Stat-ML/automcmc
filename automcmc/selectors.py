from abc import ABC, abstractmethod

import jax
from jax import random
from jax import lax
import jax.numpy as jnp

from automcmc import utils

def _draw_log_unif_bounds(rng_key):
    return lax.sort(random.exponential(rng_key, (2,)) * (-1))

class StepSizeSelector(ABC):
    """
    Abstract class for defining criteria for automatically selecting step sizes
    at each Markov step.

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
        :return: `True` if step size should grow; `False` otherwise.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def should_shrink(parameters, log_diff):
        """
        Decide if step size should shrink based on `parameters` and current log joint
        difference `log_diff`.

        :param parameters: selector parameters.
        :return: `True` if step size should shrink; `False` otherwise.
        """
        raise NotImplementedError
    
    @staticmethod
    def adapt_base_step_size(base_step_size, mean_step_size, n_samples_in_round):
        return mean_step_size
    

class AsymmetricSelector(StepSizeSelector):
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

def DeterministicAsymmetricSelector(p_lo=0.01, p_hi=0.99, *args, **kwargs):
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

class SymmetricSelector(StepSizeSelector):
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

def DeterministicSymmetricSelector(p_lo=0.01, p_hi=0.99, *args, **kwargs):
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

class FixedStepSizeSelector(StepSizeSelector):
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

###############################################################################
# executors
###############################################################################

def copy_state_extras(source, dest):
    return dest._replace(stats = source.stats, rng_key = source.rng_key)

DEBUG_ALTER_STEP_SIZE = None # anything other than None will print during step size loop

def gen_executor(kernel):
    return gen_target_acc_prob_executor(kernel)

#######################################
# acceptance probability bracketing
#######################################

def gen_alter_step_size_cond_fun(pred_fun, max_n_iter):
    def alter_step_size_cond_fun(args):
        (
            state, 
            exponent, 
            next_log_joint, 
            init_log_joint,
            selector_params, 
            precond_state
        ) = args

        # `numerically_safe_diff` is used to avoid corner cases where the step
        # size is 0 already but, because of extreme nonlinearities, the 
        # potential at this fictituous "next" point gives a log_joint that is
        # exactly equal to the next float of `init_log_joint`
        log_diff = utils.numerically_safe_diff(init_log_joint,next_log_joint)
        decision = jnp.logical_and(
            lax.abs(exponent) < max_n_iter,     # bail if max number of iterations reached
            pred_fun(selector_params, log_diff)
        )

        return decision
    return alter_step_size_cond_fun

def gen_alter_step_size_body_fun(kernel, direction):
    def alter_step_size_body_fun(args):
        (
            state, 
            exponent, 
            next_log_joint, 
            init_log_joint,
            selector_params, 
            precond_state
        ) = args
        exponent = exponent + direction
        eps = kernel.step_size(state.base_step_size, exponent)
        next_state = kernel.update_log_joint(
            kernel.involution_main(eps, state, precond_state),
            precond_state
        )
        next_log_joint = next_state.log_joint
        state = copy_state_extras(next_state, state)

        # maybe print debug info
        if DEBUG_ALTER_STEP_SIZE is not None:
            jax.debug.print(
                "dir: {d: d}: base: {bs:.8f} + exp: {e: d} = eps: {s:.8f} | (L0, L1, DL, NDL): ({l0: .2f},{l1: .2f},{dl: .2f},{ndl: .2f}) | bounds: ({a:.3f},{b:.3f})", 
                ordered=True,
                d=direction, 
                bs=state.base_step_size,
                e=exponent,
                s=eps,
                l0=init_log_joint,
                l1=next_log_joint,
                dl=next_log_joint-init_log_joint,
                ndl=utils.numerically_safe_diff(init_log_joint,next_log_joint),
                a=selector_params[0],
                b=selector_params[1]
            )
            # jax.debug.print(
            #     "{v} | {c} | {i}",
            #     v=precond_state.var,
            #     c=precond_state.var_tril_factor,
            #     i=precond_state.inv_var_triu_factor
            # )

        return (
            state, 
            exponent, 
            next_log_joint, 
            init_log_joint,
            selector_params, 
            precond_state
        )

    return alter_step_size_body_fun

def gen_target_acc_prob_executor(kernel):
    """
    Generates a function that implements the logic for automatically selecting
    a step size by bracketing the acceptance probability, as described in
    Algorithm 2 of Biron-Lattes et al. (2024) and Liu et al. (2025).

    :param kernel: An instance of :class:`autostep.AutoStep`.
    :return: A function.
    """

    selector = kernel.selector
    shrink_step_size_cond_fun = gen_alter_step_size_cond_fun(
        selector.should_shrink, selector.max_n_iter
    )
    shrink_step_size_body_fun = gen_alter_step_size_body_fun(kernel, -1)
    grow_step_size_cond_fun = gen_alter_step_size_cond_fun(
        selector.should_grow, selector.max_n_iter
    )
    grow_step_size_body_fun = gen_alter_step_size_body_fun(kernel, 1)

    def shrink_step_size(
            state, 
            selector_params, 
            next_log_joint, 
            init_log_joint, 
            precond_state
        ):
        exponent = 0
        state, exponent, *_ = lax.while_loop(
            shrink_step_size_cond_fun,
            shrink_step_size_body_fun,
            (state, exponent, next_log_joint, init_log_joint, 
                selector_params, precond_state)
        )
        return state, exponent

    def grow_step_size(
            state, 
            selector_params, 
            next_log_joint, 
            init_log_joint, 
            precond_state
        ):
        exponent = 0        
        state, exponent, *_ = lax.while_loop(
            grow_step_size_cond_fun,
            grow_step_size_body_fun,
            (state, exponent, next_log_joint, init_log_joint, 
                selector_params, precond_state)
        )

        # deduct 1 step to avoid cliffs, but only if we actually entered the 
        # loop and didn't go over the max number of iterations
        exponent = jnp.where(
            jnp.logical_and(exponent > 0, exponent < selector.max_n_iter),
            exponent-1, 
            exponent
        )
        return state, exponent
    
    def auto_step_size_fn(state, selector_params, precond_state):
        init_log_joint = state.log_joint # Note: assumes the log joint value is up to date!
        next_state = kernel.update_log_joint(
            kernel.involution_main(state.base_step_size, state, precond_state),
            precond_state
        )
        next_log_joint = next_state.log_joint
        state = copy_state_extras(next_state, state) # update state's stats and rng_key

        # try shrinking (no-op if selector decides not to shrink)
        # note: we call the output of this `state` because it should equal the 
        # initial state except for extra fields -- stats, rng_key -- which we
        # want to update
        state, shrink_exponent = shrink_step_size(
            state, selector_params, next_log_joint, init_log_joint, precond_state
        )

        # try growing (no-op if selector decides not to grow)
        state, grow_exponent = grow_step_size(
            state, selector_params, next_log_joint, init_log_joint, precond_state
        )

        # can add the two since one of them must be zero
        return state, shrink_exponent + grow_exponent
    
    return auto_step_size_fn
