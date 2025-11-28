from abc import ABCMeta, abstractmethod

import jax
from jax.experimental import checkify
from jax import lax
from jax import numpy as jnp
from jax import random

from automcmc import automcmc
from automcmc import statistics
from automcmc import utils

class AutoStep(automcmc.AutoMCMC, metaclass=ABCMeta):
    """
    Defines the interface for AutoStep MCMC kernels as described in
    Liu et al. (2025).
    """

    def init_alter_step_size_loop_funs(self):
        self.shrink_step_size_cond_fun = utils.gen_alter_step_size_cond_fun(
            self.selector.should_shrink, self.selector.max_n_iter
        )
        self.shrink_step_size_body_fun = utils.gen_alter_step_size_body_fun(
            self, -1
        )
        self.grow_step_size_cond_fun = utils.gen_alter_step_size_cond_fun(
            self.selector.should_grow, self.selector.max_n_iter
        )
        self.grow_step_size_body_fun = utils.gen_alter_step_size_body_fun(
            self, 1
        )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_alter_step_size_loop_funs()

    def step_size(self, base_step_size, exponent):
        """
        Compute the step size associated with an exponent. Default implementation
        gives a base-2 exponential lattice.

        :param base_step_size: The within-round-fixed step size.
        :param exponent: Integer enumerating the lattice of step sizes.
        :return: Step size.
        """
        return base_step_size * (2.0 ** exponent)

    @abstractmethod
    def involution_main(self, step_size, state, precond_state):
        """
        Apply the main part of the involution. This is usually the part that 
        modifies the variables of interests.

        :param step_size: Step size to use in the involutive transformation.
        :param state: Current state.
        :param precond_state: Preconditioner state.
        :return: Updated state.
        """
        raise NotImplementedError
    
    @abstractmethod
    def involution_aux(self, step_size, state, precond_state):
        """
        Apply the auxiliary part of the involution. This is usually the part that
        is not necessary to implement for the respective involutive MCMC algorithm
        to work correctly (e.g., momentum flip in HMC).
        Note: it is assumed that the augmented target is invariant to this transformation.

        :param step_size: Step size to use in the involutive transformation.
        :param state: Current state.
        :param precond_state: Preconditioner state.
        :return: Updated state.
        """
        raise NotImplementedError
    
    def sample_single_chain(self, state, model_args, model_kwargs):
        # generate rng keys and store the updated master key in the state
        (
            rng_key, 
            precond_key, 
            aux_key,
            selector_key, 
            accept_key
        ) = random.split(state.rng_key, 5)
        state = state._replace(rng_key = rng_key)

        # build a (possibly randomized) preconditioner
        precond_state = self.preconditioner.maybe_alter_precond_state(
            state.base_precond_state, precond_key
        )
        # jax.debug.print(
        #     "precond_state: var={v}   chol_tril={c}   inv_fac={i}", 
        #     ordered=True, 
        #     v=precond_state.var,
        #     c=precond_state.var_tril_factor,
        #     i=precond_state.inv_var_triu_factor
        # )

        # refresh auxiliary variables (e.g., momentum), update the log joint 
        # density, and finally check if the latter is finite
        # Checker needs checkifying twice for some reason
        state = self.update_log_joint(
            self.refresh_aux_vars(aux_key, state, precond_state), precond_state
        )
        checkify.checkify(utils.checkified_is_finite)(state.log_joint)[0].throw()

        # draw selector parameters
        selector_params = self.selector.draw_parameters(selector_key)

        # forward step size search
        # jax.debug.print("fwd autostep: init_log_joint={i}, init_x={x}, init_v={v}", ordered=True, i=state.log_joint, x=state.x, v=state.p_flat)
        state, fwd_exponent = self.auto_step_size(
            state, selector_params, precond_state
        )
        fwd_step_size = self.step_size(state.base_step_size, fwd_exponent)
        proposed_state = self.update_log_joint(
            self.involution_main(fwd_step_size, state, precond_state),
            precond_state
        )
        # jax.debug.print(
        #     "fwd done: step_size={s}, init_log_joint={l}, next_log_joint={ln}, log_joint_diff={ld}, prop_x={x}, prop_v={v}",
        #     ordered=True, s=fwd_step_size, l=state.log_joint, ln=proposed_state.log_joint, 
        #     ld=proposed_state.log_joint-state.log_joint, x=proposed_state.x, v=proposed_state.p_flat)

        # backward step size search
        # don't recompute log_joint for flipped state because we assume inv_aux 
        # leaves it invariant
        prop_state_flip = self.involution_aux(
            fwd_step_size, proposed_state, precond_state
        )
        # jax.debug.print("bwd begin", ordered=True)
        prop_state_flip, bwd_exponent = self.auto_step_size(
            prop_state_flip, selector_params, precond_state
        )
        reversibility_passed = fwd_exponent == bwd_exponent
        
        # sanitize possible nan in proposed log joint, setting them to -inf
        # this may happen for some too large initial step sizes, and then 
        # `shrink_step_size` may fail to fix them before the max num of iters
        proposed_log_joint = jnp.where(
            jnp.isnan(proposed_state.log_joint),
            -jnp.inf,
            proposed_state.log_joint
        )

        # Metropolis-Hastings step
        # note: when the magnitude of log_joint is ~ 1e8, the difference in
        # Float32 precision of two floats next to each other can be >> 1.
        # For this reason, we consider 2 consecutive floats to be equal.
        log_joint_diff = utils.numerically_safe_diff(
            state.log_joint, proposed_log_joint
        )
        acc_prob = lax.clamp(
            0., reversibility_passed * lax.exp(log_joint_diff), 1.
        )
        # jax.debug.print(
        #     "bwd done: reversibility_passed={r}, acc_prob={a}", ordered=True,
        #     r=reversibility_passed, a=acc_prob)

        # build the next state depending on the MH outcome
        next_state = lax.cond(
            random.bernoulli(accept_key, acc_prob),
            utils.next_state_accepted,
            utils.next_state_rejected,
            (state, proposed_state, prop_state_flip)
        )

        # collect statistics
        bwd_step_size = self.step_size(state.base_step_size, bwd_exponent)
        avg_fwd_bwd_step_size = 0.5 * (fwd_step_size + bwd_step_size)
        new_stats = statistics.record_post_sample_stats(
            next_state.stats, avg_fwd_bwd_step_size, acc_prob, reversibility_passed,
            jax.flatten_util.ravel_pytree(getattr(next_state, self.sample_field))[0]
        )
        next_state = next_state._replace(stats = new_stats)

        # maybe adapt
        next_state = self.adapt(next_state)

        return next_state

    def auto_step_size(self, state, selector_params, precond_state):
        init_log_joint = state.log_joint # Note: assumes the log joint value is up to date!
        next_state = self.update_log_joint(
            self.involution_main(state.base_step_size, state, precond_state),
            precond_state
        )
        next_log_joint = next_state.log_joint
        state = utils.copy_state_extras(next_state, state) # update state's stats and rng_key

        # try shrinking (no-op if selector decides not to shrink)
        # note: we call the output of this `state` because it should equal the 
        # initial state except for extra fields -- stats, rng_key -- which we
        # want to update
        state, shrink_exponent = self.shrink_step_size(
            state, selector_params, next_log_joint, init_log_joint, precond_state
        )

        # try growing (no-op if selector decides not to grow)
        state, grow_exponent = self.grow_step_size(
            state, selector_params, next_log_joint, init_log_joint, precond_state
        )

        # check only one route was taken
        # Needs checkifying twice for some reason
        checkify.checkify(utils.checkified_is_zero)(
            shrink_exponent * grow_exponent
        )[0].throw()

        return state, shrink_exponent + grow_exponent
    
    def shrink_step_size(
            self, 
            state, 
            selector_params, 
            next_log_joint, 
            init_log_joint, 
            precond_state
        ):
        exponent = 0
        state, exponent, *_ = lax.while_loop(
            self.shrink_step_size_cond_fun,
            self.shrink_step_size_body_fun,
            (state, exponent, next_log_joint, init_log_joint, 
             selector_params, precond_state)
        )
        return state, exponent
    
    def grow_step_size(self, state, selector_params, next_log_joint, init_log_joint, precond_state):
        exponent = 0        
        state, exponent, *_ = lax.while_loop(
            self.grow_step_size_cond_fun,
            self.grow_step_size_body_fun,
            (state, exponent, next_log_joint, init_log_joint, 
             selector_params, precond_state)
        )

        # deduct 1 step to avoid cliffs, but only if we actually entered the 
        # loop and didn't go over the max number of iterations
        exponent = jnp.where(
            jnp.logical_and(exponent > 0, exponent < self.selector.max_n_iter),
            exponent-1, 
            exponent
        )
        return state, exponent

