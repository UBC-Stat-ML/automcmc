from abc import ABCMeta, abstractmethod

import jax
from jax.experimental import checkify
from jax import lax
from jax import numpy as jnp
from jax import random

from automcmc import automcmc, selectors, statistics, utils

class AutoStep(automcmc.AutoMCMC, metaclass=ABCMeta):
    """
    Defines the interface for AutoStep MCMC kernels as described in
    Liu et al. (2025).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_step_size_fn = selectors.gen_executor(self)

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
    
    def auto_step_size(self, state, selector_params, precond_state):
        """
        Find an appropriate step size using the criterion defined by the 
        `selector`, and depending on the `state` and `selector parameters`.

        :param state: Current state.
        :param selector_params: The possibly randomized selector parameters of
            the current iteration.
        :param precond_state: Preconditioner state.
        :return: A step size.
        """
        return self._auto_step_size_fn(state, selector_params, precond_state)
    
    def sample_single_chain(self, state, model_args, model_kwargs):
        """
        Implements a single step of Algorithm 1 in Biron-Lattes et al. (2024).

        :param state: Current state.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: The updated state.
        """
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
        if selectors.DEBUG_ALTER_STEP_SIZE is not None:
            jax.debug.print("\nautostep forward:", ordered=True)
        state, fwd_exponent = self.auto_step_size(
            state, selector_params, precond_state
        )
        fwd_step_size = self.step_size(state.base_step_size, fwd_exponent)
        proposed_state = self.update_log_joint(
            self.involution_main(fwd_step_size, state, precond_state),
            precond_state
        )

        # backward step size search
        # don't recompute log_joint for flipped state because we assume inv_aux 
        # leaves it invariant
        prop_state_flip = self.involution_aux(
            fwd_step_size, proposed_state, precond_state
        )
        if selectors.DEBUG_ALTER_STEP_SIZE is not None:
            jax.debug.print("autostep backward:", ordered=True)
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
        if selectors.DEBUG_ALTER_STEP_SIZE is not None:
            jax.debug.print(
                "reversible? {}, acc_prob={}, fwd_step_size={}",
                reversibility_passed,
                acc_prob, 
                fwd_step_size,
                ordered=True
            )

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

    