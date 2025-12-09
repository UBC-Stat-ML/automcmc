from abc import ABCMeta
from functools import partial

import jax
from jax import flatten_util
from jax import numpy as jnp
from jax import lax
from jax import random

from automcmc import statistics
from automcmc.automcmc import AutoMCMC

class SliceSampler(AutoMCMC, metaclass=ABCMeta):
    """
    Interface for defining slice sampling algorithms that leverage the 
    univariate algorithm of Neal (2003). The `p_flat` component of the
    state dictates the direction along which the univariate sampler is
    applied. For example, by using one-hot vectors, we can implement
    either the deterministic or random scan versions. Similarly, by
    drawing arbitrary directions at random, we can implement a 
    Hit-and-Run slicer.
    """
    def __init__(
        self,
        *args,
        max_grow_steps = 20,
        init_window_size = 4.0, # min-cost-optimal for N(0,1)
        **kwargs
    ):
        """
        Initialize the slice sampler.

        :param args: Passed to the :class:`AutoMCMC` constructor.
        :param max_grow_steps: Maximum number of iterations of the stepping-out phase.
        :param init_window_size: Size of initial window. Internally we identify
            this parameter with the step size in the :class:`AutoMCMC` 
            interface.
        :param kwargs: Passed to the :class:`AutoMCMC` constructor.
        """
        super().__init__(*args, init_base_step_size=init_window_size, **kwargs)
        self.max_grow_steps = max_grow_steps
    
    @staticmethod
    def in_slice(state, target_log_joint):
        return state.log_joint > target_log_joint

    def init_endpoints(self, rng_key, state, precond_state):
        # randomly place a window of fixed width around the initial `x` in the
        # direction of `p_flat`
        width = state.base_step_size
        p_flat = state.p_flat
        x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
        u = random.uniform(rng_key)
        x_flat_bwd = x_flat - p_flat*width*u
        x_bwd = unravel_fn(x_flat_bwd)
        x_fwd = unravel_fn(x_flat_bwd + p_flat*width)
        
        # make new states, update their log joints, and return
        bwd_state = self.update_log_joint(
            state._replace(x=x_bwd), precond_state
        )
        fwd_state = self.update_log_joint(
            state._replace(x=x_fwd), precond_state
        )
        return bwd_state, fwd_state

    def grow_slice(
            self, 
            rng_key, 
            bwd_state, 
            fwd_state, 
            target_log_joint, 
            precond_state
        ):
        width = bwd_state.base_step_size
        max_grow_steps = self.max_grow_steps

        # randomly allocate the budget
        max_grow_steps_bwd = random.randint(rng_key, (), 0, max_grow_steps)
        max_grow_steps_fwd = max_grow_steps - max_grow_steps_bwd
        
        # loop condition
        def cond_fn(carry):
            state, budget = carry
            # jax.debug.print(
            #     "grow: lp={}, lpt={}",
            #     state.log_joint, target_log_joint,
            #     ordered=True
            # )
            return jnp.logical_and(
                budget>0, self.in_slice(state, target_log_joint)
            )
        
        # grow function
        def body_fn(dx, carry):
            state, budget = carry
            x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
            x_new = unravel_fn(x_flat + dx)
            state = self.update_log_joint(
                state._replace(x=x_new), precond_state
            )
            return (state, budget-1)

        # define increment
        fwd_dx = fwd_state.p_flat*width # p_flat should be the same on both states

        # grow bwd
        bwd_state, _ = lax.while_loop(
            cond_fn, 
            partial(body_fn, -fwd_dx),
            (bwd_state, max_grow_steps_bwd)
        )

        # grow fwd
        fwd_state, _ = lax.while_loop(
            cond_fn, 
            partial(body_fn, fwd_dx),
            (fwd_state, max_grow_steps_fwd)
        )

        return bwd_state, fwd_state
    
    # slice shrinking
    # recall that this algorithm repeatedly samples a point x in the 
    # segment [x_bwd, x_fwd] and updates these endpoints depending on the
    # location of the starting point x_0. Writing the endpoints as
    #    x_fwd = x_0 + s_f d => s_f = d^T(x_fwd-x_0)/|d|^2 = (d/|d|)^T[(x_fwd-x_0)/|d|]
    #    x_bwd = x_0 + s_b d => s_b = d^T(x_bwd-x_0)/|d|^2 = (d/|d|)^T[(x_bwd-x_0)/|d|]
    # any interior point x can be written as
    #    x = x_bwd + u (x_fwd-x_bwd)
    #      = x_0 + s_bd + u(s_f-s_b)d
    #      = x_0 + s(u)d
    # with
    #    s(u) := [s_b + u(s_f-s_b)]
    # so
    #    x = x_0 <=> s(u)=0
    #    s_f >= s(u) > 0 <=> x_0 in [x, x_fwd] 
    #                    => (s'_bwd, s'_fwd)=(s(u), s_fwd)
    #    s_b <= s(u) < 0 <=> x_0 in [x_bwd, x] 
    #                    => (s'_bwd, s'_fwd)=(s_bwd, s(u))
    # which are the update rules for the scalar steps
    def shrink_slice(
            self, 
            rng_key, 
            init_state, 
            bwd_state, 
            fwd_state, 
            target_log_joint, 
            precond_state
        ):
        # get scalar steps of the initial boundaries
        x_flat_bwd, unravel_fn = flatten_util.ravel_pytree(bwd_state.x)
        x_flat_fwd    = flatten_util.ravel_pytree(fwd_state.x)[0]
        x_flat        = flatten_util.ravel_pytree(init_state.x)[0]
        p_flat        = fwd_state.p_flat
        p_flat_norm   = jnp.linalg.norm(p_flat)
        unit_flat     = p_flat/p_flat_norm 
        init_step_bwd = jnp.dot(unit_flat, (x_flat_bwd-x_flat)/p_flat_norm)
        init_step_fwd = jnp.dot(unit_flat, (x_flat_fwd-x_flat)/p_flat_norm)

        # shrinking loop body
        def body_fn(carry):
            i, rng_key, step_fwd, step_bwd, new_state = carry
            rng_key, draw_key = random.split(rng_key)
            u = random.uniform(draw_key)
            step_new = step_bwd + u*(step_fwd-step_bwd)
            x_new = unravel_fn(x_flat + step_new*p_flat)
            new_state = self.update_log_joint(
                new_state._replace(x = x_new), precond_state
            )
            step_bwd = jnp.where(step_new<0, step_new, step_bwd)
            step_fwd = jnp.where(step_new>0, step_new, step_fwd)
            return (i+1, rng_key, step_fwd, step_bwd, new_state)

        # shrinking loop cond
        def cond_fn(carry):
            i, _, step_fwd, step_bwd, new_state = carry
            # jax.debug.print(
            #     "shrink: [s_b, s_f]=[{}, {}], lp={}, lpt={}",
            #     step_bwd, step_fwd, new_state.log_joint, target_log_joint,
            #     ordered=True
            # )
            eps = jnp.sqrt(jnp.finfo(step_bwd.dtype).eps)
            return jnp.logical_or(
                i == 0, # always do at least 1 iteration
                jnp.logical_not(
                    jnp.logical_or(
                        self.in_slice(new_state, target_log_joint),
                        jnp.isclose(step_fwd, step_bwd, atol=eps, rtol=eps)
                    )
                )
            )
        
        # shrinking loop
        *_, step_fwd, step_bwd, new_state = lax.while_loop(
            cond_fn, 
            body_fn,
            (0, rng_key, init_step_fwd, init_step_bwd, init_state) 
        )

        return new_state, step_fwd, step_bwd
            
    def sample_single_chain(self, state, model_args, model_kwargs):
        # generate rng keys and store the updated master key in the state
        (
            rng_key, 
            precond_key, 
            aux_key,
            init_key, 
            height_key,
            grow_key,
            shrink_key
        ) = random.split(state.rng_key, 7)
        state = state._replace(rng_key = rng_key)

        # build a (possibly randomized) preconditioner
        precond_state = self.preconditioner.maybe_alter_precond_state(
            state.base_precond_state, precond_key
        )

        # refresh the direction
        state = self.update_log_joint(
            self.refresh_aux_vars(aux_key, state, precond_state), precond_state
        )

        # draw a target log height
        # note: y = pi(z)U(0,1) <=> log(y) = logpi(z) - Exp(1)
        target_log_joint = state.log_joint - random.exponential(height_key)

        # init slice endpoints
        bwd_state, fwd_state = self.init_endpoints(
            init_key, state, precond_state
        )

        # grow slice
        bwd_state, fwd_state = self.grow_slice(
            grow_key, 
            bwd_state, 
            fwd_state, 
            target_log_joint, 
            precond_state
        )

        # shrink slice
        new_state, step_fwd, step_bwd = self.shrink_slice( 
            shrink_key, 
            state, 
            bwd_state, 
            fwd_state, 
            target_log_joint, 
            precond_state
        )

        # collect statistics
        new_x = jax.flatten_util.ravel_pytree(
            getattr(new_state, self.sample_field)
        )[0]
        estimated_slice_width = step_fwd-step_bwd # avg of this seems to shrink as d^{-1/2} for N(0,I)
        new_stats = statistics.record_post_sample_stats(
            new_state.stats, estimated_slice_width, 1.0, True, new_x            
        )
        new_state = new_state._replace(stats = new_stats)

        # maybe adapt
        new_state = self.adapt(new_state)

        return new_state

    
class DeterministicScanSliceSampler(SliceSampler):

    def init_extras(self, state):
        init_fn = lambda vec: jax.nn.one_hot(0, len(vec), dtype=vec.dtype)
        if jnp.shape(state.rng_key) == ():
            p_flat = init_fn(state.p_flat)
        else:
            p_flat = jax.vmap(init_fn)(state.p_flat)
        return state._replace(p_flat = p_flat)

    def refresh_aux_vars(self, rng_key, state, precond_state):
        return state._replace(p_flat = jnp.roll(state.p_flat, 1))


class HitAndRunSliceSampler(SliceSampler):
    """
    Hit-and-Run sampler implemented via slice sampling along lines.

    Kiatsupaibul, S., Smith, R. L., & Zabinsky, Z. B. (2011). An analysis of a 
    variation of hit-and-run for uniform sampling from general regions. ACM 
    Transactions on Modeling and Computer Simulation (TOMACS), 21(3), 1-11.
    """
    # sample p ~ N(0,S), where S is approx the posterior covariance
    # equivalent to v~N(0,I) and p = Lv, with LL^T = S.
    def refresh_aux_vars(self, rng_key, state, precond_state):
        v_flat = random.normal(rng_key, jnp.shape(state.p_flat))
        L = precond_state.var_tril_factor
        p_flat = L @ v_flat if jnp.ndim(L) == 2 else L * v_flat
        return state._replace(p_flat = p_flat)
    