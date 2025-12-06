from jax import flatten_util
from jax import numpy as jnp
from jax import random

from automcmc import autostep

class AutoRWMH(autostep.AutoStep):
   
    def refresh_aux_vars(self, rng_key, state, precond_state):
        p_flat = random.normal(rng_key, jnp.shape(state.p_flat))
        return state._replace(p_flat = p_flat)
    
    def involution_main(self, step_size, state, precond_state):
        x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
        if jnp.ndim(precond_state.var_tril_factor) == 2:
            prec_p_flat = precond_state.var_tril_factor @ state.p_flat
        else:
            prec_p_flat = precond_state.var_tril_factor * state.p_flat
        x_new = unravel_fn(x_flat + step_size * prec_p_flat)
        return state._replace(x = x_new)
    
    def involution_aux(self, state):
        return state._replace(p_flat = -state.p_flat)

