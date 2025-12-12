from functools import partial

import jax
from jax import flatten_util
from jax import lax
from jax import numpy as jnp
from jax import random

from automcmc import autostep
from automcmc import tempering

class AutoHMC(autostep.AutoStep):

    def __init__(
            self, 
            *args, 
            n_leapfrog_steps=32, 
            forward_mode_ad=False, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.n_leapfrog_steps = n_leapfrog_steps
        self.forward_mode_ad = forward_mode_ad
        self.integrator = None
            
    def init_extras(self, initial_state):
        if jnp.shape(initial_state.rng_key) == ():
            proto_x = initial_state.x
        else:
            proto_x = jax.tree.map(lambda xi: xi[0], initial_state.x)
        self.integrator = gen_integrator(
            self.logprior_and_loglik, proto_x, self.forward_mode_ad
        )
        return initial_state
    
    def kinetic_energy(self, state, precond_state):
        # compute kinetic energy
        # autoHMC works in "momentum" convention -- i.e., p_flat ~ N(0,M) -- as
        # opposed to the "velocity" convention used by autoRWMH.
        # This is to minimize the number of matrix operations per leapfrog step.
        p_flat = state.p_flat
        Minv = precond_state.var
        Minv_p_flat = Minv @ p_flat if jnp.ndim(Minv) == 2 else Minv * p_flat
        return 0.5*jnp.dot(Minv_p_flat, p_flat)
    
    def refresh_aux_vars(self, rng_key, state, precond_state):
        v_flat = random.normal(rng_key, jnp.shape(state.p_flat))
        U = precond_state.inv_var_triu_factor
        p_flat = U @ v_flat if jnp.ndim(U) == 2 else U * v_flat
        return state._replace(p_flat = p_flat)
    
    def involution_main(self, step_size, state, precond_state):
        return self.integrator(
            step_size, state, precond_state, self.n_leapfrog_steps
        )
    
    def involution_aux(self, state):
        return state._replace(p_flat = -state.p_flat)


##############################################################################
# leapfrog (velocity Verlet) integrator
# The update is
# 	p* = p  - (eps/2)grad(U)(x)
# 	x' = x  + eps M^{-1}p*
# 	p' = p* - (eps/2)grad(U)(x')
# Let L lower triangular such that
#   M = LL^T
# If y ~ N(0,I), then
#   p = Ly ~ N(0,M)
# Working in this space requires
#   - Getting L=chol(M) and M^{-1}= [L^{-1}]^T [L^{-1}]
#   - At each step we need 3 matrix ops:
#       1) Sampling p = Ly
#       2) kinetic energy eval: [L^{-1}p]^T [L^{-1}p]
#       3) For each leapfrog, a single matop (M^{-1}p)
# The alternative is to work with y:=L^{-1}p, which gives the updates 
# 	y* = y  - (eps/2)L^{-1}grad(U)(x)
# 	x' = x  + eps (L^T)^{-1}y
# 	y' = y* - (eps/2)L^{-1}grad(U)(x')
# This means we need 
#   - To get L^{-1} and (L^{-1})^T
#   - No matrix ops outside leapfrog
#   - ...but 3 O(d^2) ops for every LF step!
# So this is too expensive.
#
# Note: when M = S^{-1}, with S=Cov(x). Clearly, M^{-1}=S. But also,
#   S = LL^T <=> M = S^{-1} = (L^T)^{-1}L^{-1} = UU^T
# where U := (L^T)^{-1}. This solve is O(d^2) because L^T is upper triangular,
# which also means that U is also upper triangular. Then we have
#   p = Uy ~ N(0,M)
# Hence, we need S (no-op as this is directly estimated) and U.
##############################################################################

def velocity_step(p_flat, step_size, grad_flat):
    return p_flat - step_size * grad_flat

def position_step(x_flat, step_size, precond_state, p_flat):
    if jnp.ndim(precond_state.var) == 2:
        prec_p_flat = precond_state.var @ p_flat
    else:
        prec_p_flat = precond_state.var * p_flat
    return x_flat + step_size * prec_p_flat

# leapfrog integrator using Neal (2011, Fig. 2) trick to use only (n_steps+1)
# gradient evaluations
def gen_integrator(logprior_and_loglik, proto_x, forward_mode_ad):
    unravel_fn = flatten_util.ravel_pytree(proto_x)[1]
    tempered_potential = partial(
        tempering.tempered_potential, logprior_and_loglik
    )

    def grad_flat_x_flat(x_flat, inv_temp):
        """
        Flattened gradient of the potential evaluated at a flattened 
        unconstrained state.
        """
        # unflatten state and take gradient 
        # note: by default, `grad` and `jacfwd` take diff w.r.t. the first arg only
        x = unravel_fn(x_flat)
        if forward_mode_ad:
            grad_x = jax.jacfwd(tempered_potential)(x, inv_temp)
        else:
            grad_x = jax.grad(tempered_potential)(x, inv_temp)

        # flatten grad and return
        return flatten_util.ravel_pytree(grad_x)[0]

    # full position and velocity updates
    def integrator_scan_body_fn(carry, _):
        x_flat, p_flat, step_size, precond_state, inv_temp = carry
        x_flat    = position_step(x_flat, step_size, precond_state, p_flat)
        grad_flat = grad_flat_x_flat(x_flat, inv_temp)
        p_flat    = velocity_step(p_flat, step_size, grad_flat)
        return (x_flat, p_flat, step_size, precond_state, inv_temp), None
    
    def integrator(step_size, state, precond_state, n_steps):
        # jax.debug.print("start: step_size={s}, precond_state={d}", ordered=True, s=step_size, d=precond_state)
        # explicitly unpack state: avoid using tuple unpacking as it breaks
        # when the definition of `state` changes
        x = state.x
        p_flat = state.p_flat
        inv_temp = state.inv_temp

        # first velocity half-step
        x_flat    = flatten_util.ravel_pytree(x)[0]
        grad_flat = grad_flat_x_flat(x_flat, inv_temp)
        # jax.debug.print("pre 1st momentum half-step: x={x}, x_flat={xf}, grad={g}, p_flat={v}", ordered=True, x=x, xf=x_flat, g=grad_flat, v=p_flat)
        p_flat    = velocity_step(p_flat, 0.5*step_size, grad_flat)
        # jax.debug.print("post: p_flat={v}", ordered=True, v=p_flat)

        # loop full position and velocity leapfrog steps
        # note: slight modification from Neal's to avoid the "if" inside the loop
        # In particular, MALA (n_steps=1) doesn't enter the loop
        x_flat, p_flat, *_ = lax.scan(
            integrator_scan_body_fn,
            (x_flat, p_flat, step_size, precond_state, inv_temp),
            None,
            n_steps-1
        )[0]
        # jax.debug.print("post loop: x_flat={xf}, p_flat={v}", ordered=True, xf=x_flat, v=p_flat)

        # final full position step plus half velocity step
        x_flat    = position_step(x_flat, step_size, precond_state, p_flat)
        grad_flat = grad_flat_x_flat(x_flat, inv_temp)
        p_flat    = velocity_step(p_flat, 0.5*step_size, grad_flat)
        
        # unravel, update state, and return it
        x_new = unravel_fn(x_flat)
        # jax.debug.print("final: x_new={x}, x_flat={xf}, grad={g}, p_flat={v}", ordered=True, x=x_new, xf=x_flat, g=grad_flat, v=p_flat)

        return state._replace(x = x_new, p_flat = p_flat)
    
    return integrator


## autoMALA alias
def AutoMALA(*args, **kwargs):
    return AutoHMC(n_leapfrog_steps=1, *args, **kwargs)
