from jax import flatten_util
from jax import numpy as jnp
from jax import random

from automcmc import autostep, selectors

class AutoPCN(autostep.AutoStep):
    """
    Involutive implementation of a finite-dimensional preconditioned 
    Crank-Nicolson sampler à la [1]. We do this by identifying the parameter
    of the AR1 proposal as the sine of an angle :math:`theta\\in(-pi,pi)`. This 
    allows us to view the pCN proposal as a joint rotation of the position and 
    the Gaussian variable, which is clearly involutive if we tack on an angle
    sign flip. Note that this technically increases the set of allowed 
    proposals, as we can access linear combinations of the previous state and
    the Gaussian innovation that have negative coefficients.

    Another important difference with the approach in [1] is that, instead of
    assuming the prior is Gaussian and drawing directly from it, we track a 
    normal approximation to the posterior distribution using the 
    preconditioning machinery of `automcmc`.
    
    .. note::
       For implementation purposes, we handle :math:`|theta|` with the step 
       size, whereas :math:`sgn(theta)` is stored in the `idiosyncratic` field
       of :class:`AutoMCMCState`.

    .. warning:: This class is still under development.

    [1] Cotter, S. L., Roberts, G. O., Stuart, A. M., & White, D. (2013). 
    MCMC methods for functions: Modifying old algorithms to make them faster.
    *Statistical Science*, 424-446.
    """

    def __init__(
            self, 
            *args, 
            selector = selectors.MaxEJDSelector(),
            **kwargs
        ):
        super().__init__(*args, selector=selector, **kwargs)
        self._auto_step_size_fn = self.selector.gen_executor(self)


    # use the optional `idiosyncratic` field to store the sign of the angle
    def init_extras(self, state):
        return state._replace(
            idiosyncratic = jnp.ones((), state.base_step_size.dtype)
        )
    
    # compute kinetic energy for p ~ N(m, S). Recall that we have U s.t.
    #   UU^T = S^{-1}
    # So the kinetic energy is
    #   0.5 (p-m)^T S^{-1} (p-m) = 0.5 (p-m)^T(UU^T)(p-m) = 0.5 v^Tv
    # where v:=U^T(p-m), which is the actual variable we carry around
    # Note: we cannot skip this step like in AutoRWMH because the pCN 
    # involution does affect the momentum variable
    def kinetic_energy(self, state, precond_state):
        v_flat = state.p_flat
        return 0.5*jnp.dot(v_flat, v_flat)
    
    # sample p ~ N(m,S), where m and S are the approx posterior mean and 
    # covariance, respectively. Equivalent to v~N(0,I) and p = m + Lv, with 
    # LL^T = S. Thus, we instead draw v and store it in `p_flat`
    def refresh_aux_vars(self, rng_key, state, precond_state):
        v_flat = random.normal(rng_key, jnp.shape(state.p_flat))
        return state._replace(p_flat = v_flat)
    
    # pCN as joint elliptical rotation (along circle in standardized space)
    def involution_main(self, step_size, state, precond_state):
        x_flat, unravel_fn = flatten_util.ravel_pytree(state.x)
        v_flat = state.p_flat
        m, _, L, U = precond_state
        
        # standardize x
        dense = jnp.ndim(U) == 2
        x_flat_cen = x_flat-m
        x_flat_std = jnp.dot(x_flat_cen, U) if dense else U * x_flat_cen # jnp.dot(v, A) == A.T @ v

        # jointly rotate the standardized vectors
        theta = step_size * state.idiosyncratic
        sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
        x_flat_std_new =  cos_theta*x_flat_std + sin_theta*v_flat
        v_flat_new     = -sin_theta*x_flat_std + cos_theta*v_flat

        # undo standardization, update state, and return
        x_flat_new = m + (L @ x_flat_std_new if dense else L * x_flat_std_new)
        return state._replace(x = unravel_fn(x_flat_new), p_flat = v_flat_new)
    
    # flip theta sign
    def involution_aux(self, state):
        return state._replace(idiosyncratic = -state.idiosyncratic)
    
    # map the integers into a lattice in (0,pi), such that
    #   1) exponent = 0 => |theta'| = |theta|
    #   2) exponent -> -inf => |theta'| -> 0
    #   3) exponent ->  inf => |theta'| -> pi
    def step_size(self, base_step_size, exponent):
        abs_theta = base_step_size
        return 2*jnp.arctan(jnp.tan(abs_theta/2) * (2.0 ** exponent))

