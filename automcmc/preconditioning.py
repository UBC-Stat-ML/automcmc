from abc import ABC, abstractmethod

from collections import namedtuple

import jax
from jax import lax
from jax import numpy as jnp

###############################################################################
# Preconditioner state
# 
# The most demanding application is for dense preconditioners. To handle both
# AutoRWMH and AutoHMC. Let S = Cov(x), M = S^{-1}, an y~N(0,I). We need,
#   - S, for
#       - AutoHMC: kinetic energy 0.5p^TM^{-1}p = 0.5*jnp.dot(S@p,p)
#       - AutoHMC: LF postition update x' = x + eps M^{-1}p = x + eps Sp
#   - L = chol(S), for 
#       - AutoRWMH: proposal x' = x + epsLy
#   - A factorization of the form M = UU^T for
#       - AutoHMC: sample momentum p = Uy ~ N(0,M)
# 
# The last decomposition can be gotten by manipulating a Cholesky factorization
# of S, since
#     S = LL^T <=> M = S^{-1} = (L^T)^{-1}L^{-1} = UU^T  
# This approach---an O(d^3) chol followed by an O(d^2) triangular solve---is 
# much more efficient than doing 2 O(d^3) ops in chol(inv(S)). We don't need
# U to be lower triangular or even triangular. Any decomposition M=UU^T works. 
#
# An example to check that this works:
# 
# # sample a symmetric mat (a.s. posdef) to use as variance
# var_key, sim_key = random.split(random.key(2176543))
# N = random.normal(var_key, (2,2))
# S = N.T @ N

# # chol of variance mat
# L = jax.lax.linalg.cholesky(S)
# jnp.allclose(S, L @ L.T)

# # decompose M = S^{-1} = UU^T by doing a triangular solve of L^{T}
# U = jax.lax.linalg.triangular_solve(L.T, jnp.identity(2))
# M = jnp.linalg.inv(S)
# jnp.allclose(M, U @ U.T)

# # check that p=Uy ~ N(0,M) when y ~ N(0,I)
# ps = jax.vmap(
#     lambda rng_key: U @ random.normal(rng_key, (2,)),
#     out_axes=1
# )(random.split(sim_key, 10000))
# jnp.allclose(M, jnp.cov(ps), rtol=0.05)
###############################################################################

PreconditionerState = namedtuple(
    "PreconditionerState",
    [
        "mean",
        "var",
        "var_tril_factor",
        "inv_var_triu_factor"
    ]
)
"""
A :func:`~collections.namedtuple` defining the values associated with a 
preconditioner. It consists of the fields:

 - **mean** - Estimated target mean.
 - **var** - Estimated and regularized target variance.
 - **var_tril_factor** - A lower-triangular (tril) matrix `L` such that 
     `var == L @ L.T`.
 - **inv_var_triu_factor** - An upper-triangular (triu) matrix `U` such that 
     `inv(var) == U @ U.T`.
"""

###############################################################################
# type definitions
###############################################################################

class Preconditioner(ABC):

    @staticmethod
    @abstractmethod
    def maybe_alter_precond_state(precond_state, rng_key):
        """
        Build a (possible randomized) preconditioner.

        :param precond_state: An array representing the square-root of a covariance matrix.
            It can be a matrix---for dense preconditioners---or a vector---for
            the diagonal case.
        :param rng_key: A PRNG key that could be used for random preconditioning.
        :return: A vector representing a diagonal preconditioner.
        """
        pass

#######################################
# Diagonal
#######################################

class IdentityDiagonalPreconditioner(Preconditioner):

    @staticmethod
    def maybe_alter_precond_state(precond_state, rng_key):
        I = jnp.ones_like(precond_state.var)
        return PreconditionerState(precond_state.mean, I, I, I)

class FixedDiagonalPreconditioner(Preconditioner):

    @staticmethod
    def maybe_alter_precond_state(precond_state, rng_key):
        return precond_state

#######################################
# Dense
#######################################

class FixedDensePreconditioner(Preconditioner):

    @staticmethod
    def maybe_alter_precond_state(precond_state, rng_key):
        return precond_state

def is_dense(preconditioner):
    return (
        isinstance(preconditioner, FixedDensePreconditioner)
    )

###############################################################################
# methods
###############################################################################

# initialization
def init_base_precond_state(prototype_x_flat, preconditioner):
    zeros = jnp.zeros_like(prototype_x_flat)
    if is_dense(preconditioner):
        I = jnp.identity(prototype_x_flat.shape[0],prototype_x_flat.dtype)
        return PreconditionerState(zeros, I, I, I)
    else: 
        ones = jnp.ones_like(prototype_x_flat)
        return PreconditionerState(zeros, ones, ones, ones)

def fix_cond_number(sample_var):
    eps = jnp.finfo(sample_var.dtype).eps
    dim = sample_var.shape[-1]
    if jnp.ndim(sample_var) == 2:
        # add a nugget to the estimated sample variance
        # inspired by default `rtol` in `jnp.matrix_rank`
        # https://github.com/jax-ml/jax/blob/30582db24e8794abd09df2b3120aa5b58af8e9fe/jax/_src/numpy/linalg.py#L463
        sample_var_eigvals = jnp.linalg.eigvalsh(sample_var)
        nugget = sample_var_eigvals[-1] * dim * eps
        corrected_sample_var = sample_var + nugget*jnp.identity(dim)
    else:
        # need less severe correction -> no eigval and backstop with sqrt{eps}
        nugget = jnp.minimum(dim * eps, jnp.sqrt(eps))
        corrected_sample_var = sample_var + nugget

    return corrected_sample_var

# adapt the base preconditioner state, regularizing to avoid issues with
# ill-conditioned sample variances
def adapt_base_precond_state(base_precond_state, adapt_stats):
    n_samples = adapt_stats.sample_idx
    sample_var = adapt_stats.sample_var
    corrected_sample_var = fix_cond_number(sample_var)

    # new variance as weighted average of old and corrected sample var
    # note: the decaying schedule is from NumPyro (which says it's from Stan):
    # https://github.com/pyro-ppl/numpyro/blob/ab1f0dc6e954ef7d54724386667e33010b2cfc8b/numpyro/infer/hmc_util.py#L219
    w_new = n_samples / (n_samples+5)
    var = (1-w_new)*base_precond_state.var + w_new*corrected_sample_var

    # compute the remaining terms, depending on shape
    if jnp.ndim(sample_var) == 2:
        var_tril_factor = lax.linalg.cholesky(var)
        inv_var_triu_factor = lax.linalg.triangular_solve(
            var_tril_factor, 
            jnp.identity(var.shape[-1]), 
            transpose_a=True, 
            lower=True
        )
    else:
        var_tril_factor = jnp.sqrt(var)
        inv_var_triu_factor = jnp.reciprocal(var_tril_factor)

    return PreconditionerState(
        adapt_stats.sample_mean, var, var_tril_factor, inv_var_triu_factor
    )
