import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import checkify

###############################################################################
# basic utilities
###############################################################################

@checkify.checkify
def checkified_is_finite(x):
  checkify.check(lax.is_finite(x), f"Found non-finite value x = {x}")
  return True

@checkify.checkify
def checkified_is_zero(x):
  checkify.check(x==0, f"Expected zero but x = {x}")
  return True

def pytree_norm(x, *args, **kwargs):
    return jnp.linalg.norm(jax.flatten_util.ravel_pytree(x)[0], *args, **kwargs)

def ceil_log2(x):
    """
    Ceiling of log2(x). Guaranteed to be an integer.
    """
    n_bits = jax.lax.clz(jnp.zeros_like(x))
    return n_bits - jax.lax.clz(x) - (jax.lax.population_count(x)==1)

def numerically_safe_diff(x0, x1):
    """
    Return `x1-x0` if x1 is not the next float after x0, and 0 otherwise.
    """
    return jnp.where(
        jax.lax.nextafter(x0, x1) == x1, jnp.zeros_like(x0), x1-x0
    )

###############################################################################
# Rounds-based sampling arithmetic
#
# Running a rounds-based sampler for "n_r" rounds means we take a total of
#   n_samples = 2 + 4 + 8 + ... + 2^{n_r} = 2^(n_r+1) - 2 = [2^{n_r} - 2] + [2^{n_r}]
# samples. The decomposition in the RHS shows that this corresponds to
#   - A warmup phase of n_r-1 rounds, with a total of 2^{n_r} - 2 samples.
#     We call this quantity "n_warmup".
#   - A main sampling phase comprised of a final round with 2^{n_r} steps.
#     We call this quantity "n_keep".
# We use the name "n_samples" for the sum of "n_warmup" and "n_keep". Finally,
# we call "sample_idx" the current step within a round, which resets at the end
# of every round.
###############################################################################

def n_steps_in_round(round):
    return 2 ** round

def split_n_rounds(n_rounds):
    n_keep = n_steps_in_round(n_rounds)
    return (n_keep-2, n_keep)

def current_round(n_samples):
    return ceil_log2(n_samples + 2) - 1

def n_warmup_to_adapt_rounds(n_warmup):
    return ceil_log2(n_warmup + 2) - 1
