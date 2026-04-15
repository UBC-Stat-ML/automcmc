from typing import Callable, Optional

import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import checkify
from jax.typing import ArrayLike

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

###############################################################################
# numerical utils
###############################################################################

def newton_default_tol(x):
    return 10*jnp.finfo(x.dtype).eps

def newton_fn_value_err(val):
    return jnp.abs(val).max()

def newton(
        f: Callable[[ArrayLike], jax.Array],
        x0: ArrayLike,
        tol: Optional[float] = None,
        max_iter: int = 100,
        mode: str = "gmres"
    ) -> tuple:
    """
    A Newton root solver.

    :param Callable[[ArrayLike], jax.Array] f: target function
    :param ArrayLike x0: initial guess
    :param Optional[float] tol: convergence tolerance, defaults to None, in
        which case an adequate value is chosen depending on the float type of
        the inputs.
    :param int max_iter: maximum number of Newton steps, defaults to 100
    :param str mode: one of `("direct", "gmres")`. The default `"gmres"` uses
        the iterative GMRES solver together with :func:`jax.linearize` to avoid
        forming the full Jacobian. When `mode="direct"`, the full Jacobian is
        formed and the update direction is obtained via a linear solve.
    :return tuple: A tuple of

        * `x`: root
        * `n`: number of iterations
        * `val`: function value at the root
        * `err`: maximum absolute value of `val`
        * `d_err`: change in `err` in the last iteration
        * `flag`: true if `err<tol` (i.e., success)
    """
    dim = len(x0)
    val0 = f(x0)
    assert len(val0) == dim
    err0 = newton_fn_value_err(val0)
    if tol is None:
        tol = newton_default_tol(err0)

    def cond_fn(carry):
        x, n, val, err, d_err = carry
        return jnp.logical_and(
            n < max_iter,                        # still have budget to go
            jnp.logical_and(
                err >= tol,                      # error still high
                jnp.logical_or(n<3, d_err < tol) # after 3rd round, error is not increasing significantly
            )
        )

    def body_fn(carry):
        x, n, val0, err0, _ = carry
        n += 1

        # solve for Newton's update
        #   J_f(x) dx = -f(x) ==> x' = x + dx
        if mode == "direct":
            # form full Jacobian and use a direct solver
            dx = jnp.linalg.solve(jax.jacobian(f)(x), -val0)
        elif mode == "gmres":
            # use GMRES with Jacobian-vector products
            # Note: we use this solver in a setting where dim^2 storage is ok,
            # and the absolute worst case time complexity O(dim^3) is tolerable
            # every now and then. Therefore, we avoid restarting
            jvp = jax.linearize(f, x)[1] # faster than e.g. lambda v: jax.jvp(f, (x,), (v,))[1]
            dx = jax.scipy.sparse.linalg.gmres(
                jvp, -val0, tol=tol, restart=dim
            )[0]
        else:
            raise ValueError(f"Unknown mode `{mode}`")

        # return updated carry
        x += dx
        val = f(x)
        err = newton_fn_value_err(val)
        return (x, n, val, err, err-err0)

    # run loop and return full carry for diagnostics plus flag
    carry = jax.lax.while_loop(cond_fn, body_fn, (x0, 0, val0, err0, err0))
    return (*carry, carry[3]<tol)
