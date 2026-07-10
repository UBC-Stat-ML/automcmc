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
    Return `x1-x0` if `x1` is not the next float after `x0` or if any of them
    is not finite. Otherwise, return `0`.
    """
    return jnp.where(
        jnp.logical_and(
            jnp.logical_and(jnp.isfinite(x0), jnp.isfinite(x1)),
            jnp.nextafter(x0, x1) == x1
        ),
        jnp.zeros_like(x0),
        x1-x0
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

@jax.jit
def _symmetric_allclose_thresholds(a,b,rtol,atol):
    elemwise_smallest_abs = jnp.minimum(jnp.abs(a), jnp.abs(b))
    elemwise_thresholds = jnp.maximum(atol, rtol*elemwise_smallest_abs)
    return elemwise_thresholds

@jax.jit
def symmetric_allclose(
        a: ArrayLike,
        b: ArrayLike,
        rtol: ArrayLike,
        atol: ArrayLike
    ) -> jax.Array:
    """
    Symmetric version of :func:`jnp.allclose`, which is basically elementwise
    :func:`jnp.isclose` plus a :func:`jnp.all` reduction. Note that
    ```
        |a-b| < max{atol, rtol*min{|a|,|b|}} < max{atol,rtol|b|} < atol+rtol|b|
    ```
    The first inequality is symmetric in `(a,b)`, and the last term on the
    right is the threshold used in `jnp.isclose`. So the symmetric version is
    also stricter.

    :param ArrayLike a: first array to compare.
    :param ArrayLike b: second array to compare.
    :param ArrayLike rtol: relative tolerance.
    :param ArrayLike atol: absolute tolerance
    :return jax.Array: `True` if close.
    """
    assert jnp.shape(a) == jnp.shape(b)
    elemwise_thresholds = _symmetric_allclose_thresholds(a, b, rtol, atol)
    return jnp.all(jnp.abs(a-b) < elemwise_thresholds)

def newton_default_tol(x: ArrayLike) -> jax.Array:
    # for float64, eps^(0.32) ~ 1e-5 which is the std tol use in most
    # implementations of Newton algorithm. We use eps^0.5 for a slightly
    # tighter requirement. This is then increased linearly because we focus on
    # max-error, which in worse case (heavy tailed sequences) scales as sum so
    # O(n). It would be logarithmically in the iid case if we also assume
    # subexponential tails, but this is not representative of the errors seen
    # in experiments (very heavy tailed).
    return x.size*jnp.sqrt(jnp.finfo(x.dtype).eps)

def newton_fn_value_err(val):
    return jnp.abs(val).max()

def newton(
        f: Callable[[ArrayLike], jax.Array],
        x0: ArrayLike,
        tol: ArrayLike,
        max_iter: int = 50,
        mode: str = "direct"
    ) -> tuple:
    """
    A Newton root solver.

    :param Callable[[ArrayLike], jax.Array] f: target function
    :param ArrayLike x0: initial guess
    :param Optional[float] tol: convergence tolerance, defaults to None, in
        which case an adequate value is chosen depending on the float type of
        the inputs.
    :param int max_iter: maximum number of Newton steps, defaults to 100
    :param str mode: one of `("direct", "gmres")`. The option `"gmres"` uses
        the iterative GMRES solver together with :func:`jax.linearize` to avoid
        forming the full Jacobian. When `mode="direct"`, the full Jacobian is
        formed and the update direction is obtained via a linear solve. The
        latter is the default choice, as this solver is intended for use in a
        setting where :math:`O(d^2)` storage is acceptable, and direct solvers
        should always be preferred when feasible.
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
    n_skip_err_inc = max_iter // 10 # first tenth of max iters can increase error
    err0 = newton_fn_value_err(val0)

    def cond_fn(carry):
        x, n, val, err, d_err = carry
        return jnp.logical_and(
            # still have budget to go
            n < max_iter,
            jnp.logical_and(
                # error still high
                err >= tol,
                # after `n_skip_err_inc` round, error is not increasing significantly
                jnp.logical_or(n<n_skip_err_inc, d_err < tol)
            )
        )

    def body_fn(carry):
        x, n, val0, err0, _ = carry
        n += 1

        # solve for Newton's update
        #   J_f(x) dx = -f(x) ==> x' = x + dx
        if mode == "direct":
            # form full Jacobian and use a direct solver
            # use fwd mode since the system is square
            dx = jnp.linalg.solve(jax.jacfwd(f)(x), -val0)
        elif mode == "gmres":
            # use GMRES with Jacobian-vector products (i.e. fwd mode autodiff)
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
