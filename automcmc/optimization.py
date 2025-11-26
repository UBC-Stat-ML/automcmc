from copy import deepcopy

import math

import numpy as np

import jax
from jax import numpy as jnp

import optax
import tqdm

from automcmc import utils

# NADAMW
# More robust than ADAM according to https://arxiv.org/abs/2306.07179)
def make_nadamw_solver(target_fun, solver_params, verbose):
    if verbose:
        print(f'NADAMW optimization loop.')
    
    # build solver
    solver = optax.nadamw(**solver_params)

    # build loop function and jit it
    @jax.jit
    def step_fn(params, state):
        value, grad = jax.value_and_grad(target_fun)(params)
        updates, state = solver.update(grad, state, params)
        params = optax.apply_updates(params, updates)
        grad_norm = utils.pytree_norm(grad, ord=jnp.inf) # sup norm
        return params, state, value, grad_norm
    
    return solver, step_fn


# L-BFGS
def make_lbfgs_solver(target_fun, solver_params, verbose):
    if verbose:
        print(f'L-BFGS optimization loop.')

    # can reuse stuff because target is deterministic
    # see https://optax.readthedocs.io/en/stable/api/optimizers.html#lbfgs
    cached_value_and_grad = optax.value_and_grad_from_state(target_fun)

    # build solver and loop function
    solver = optax.lbfgs(**solver_params)
    @jax.jit
    def step_fn(params, opt_state):
        value, grad = cached_value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=target_fun
        )
        params = optax.apply_updates(params, updates)
        grad_norm = utils.pytree_norm(grad, ord=jnp.inf) # sup norm
        return params, opt_state, value, grad_norm
    
    return solver, step_fn

def optimization_loop(
        target_fun,
        step_fn,
        params,
        opt_state,
        n_iter, 
        tol = None,
        max_consecutive = 2, 
        verbose = True
    ):
    if tol is None:
        tol = 10*jnp.finfo(jax.tree.leaves(params)[0].dtype).eps
    
    value = old_value = target_fun(params)
    verbose and print(f'Initial energy: {value:.1e}')
    old_params = params
    grad_norm = value_abs_diff = params_diff_norm = jnp.full_like(value, 10*tol)
    n_consecutive = np.zeros((3,), np.int32) # one counter for each termination criterion
    n = 0
    with tqdm.tqdm(total=n_iter, disable=(not verbose)) as t:
        while (n < n_iter and np.all(n_consecutive < max_consecutive)):
            # take one optim step
            params, opt_state, value, grad_norm = step_fn(params, opt_state)

            # update termination indicators
            value_abs_diff = jnp.abs(value-old_value)
            old_value = value
            params_diff_norm = utils.pytree_norm(
                jax.tree.map(lambda a,b: a-b, params, old_params),
                ord=jnp.inf # sup norm
            )
            old_params = params

            # update termination counters
            for (i,eps) in enumerate((grad_norm,value_abs_diff,params_diff_norm)):
                n_consecutive[i] = n_consecutive[i]+1 if eps < tol else 0

            # update progress bar and iteration counter
            diag_str = "f={:.1e}, Δf={:.0e}, |g|={:.0e}, |Δx|={:.0e}" \
                .format(value, value_abs_diff, grad_norm, params_diff_norm)
            t.set_postfix_str(diag_str, refresh=False) # will refresh with `update`
            t.update()
            n += 1
    print(f'Final energy: {value:.1e}')
    return params, opt_state, value

DEFAULT_OPTIMIZE_FUN_SETTINGS = {
    "NADAMW": {"n_iter": 1024, "solver_params": {"learning_rate": 0.003}},
    "L-BFGS": {"n_iter": 32, "solver_params": {}}
}

def optimize_fun(
        target_fun, 
        init_params, 
        settings = DEFAULT_OPTIMIZE_FUN_SETTINGS,
        verbose = True,
        **kwargs
    ):
    """
    Gradient based optimization. By default it uses a two stage procedure:
    an initial stage with NADAMW, and a second stage with L-BFGS. The idea is
    to use NADAMW to find a good enough point for L-BFGS to take it from there.
    This is especially useful when working with lower precision floats such as
    `jnp.float32` (see e.g. [1]).
    
    :param target_fun: Function to minimize
    :param init_params: Starting point
    :param settings: Dictionary of settings passed to the solvers. See 
        :data:`DEFAULT_OPTIMIZE_FUN_SETTINGS` for an example.
    :param verbose: Should we print info?
    :param kwargs: Passed to :func:`optimization_loop`
    :return: Solution found

    .. rubric:: References

    .. [1] Kiyani, E., Shukla, K., Urbán, J. F., Darbon, J., & Karniadakis, 
        G. E. (2025). Optimizing the optimizer for physics-informed neural 
        networks and Kolmogorov-Arnold networks. *Computer Methods in Applied 
        Mechanics and Engineering, 446*, 118308.
    """
    # start with NADAMW
    init_value = value_nadamw = target_fun(init_params)
    opt_params = init_params
    if "NADAMW" in settings:
        solver, step_fn = make_nadamw_solver(
            target_fun, settings["NADAMW"]['solver_params'], verbose
        )
        opt_state = solver.init(init_params)
        opt_params_nadamw, _, value_nadamw = optimization_loop(
            target_fun,
            step_fn,
            init_params,
            opt_state,
            settings["NADAMW"]['n_iter'],
            verbose=verbose,
            **kwargs
        )
        if jnp.isfinite(value_nadamw) and value_nadamw<init_value:
            opt_params = opt_params_nadamw
        
    # refine with L-BFGS
    if "L-BFGS" in settings:
        solver, step_fn = make_lbfgs_solver(
            target_fun, settings["L-BFGS"]['solver_params'], verbose
        )
        opt_state = solver.init(opt_params)
        opt_params_lbfgs, _, value_lbfgs = optimization_loop(
            target_fun,
            step_fn,
            opt_params,
            opt_state,
            settings["L-BFGS"]['n_iter'],
            verbose=verbose,
            **kwargs
        )
    
    # return lbfgs if better 
    if value_lbfgs < value_nadamw:
        return opt_params_lbfgs
    else:
        return opt_params_nadamw
