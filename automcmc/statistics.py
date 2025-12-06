from collections import namedtuple

import jax
from jax import numpy as jnp

from automcmc.utils import current_round
from automcmc.preconditioning import is_dense, adapt_base_precond_state

AutoMCMCAdaptStats = namedtuple(
    "AutoMCMCAdaptStats",
    [
        "sample_idx",
        "mean_step_size",
        "mean_acc_prob",
        "rev_rate",
        "sample_mean",
        "sample_var"
    ],
    defaults=(0, 0.0, 0.0, 0.0, None, None)
)
"""
A :func:`~collections.namedtuple` used for round-based adaptation. Its contents
are cleared at the end of each round. Its fields are:

 - **sample_idx** - current sample in the current round.
 - **mean_step_size** - online mean step size for the round.
 - **mean_acc_prob** - online mean acceptance probability for the round.
 - **rev_rate** - online reversibility rate for the round.
 - **sample_mean** - online mean of the flattened sample field.
 - **sample_var** - online variance estimate of the flattened sample field.
"""

def make_adapt_stats_recorder(
        prototype_x_flat, 
        sample_var_shape = None,
        preconditioner = None
    ):
    if sample_var_shape is None:
        sample_var_shape = (
            2*prototype_x_flat.shape
            if is_dense(preconditioner) 
            else prototype_x_flat.shape
        )
    return AutoMCMCAdaptStats()._replace(
        sample_mean = jnp.zeros_like(prototype_x_flat),
        sample_var = jnp.zeros_like(prototype_x_flat,shape=sample_var_shape)
    )

def empty_adapt_stats_recorder(adapt_stats):
    return make_adapt_stats_recorder(
        adapt_stats.sample_mean,
        sample_var_shape=jnp.shape(adapt_stats.sample_var)
    )

AutoMCMCStats = namedtuple(
    "AutoStepStats",
    [
        "n_samples",
        "adapt_stats"
    ],
    defaults=(0, AutoMCMCAdaptStats())
)
"""
A :func:`~collections.namedtuple` consisting of the following fields:

 - **n_samples** - total number of calls to ``sample`` so far. At the end of
   a run, this should be equal to `num_warmup+num_samples`.
 - **adapt_stats** - an ``AutoMCMCAdaptStats`` namedtuple which contains adaptation
   information pertaining to the current round.
"""

def make_stats_recorder(prototype_x_flat, preconditioner):
    return AutoMCMCStats()._replace(
        adapt_stats = make_adapt_stats_recorder(
            prototype_x_flat, preconditioner=preconditioner
        )
    )

# update sample statistics
# use the "tracking-error" formula for updating means, based on the identity
#     m_n+1 = (n*m_n + x_{n+1}) / (n+1) = ((n+1)*m_n + (x_{n+1}-m_n)) / (n+1) 
#           = m_n + (x_{n+1}-m_n)/(n+1)
# use Welford's_online_algorithm to update covariance matrices
#     v_n+1 = [n*v_n + (x_n+1 - m_n+1)(x_n+1 - m_n)^T]/(n+1)
def record_post_sample_stats(stats, avg_fwd_bwd_step_size, acc_prob, rev_pass, x_flat):
    # update whole-run statistics
    n_samples, adapt_stats = stats
    n_samples = n_samples + 1

    # update round statistics
    sample_idx, mean_step_size, mean_acc_prob, rev_rate, sample_mean, sample_var = adapt_stats
    sample_idx = sample_idx + 1
    new_mean_step_size = mean_step_size + (avg_fwd_bwd_step_size-mean_step_size)/sample_idx
    new_mean_acc_prob = mean_acc_prob + (acc_prob-mean_acc_prob)/sample_idx
    new_rev_rate = rev_rate + (rev_pass-rev_rate)/sample_idx
    new_sample_mean = sample_mean + (x_flat - sample_mean)/sample_idx
    dvars = delta_vars(sample_var, x_flat - sample_mean, x_flat - new_sample_mean)
    new_sample_var = ((sample_idx-1)*sample_var + dvars) / sample_idx
    return AutoMCMCStats(
        n_samples, 
        AutoMCMCAdaptStats(
            sample_idx, 
            new_mean_step_size, 
            new_mean_acc_prob, 
            new_rev_rate,
            new_sample_mean,
            new_sample_var
        )
    )

def delta_vars(sample_var, dx1, dx2):
    if jnp.ndim(sample_var)==1:
        return dx1 * dx2
    else:
        return jnp.outer(dx1, dx2)


## Update sampler parameters using the adaptation statitics of a round
## See `adapt` method in AutoStep
def update_sampler_params(step_size_selector, args):
    base_step_size, base_precond_state, adapt_stats = args
    n_samples_in_round = adapt_stats.sample_idx

    # set the average step size of the prev round as the new base step size
    # use the same smoothing as for the preconditioner
    new_base_step_size = step_size_selector.adapt_base_step_size(
        base_step_size, adapt_stats.mean_step_size, n_samples_in_round
    )

    # adapt the preconditioner
    new_base_precond_state = adapt_base_precond_state(
        base_precond_state, adapt_stats
    )

    # # debug
    # jax.debug.print(
    #     "End of round {}\nbase_precond_state={}\nadapt_stats={}\nnew_base_precond_state={}",
    #     current_round(n_samples_in_round),
    #     base_precond_state,
    #     adapt_stats,
    #     new_base_precond_state
    # )

    # empty the adapt recorder and return
    new_adapt_stats = empty_adapt_stats_recorder(adapt_stats)
    return (new_base_step_size, new_base_precond_state, new_adapt_stats)
