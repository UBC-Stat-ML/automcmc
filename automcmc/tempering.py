from functools import partial

import jax
from jax import lax
from jax import numpy as jnp

from numpyro.handlers import substitute, trace
from numpyro.infer import util
from numpyro.distributions.util import is_identically_one


##############################################################################
# log density evaluation utilities
##############################################################################

# slight modification of numpyro.infer.util.compute_log_probs
def log_prob_site(site):
    value = site["value"]
    intermediates = site["intermediates"]
    scale = site["scale"]
    if intermediates:
        log_prob = site["fn"].log_prob(value, intermediates)
    else:
        guide_shape = jnp.shape(value)
        model_shape = tuple(site["fn"].shape()) # TensorShape from tfp needs casting to tuple
        try:
            lax.broadcast_shapes(guide_shape, model_shape)
        except ValueError:
            raise ValueError(
                "Model and guide shapes disagree at site: '{}': {} vs {}".format(
                    site["name"], model_shape, guide_shape
                )
            )
        log_prob = site["fn"].log_prob(value)

    log_prob_sum = jnp.sum(log_prob)

    if (scale is not None) and (not is_identically_one(scale)):
        log_prob_sum = scale * log_prob_sum

    return log_prob_sum

def is_observation(site):
    return (
        site["is_observed"] and (not site['infer'].get('is_auxiliary', False))
    )

def log_prior(model_trace):
    return sum(
        log_prob_site(site)
        for site in model_trace.values()
        if site["type"] == "sample" and (not is_observation(site))
    )

def log_lik(model_trace):
    return sum(
        log_prob_site(site)
        for site in model_trace.values()
        if site["type"] == "sample" and is_observation(site)
    )

# full trace from unconstrained sample
def trace_from_unconst_samples(
        model,
        model_args,
        model_kwargs,
        unconstrained_sample,
    ):
    """
    Generate a full model trace from a sample in unconstrained space.

    :param model: A numpyro model.
    :param model_args: Model arguments.
    :param model_kwargs: Model keyword arguments.
    :param unconstrained_sample: A sample from the model in unconstrained space.
    :return: A trace.
    """
    substituted_model = substitute(
        model, substitute_fn=partial(
            util._unconstrain_reparam,
            unconstrained_sample
        )
    )
    return trace(substituted_model).get_trace(*model_args, **model_kwargs)

# tempered potential of a model
def model_logprior_and_loglik(model, model_args, model_kwargs, unconstrained_sample):
    """
    Compute the log-prior and log-likelihood of a model at a specified
    unconstrained sample.

    :param model: A numpyro model.
    :param model_args: Model arguments.
    :param model_kwargs: Model keyword arguments.
    :param unconstrained_sample: A sample from the model in unconstrained space.
    :return: The log-prior and log-likelihood evaluated at the given sample.
    """
    model_trace = trace_from_unconst_samples(
        model,
        model_args,
        model_kwargs,
        unconstrained_sample
    )
    return (log_prior(model_trace), log_lik(model_trace))

# tempered potential of a model
def tempered_potential_from_logprior_and_loglik(log_prior, log_lik, inv_temp):
    """
    Calculate a tempered potential. Specifically,

    .. code-block:: python
        tempered_potential(x) = -log(pi_beta(x))

    where `beta` is the inverse temperature, and

    .. code-block:: python
        pi_beta(x) = prior(x) * likelihood(x) ** beta

    Hence, when `inv_temp=0`, the tempered model reduces to the prior, whereas
    for `inv_temp=1`, the original posterior distribution is recovered.

    :param log_prior: Log-prior value.
    :param log_lik: Log-likelihood value.
    :param inv_temp: An inverse temperature (non-negative number).
    :return: The tempered potential.
    """
    if inv_temp is None:
        return -(log_prior + log_lik) # default to log posterior
    else:
        # avoid nan by enforcing 0*inf=0
        actual_zero = jnp.zeros_like(inv_temp)
        scaled_loglik = jnp.where(
            inv_temp == actual_zero,
            actual_zero,
            inv_temp*log_lik
        )
        return -(log_prior + scaled_loglik)

# shortcut utility
def tempered_potential(logprior_and_loglik, params, inv_temp):
    return tempered_potential_from_logprior_and_loglik(
        *logprior_and_loglik(params), inv_temp
    )
