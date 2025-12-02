[![Build Status](https://github.com/UBC-Stat-ML/automcmc/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UBC-Stat-ML/automcmc/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/UBC-Stat-ML/automcmc/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-Stat-ML/automcmc)

# `automcmc`

*A NumPyro-compatible JAX implementation of AutoStep and other automatically tuned samplers.*

## Installation

```bash
pip install "automcmc @ git+https://github.com/UBC-Stat-ML/automcmc.git"
```

## Eight-schools example

We apply autoHMC to the classic toy eight schools problem. We use all default
settings (32 leapfrog steps, `DeterministicSymmetricSelector` for the step
size adaptation critetion), except for the preconditioner. Since the problem
is low dimensional, we can afford to use a full dense mass matrix to drastically
improve the conditioning of the target.
```python
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC
from automcmc import preconditioning
from automcmc.autohmc import AutoHMC
from automcmc import utils

# define model
y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

def eight_schools(sigma, y=None):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('J', len(sigma)):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))
        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

# instantiate sampler and run
n_rounds = 12
n_warmup, n_keep = utils.split_n_rounds(n_rounds) # translate rounds to warmup/keep
kernel = AutoHMC(
    eight_schools,
    preconditioner = preconditioning.FixedDensePreconditioner()
)
mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_keep)
mcmc.run(random.key(9), sigma, y=y)
mcmc.print_summary()
```
```
sample: 100%|███| 8190/8190 [00:13<00:00, 614.36it/s, base_step 1.01e-01, rev_rate=0.96, acc_prob=0.90]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.38      3.40      4.36     -1.31      9.54  11296.80      1.00
       tau      3.67      3.23      2.73      0.08      7.85    224.57      1.00
  theta[0]      6.28      5.75      5.74     -2.11     15.37   1661.16      1.00
  theta[1]      4.87      4.76      4.91     -4.28     11.59   7753.07      1.00
  theta[2]      3.90      5.48      4.18     -4.87     12.24   4461.17      1.00
  theta[3]      4.83      4.99      4.83     -3.90     11.89   6988.49      1.00
  theta[4]      3.60      4.72      3.76     -4.05     11.02   3519.48      1.00
  theta[5]      3.92      4.88      4.14     -3.75     11.89   4680.06      1.00
  theta[6]      6.30      5.13      5.85     -1.38     14.61   1916.97      1.00
  theta[7]      4.89      5.22      4.81     -3.07     13.13   5879.55      1.00
```
In less than 15 seconds, the sampler achieves `r_hat~1` across latent variables,
as well as a minimum effective sample size of 224.57.

## TODO

- autoHMC with randomized number of steps (RHMC)
- Re-implement the `MixDiagonalPreconditioner` in the new framework

## References

Biron-Lattes, M., Surjanovic, N., Syed, S., Campbell, T., & Bouchard-Côté, A.. (2024). 
[autoMALA: Locally adaptive Metropolis-adjusted Langevin algorithm](https://proceedings.mlr.press/v238/biron-lattes24a.html).
*Proceedings of The 27th International Conference on Artificial Intelligence and Statistics*, 
in *Proceedings of Machine Learning Research* 238:4600-4608.

Liu, T., Surjanovic, N., Biron-Lattes, M., Bouchard-Côté, A., & Campbell, T. (2025). 
[AutoStep: Locally adaptive involutive MCMC](https://proceedings.mlr.press/v267/liu25br.html).
*Proceedings of the 42nd International Conference on Machine Learning*,
in *Proceedings of Machine Learning Research* 267:39624-39650.
