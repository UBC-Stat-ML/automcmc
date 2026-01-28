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
sample: 100%|███| 8190/8190 [00:14<00:00, 584.29it/s, avg_ss=1.2e-01, rr=0.98, ap=0.88, lj=-5.2e+01]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.43      3.28      4.46     -1.09      9.57   4443.32      1.00
       tau      3.55      3.06      2.68      0.30      7.70    177.00      1.00
  theta[0]      6.22      5.55      5.62     -2.43     14.22   1497.87      1.00
  theta[1]      4.90      4.58      4.87     -2.73     11.94   5769.13      1.00
  theta[2]      3.97      5.07      4.19     -3.62     12.50   3606.76      1.00
  theta[3]      4.80      4.64      4.69     -3.21     11.82   5888.14      1.00
  theta[4]      3.66      4.48      3.91     -3.38     11.01   2564.33      1.00
  theta[5]      4.05      4.72      4.19     -3.61     11.49   5252.53      1.00
  theta[6]      6.27      4.92      5.82     -1.49     14.45   1155.05      1.00
  theta[7]      4.82      5.42      4.82     -3.92     12.93   5499.10      1.00
```
In less than 15 seconds, the sampler achieves `r_hat~1` across latent variables,
as well as a minimum effective sample size of over 100.

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
