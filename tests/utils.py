import numpy
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

###############################################################################
# toy examples
###############################################################################

def gaussian_potential(x):
    return ((x - 2) ** 2).sum()

def make_const_off_diag_corr_mat(dim, rho):
    return jnp.full((dim,dim), rho) + (1-rho)*jnp.eye(dim, dim)

def make_correlated_Gaussian_potential(S=None, dim=None, rho=None):
    S = make_const_off_diag_corr_mat(dim, rho) if S is None else S
    P = jnp.linalg.inv(S)
    def pot_fn(x):
        return 0.5*jnp.dot(x.T, jnp.dot(P, x))
    return pot_fn

def toy_unid(n_flips, n_heads=None):
    p1 = numpyro.sample('p1', dist.Uniform())
    p2 = numpyro.sample('p2', dist.Uniform())
    p = p1*p2
    numpyro.sample('n_heads', dist.Binomial(n_flips, p), obs=n_heads)

# Toy conjugate Gaussian example, admits closed form tempering path
# For d in ints, m in R, sigma0 >0, and beta>=0,
#   x ~ pi_beta = N_d(mu(beta), v(beta))
# where
#   mu(beta) := beta m v(beta)
#   v(beta)  := (beta + sigma0^{-2})^{-1}
#
# Ref: Biron-Lattes, Campbell, & Bouchard-Côté (2024)
def toy_conjugate_normal(
        d = jnp.int32(3),
        m = jnp.float32(2.),
        sigma0 = jnp.float32(2.)
    ):
    def model(sigma0, y):
        with numpyro.plate('dim', len(y)):
            x = numpyro.sample('x', dist.Normal(scale=sigma0))
            numpyro.sample('obs', dist.Normal(x), obs=y)

    # inputs
    y = jnp.full((d,), m)
    model_args = (sigma0, y)
    model_kwargs = {}
    return model, model_args, model_kwargs

def make_eight_schools():
    y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    def eight_schools(sigma, y=None):
        mu = numpyro.sample('mu', dist.Normal(0, 5))
        tau = numpyro.sample('tau', dist.HalfCauchy(5))
        with numpyro.plate('J', len(sigma)):
            theta = numpyro.sample('theta', dist.Normal(mu, tau))
            numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

    model=eight_schools
    model_args = (sigma,)
    model_kwargs = {'y': y}
    return model, model_args, model_kwargs

#######################################
# constrained problems
#######################################

####### T^2 torus embedded in R^3 #####
# Example 1 in Zappa & Holmes-Cerfon (2018)
#   F(x,y,z)=(R - sqrt{x^2+y^2})^2 + z^2 - r^2
#   dF/dx = 2x(R - sqrt{x^2+y^2})/sqrt{x^2+y^2} = 2x(R/sqrt{x^2+y^2} - 1)
#   dF/dy = 2y(R/sqrt{x^2+y^2} - 1) // symmetry
#   dF/dz = 2z
# => JJ^T = 4[x^2+y^2](R/sqrt{x^2+y^2} - 1)^2 + 4z^2
#   = 4(R - sqrt{x^2+y^2})^2 + 4z^2
#   = 4[F(x,y,z)+r^2]
# So when F(x,y,z)=0 and also r=1/2, we have
#   JJ^T = 4/4 = 1 => |JJ^T|^{-1/2} = 1 => log(|JJ^T|^{-1/2}) = 0
def torus_constraint(R, r, x):
    return jnp.array([
        jnp.square(R - jnp.linalg.norm(x[:-1])) + x[-1]*x[-1] - r*r
    ])

def torus_param(R, r, theta, phi):
    cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
    cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
    u = R + r*cos_phi
    return jnp.array([u*cos_theta, u*sin_theta, r*sin_phi])

# need arctan2 to return angles in [0,2pi)
arctan2pi = lambda x,y: jnp.remainder(jnp.arctan2(x,y), 2*jnp.pi)
def inv_torus_param(R, r, x, y, z):
    theta = arctan2pi(y, x)
    d = jnp.hypot(x,y) - R
    phi = arctan2pi(z, d)
    return (theta, phi)

###### cone embedded in R^3 #####
# Example 2 in Zappa & Holmes-Cerfon (2018)
# inequality constraints passed through pontential fn
#   F(x,y,z) = z - sqrt(x^2 + y^2)
#   dF/dx = x(x^2 + y^2)^{-1/2}
#   dF/dy = y(x^2 + y^2)^{-1/2}
#   dF/dz =  1
#    JJ^T = 1 + x^2(x^2 + y^2)^{-1} + y^2(x^2 + y^2)^{-1}
#      = 1 + (x^2 + y^2)(x^2 + y^2)^{-1}
#      = 1 + 1
#      = 2
#   |JJ^T|^{-1/2} = 2^{-1/2} => log(|JJ^T|^{-1/2}) = -0.5log(2)
# It follows that the ambient uniform induces the uniform distribution on each
# level set (i.e., each cone)
# Note: this wouldn't be the case if we used the alternative function
#   G(x,y,z) = z^2 - x^2 + y^2
# even though G^{-1}({0})=F^{-1}({0})! This is because the general equality is
#
#   for all r: G^{-1}({r})=F^{-1}({r^2})
# Nevertheless, the total integral int dr int_cone(r) must end up being the
# same since the LHS int_{R^3} f(x) dx is invariant to the choice of F,G.
def cone_constraint(x):
    return jnp.array([x[-1] - jnp.linalg.norm(x[:-1])])
    # return jnp.array([x[-1]*x[-1] - jnp.square(x[:-1]).sum()])

def cone_potential(x):
    return jnp.where(
        jnp.logical_and(jnp.square(x[:-1]).sum()<=1, x[-1] >= 0),
        jnp.zeros_like(x, shape=()),
        jnp.inf,
    )

###############################################################################
# diagnostics
###############################################################################

def extremal_diagnostics(mcmc):
    """
    Compute maximum of the Gelman--Rubin diagnostic across dimensions, and the
    minimum ESS across dimensions.

    :param mcmc: An instance of `numpyro.infer.MCMC`.
    :return: Worst-case Gelman--Rubin (max) and ESS (min) across dimensions.
    """
    grouped_samples = mcmc.get_samples(group_by_chain=True)
    diags = numpyro.diagnostics.summary(grouped_samples)
    max_grd = next(numpy.max(v["r_hat"].max() for v in diags.values()))
    min_ess = next(numpy.min(v["n_eff"].min() for v in diags.values()))
    return (max_grd, min_ess)
