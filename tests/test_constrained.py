import unittest
import jax
from jax import numpy as jnp

from automcmc import constrained

class TestConstrained(unittest.TestCase):

    def test_jac_op(self):
        m = 6
        n = 145
        J_key, v_key = jax.random.split(jax.random.key(1),2)
        v = jax.random.normal(v_key, (n,))
        J = jax.random.normal(J_key, (m,n))
        tol = jnp.sqrt(jnp.finfo(J.dtype).eps)
        jop = constrained.JacobianOperator(J)
        assert jnp.isclose(1, jnp.linalg.cond(jop.Q))
        assert jnp.linalg.cond(J.T) > 1.1
        assert jnp.abs(jop.Q.T@jop.Q - jnp.identity(m)).max() < tol
        assert jnp.isclose(
            jop.log_abs_det,
            -0.5*jnp.log(jnp.abs(jnp.linalg.det(jnp.inner(J,J)))),
            rtol = 0.01
        )
        _,PTv = jop.proj_normal_tangent(v)
        assert jnp.abs(J@PTv).max() < tol # PTv should be orthogonal to every row of J


if __name__ == '__main__':
    unittest.main()