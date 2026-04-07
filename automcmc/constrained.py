import jax
from jax.typing import ArrayLike
import jax.numpy as jnp

from automcmc import utils

class JacobianOperator:
    """
    Implements several linear algebraic operations associated with :math:`J_x`,
    the Jacobian of a function evaluated at a point :math:`x`.

    Attributes:
        Q: Q factor of the QR decomposition of :math:`J_x^T`.
        log_abs_det: log of :math:`|(J_xJ_x^T)|^{-1/2}` computed using the
            R factor of the QR decomposition.
    """

    def __init__(self, J: ArrayLike):
        """
        Build the operator from the Jacobian evaluated at a point on the
        level set.

        :param J: Jacobian matrix of shape `(m,n)` with `m<n`.
        """
        m, n = jnp.shape(J)
        assert m < n
        Q,R = jnp.linalg.qr(J.T)
        self.Q = Q
        self.log_abs_det = -jnp.log(jnp.abs(jnp.diag(R))).sum()
    
    # project onto normal (N) space
    # P[N]v = J^T(JJ^T)^{-1}Jv = Q(Q^Tv)
    # Cost: O(mn^2 + nm^2)
    def proj_normal(self, v: ArrayLike) -> jax.Array:
        """
        Project a vector onto the normal space at `x`.

        :param v: vector of length `n`.
        :return: normal component of `v`.
        """
        return self.Q @ jnp.dot(v, self.Q)

    # project onto normal (N) and tangent (T) spaces
    # P[T] = v - P[N]v
    # Cost: O(mn^2 + nm^2)
    def proj_normal_tangent(self, v: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """
        Project a vector onto the normal and tangent spaces at `x`.

        :param v: vector of length `n`.
        :return: normal and tangent components of `v`.
        """
        PNv = self.proj_normal(v)
        return (PNv, v - PNv)
