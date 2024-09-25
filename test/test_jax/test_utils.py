import unittest
from jax.experimental.sparse import BCOO
import jax.numpy as jnp

from probabilistic_model.probabilistic_circuit.jax.utils import copy_bcoo

class BCOOTestCase(unittest.TestCase):

    def test_copy(self):
        x = BCOO.fromdense(jnp.array([[0, 1], [2, 3]]))
        y = copy_bcoo(x)
        self.assertTrue(jnp.allclose(x.todense(), y.todense()))
        x.data += 1
        self.assertFalse(jnp.allclose(x.todense(), y.todense()))
        y.data += 1
        self.assertTrue(jnp.allclose(x.todense(), y.todense()))
        y.data += 1
        self.assertFalse(jnp.allclose(x.todense(), y.todense()))


if __name__ == '__main__':
    unittest.main()
