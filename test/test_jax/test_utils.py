import unittest
from jax.experimental.sparse import BCOO
import jax.numpy as jnp

from probabilistic_model.probabilistic_circuit.jax.utils import copy_bcoo, simple_interval_to_open_array, \
    create_sparse_array_indices_from_row_lengths
from random_events.interval import SimpleInterval

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

    def test_create_sparse_array_indices_from_row_lengths(self):
        row_lengths = jnp.array([2, 3])
        indices = create_sparse_array_indices_from_row_lengths(row_lengths)
        result = jnp.array([[0,0],[0,1],[1, 0],[1, 1],[1, 2]])
        self.assertTrue(jnp.allclose(indices, result))


class IntervalConversionTestCase(unittest.TestCase):

        def simple_interval_to_open_array(self):
            simple_interval = SimpleInterval(0, 1)
            array = simple_interval_to_open_array(simple_interval)
            self.assertTrue(jnp.allclose(array, jnp.array([0, 1])))


if __name__ == '__main__':
    unittest.main()
