import unittest

import jax.random
from jax.experimental.sparse import BCOO, BCSR
import jax.numpy as jnp
from sympy.physics.quantum.matrixutils import sparse
from sympy.physics.units import amount

from probabilistic_model.probabilistic_circuit.jax import embed_sparse_array_in_nan_array, \
    sample_from_sparse_probabilities_bcsr, create_bcsr_indices_from_row_lengths
from probabilistic_model.probabilistic_circuit.jax.utils import copy_bcoo, simple_interval_to_open_array, \
    create_bcoo_indices_from_row_lengths, sample_from_sparse_probabilities
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
        indices = create_bcoo_indices_from_row_lengths(row_lengths)
        result = jnp.array([[0,0],[0,1],[1, 0],[1, 1],[1, 2]])
        self.assertTrue(jnp.allclose(indices, result))

    def test_create_bcsr_indices_from_row_lengths(self):
        row_lengths = jnp.array([2, 3])
        column_indices, indent_pointer = create_bcsr_indices_from_row_lengths(row_lengths)

        bcsr = BCSR.fromdense(
            jnp.array([[1, 1, 0], [1, 1, 1]])
        )

        self.assertTrue(jnp.allclose(bcsr.indices, column_indices))
        self.assertTrue(jnp.allclose(bcsr.indptr, indent_pointer))

    def test_sample_from_sparse_probabilities(self):
        probs = BCOO.fromdense(jnp.array([[0.1, 0.2, 0., .7],
                                            [0.4, 0., 0.6, 0.]]))
        probs.data = jnp.log(probs.data)
        amount = jnp.array([2, 3])

        samples = sample_from_sparse_probabilities(probs,amount, jax.random.PRNGKey(69))
        amounts = samples.sum(axis=1).todense()
        self.assertTrue(jnp.all(amounts == amount))
        self.assertTrue(jnp.all(samples.data <= 3))

    # def test_sample_from_sparse_probabilities_bcsr(self):
    #     probs = BCOO.fromdense(jnp.array([[0.1, 0.2, 0., .7],
    #                                       [0.4, 0., 0.6, 0.]]))
    #     bcoo_indices = probs.indices
    #     probs = BCSR.from_bcoo(probs)
    #     amount = jnp.array([2, 3])
    #
    #     samples = sample_from_sparse_probabilities_bcsr(probs, bcoo_indices, amount, jax.random.PRNGKey(69))
    #     amounts = samples.sum(axis=1).todense()
    #     self.assertTrue(jnp.all(amounts == amount))
    #     self.assertTrue(jnp.all(samples.data <= 3))




class IntervalConversionTestCase(unittest.TestCase):

        def simple_interval_to_open_array(self):
            simple_interval = SimpleInterval(0, 1)
            array = simple_interval_to_open_array(simple_interval)
            self.assertTrue(jnp.allclose(array, jnp.array([0, 1])))


if __name__ == '__main__':
    unittest.main()
