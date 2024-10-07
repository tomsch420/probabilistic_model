import unittest

import equinox
import jax
from jax.experimental.sparse import BCOO
from numpy import dtype
from random_events.variable import Continuous
import jax.numpy as jnp

from probabilistic_model.probabilistic_circuit.jax import embed_sparse_array_in_nan_array
from probabilistic_model.probabilistic_circuit.jax.input_layer import DiracDeltaLayer
from probabilistic_model.probabilistic_circuit.jax.inner_layer import ProductLayer


class DiracProductTestCase(unittest.TestCase):

    p1_x = DiracDeltaLayer(0, jnp.array([0., 1.]), jnp.array([1, 1]))
    p2_x = DiracDeltaLayer(0, jnp.array([2., 3.]), jnp.array([1, 1]))
    p_y = DiracDeltaLayer(1, jnp.array([4., 5.]), jnp.array([1, 1]))
    p_z = DiracDeltaLayer(2, jnp.array([6.]), jnp.array([1]))
    product_layer: ProductLayer

    def setUp(self):
        indices = jnp.array([[0, 0],
                             [0, 1],
                             [1, 0],
                             [2, 1],
                             [3, 0],
                             [3, 1]])
        values = jnp.array([0, 0, 0, 0, 1, 0])
        edges = BCOO((values, indices), shape=(4, 2)).sum_duplicates(remove_zeros=False).sort_indices()
        self.product_layer = ProductLayer([self.p_z, self.p1_x, self.p2_x, self.p_y, ], edges)


    def test_variables(self):
        self.assertTrue(jnp.allclose(self.product_layer.variables, jnp.array([0, 1, 2])))

    def test_likelihood(self):
        data = jnp.array([[0., 5., 6.],
                             [2, 4, 6]])
        likelihood = self.product_layer.log_likelihood_of_nodes(data)
        self.assertTrue(likelihood[0, 0] > -jnp.inf)
        self.assertTrue(likelihood[1, 1] > -jnp.inf)
        self.assertTrue(likelihood[0, 1] == -jnp.inf)
        self.assertTrue(likelihood[1, 0] == -jnp.inf)

    def test_sample_from_frequencies(self):
        frequencies = jnp.array([5, 3])

        samples = self.product_layer.sample_from_frequencies(frequencies, jax.random.PRNGKey(69))

        samples_n0 = samples[0].todense()
        samples_n1 = samples[1].todense()

        self.assertEqual(samples_n0.shape, (5, 3))
        self.assertEqual(samples_n1.shape, (5, 3))
        self.assertTrue(jnp.all(samples_n1[3:] == 0))
        self.assertTrue(jnp.all(samples_n0 == jnp.array([0, 5 ,6])))
        self.assertTrue(jnp.all(samples_n1[:3] == jnp.array([2, 4 ,6])))

        # s = equinox.filter_jit(self.product_layer.sample_from_frequencies)
        # s(frequencies, jax.random.PRNGKey(69))

    def test_cdf(self):
        data = jnp.array([[0, 0, 0], [0, 5, 6], [2, 4, 6], [10, 10, 10]], dtype=jnp.float32)
        cdf = self.product_layer.cdf_of_nodes(data)
        self.assertEqual(cdf.shape, (4, 2))
        result = jnp.array([[0, 0], [1., 0.], [0., 1], [1, 1]], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(cdf, result))


if __name__ == '__main__':
    unittest.main()
