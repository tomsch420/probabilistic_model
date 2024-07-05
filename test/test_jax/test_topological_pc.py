import unittest

import jax.random

from flax.linen.summary import tabulate

from probabilistic_model.learning.jax.topological_pc import *


class TPCTestCase(unittest.TestCase):

    data: jax.Array

    @classmethod
    def setUpClass(cls):
        cls.key = jax.random.PRNGKey(0)
        cls.data = jax.random.normal(cls.key, (100, 2))
        means = jnp.array([0.0, 1.0])
        scales = jnp.log(jnp.array([1.0, 1.0]))
        cls.input_layer = NormalLayer(means, scales)

    def test_sum_layer(self):
        sum_layer = SumLayer(self.input_layer, edge_mask=jnp.array([[True, True]]),
                             log_weights=jnp.array([[0.0, 0.0]]))

        log_likelihoods = sum_layer.log_likelihood(self.data)
        self.assertEqual(log_likelihoods.shape, (100, 2))

    def test_product_layer(self):
        product_layer = ProductLayer(self.input_layer, edge_mask=jnp.array([[True, True]]))

        log_likelihoods = product_layer.log_likelihood(self.data)
        self.assertEqual(log_likelihoods.shape, (100,))

if __name__ == '__main__':
    unittest.main()
