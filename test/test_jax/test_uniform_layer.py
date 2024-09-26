import math
import unittest

import jax.numpy as jnp

from probabilistic_model.probabilistic_circuit.jax.uniform_layer import UniformLayer


class UniformLayerTestCaste(unittest.TestCase):
    p_x = UniformLayer(0, jnp.array([[0, 1], [1, 3]]))

    def test_log_likelihood(self):
        data = jnp.array([0.5, 1.5, 4]).reshape(-1, 1)
        ll = self.p_x.log_likelihood_of_nodes(data)
        self.assertEqual(ll.shape, (3, 2))
        result = [[0., -float("inf")], [-float("inf"), -math.log(2)], [-float("inf"), -float("inf")]]
        self.assertTrue(jnp.allclose(ll, jnp.array(result)))
