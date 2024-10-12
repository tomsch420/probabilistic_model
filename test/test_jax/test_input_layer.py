import math
import unittest

import jax.numpy as jnp
import jax
from random_events.interval import closed

from probabilistic_model.probabilistic_circuit.jax.input_layer import DiracDeltaLayer


class DiracDeltaLayerTestCase(unittest.TestCase):
    layer = DiracDeltaLayer(0, location=jnp.array([0., 1.]), density_cap=jnp.array([1., 2.]))

    def test_likelihood(self):
        data = jnp.array([0, 1, 2]).reshape(-1, 1)
        ll = self.layer.log_likelihood_of_nodes(data)
        self.assertEqual(ll.shape, (len(data), self.layer.number_of_nodes))
        result = [[0, -jnp.inf],
                  [-jnp.inf, math.log(2)],
                  [-jnp.inf, -jnp.inf]]
        assert jnp.allclose(ll, jnp.array(result))

    def test_cdf(self):
        data = jnp.array([-1, 0, 1, 2], dtype=jnp.float32).reshape(-1, 1)
        cdf = self.layer.cdf_of_nodes(data)
        result = jnp.array([[0, 0], [1, 0], [1, 1], [1, 1]], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(cdf, result))

    def test_moment(self):
        order = jnp.array([1.], dtype=jnp.int32)
        center = jnp.array([1.5], dtype=jnp.float32)
        moment = self.layer.moment_of_nodes(order, center)
        result = jnp.array([-1.5, -0.5], dtype=jnp.float32).reshape(-1, 1)
        self.assertTrue(jnp.allclose(moment, result))

    def test_conditional_of_simple_interval(self):
        interval = closed(-0.5, 0.5).simple_sets[0]
        layer, ll = self.layer.log_conditional_from_simple_interval(interval)
        result = jnp.log(jnp.array([1, 0], dtype=jnp.float32))
        self.assertTrue(jnp.allclose(ll, result))
        layer.validate()
        self.assertEqual(layer.number_of_nodes, 1)
        self.assertTrue(jnp.allclose(layer.location, jnp.array([0.])))
        self.assertTrue(jnp.allclose(layer.density_cap, jnp.array([1.])))


if __name__ == '__main__':
    unittest.main()
