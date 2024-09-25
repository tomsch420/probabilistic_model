import unittest
from probabilistic_model.probabilistic_circuit.jax.distributions import DiracDeltaLayer
import jax.numpy as jnp
import math


class DiracDeltaLayerTestCase(unittest.TestCase):

    layer = DiracDeltaLayer(location=jnp.array([0., 1.]), density_cap=jnp.array([1., 2.]))

    def test_likelihood(self):
        data = jnp.array([0, 1, 2]).reshape(-1, 1)
        ll = self.layer.log_likelihood_of_nodes(data)
        self.assertEqual(ll.shape, ((len(data), self.layer.number_of_nodes)))
        result = [[0, -jnp.inf],
                  [-jnp.inf, math.log(2)],
                  [-jnp.inf, -jnp.inf]]
        assert jnp.allclose(ll, jnp.array(result))


if __name__ == '__main__':
    unittest.main()
