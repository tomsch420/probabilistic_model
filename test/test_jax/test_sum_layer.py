import unittest

from jax.experimental.sparse import BCOO
from random_events.variable import Continuous
import jax.numpy as jnp
from probabilistic_model.probabilistic_circuit.jax.input_layer import DiracDeltaLayer
from probabilistic_model.probabilistic_circuit.jax.inner_layer import SumLayer


class DiracSumUnitTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")

    p1_x = DiracDeltaLayer(0, jnp.array([0., 1.]), jnp.array([1, 2]))
    p2_x = DiracDeltaLayer(0,jnp.array([2.]), jnp.array([3]))
    p3_x = DiracDeltaLayer(0, jnp.array([3, 4, 5]), jnp.array([4, 5, 6]))
    p4_x = DiracDeltaLayer(0, jnp.array([6.]), jnp.array([1]))
    sum_layer: SumLayer

    @classmethod
    def setUpClass(cls):
        weights_p1 = BCOO.fromdense(jnp.array([[0, 0.1], [0.4, 0]])) * 2
        weights_p1.data = jnp.log(weights_p1.data)

        weights_p2 = BCOO.fromdense(jnp.array([[0.2], [0.3]])) * 2
        weights_p2.data = jnp.log(weights_p2.data)

        weights_p3 = BCOO.fromdense(jnp.array([[0.3, 0, 0.4], [0., 0.1, 0.2]])) * 2
        weights_p3.data = jnp.log(weights_p3.data)

        weights_p4 = BCOO.fromdense(jnp.array([[0], [0]])) * 2
        weights_p4.data = jnp.log(weights_p4.data)

        cls.sum_layer = SumLayer([cls.p1_x, cls.p2_x, cls.p3_x, cls.p4_x],
                                 log_weights=[weights_p1, weights_p2, weights_p3, weights_p4])
        cls.sum_layer.validate()

    def test_normalization_constants(self):
        log_normalization_constants = self.sum_layer.log_normalization_constants
        result = jnp.log(jnp.array([2, 2]))
        self.assertTrue(jnp.allclose(log_normalization_constants, result))

    def test_normalized_weights(self):
        normalized_weights = self.sum_layer.normalized_weights.todense()
        result = jnp.array([[0, 0.1, 0.2, 0.3, 0, 0.4, 0],
                            [0.4, 0, 0.3, 0., 0.1, 0.2, 0]])
        self.assertTrue(jnp.allclose(normalized_weights, result))

    def test_ll(self):
        data = jnp.array([0., 1., 2., 3., 4., 5., 6.]).reshape(-1, 1)
        # l = self.sum_layer.log_likelihood_of_nodes_single(data[0])

        ll = self.sum_layer.log_likelihood_of_nodes(data)
        result = jnp.log(jnp.array([[0., 0.4,],
                               [0.1 * 2, 0.,],
                               [0.2 * 3, 0.3 * 3,],
                               [0.3 * 4, 0.,],
                               [0., 0.1 * 5,],
                               [0.4 * 6, 0.2 * 6,],
                               [0., 0.,]]))
        assert jnp.allclose(ll, result)
