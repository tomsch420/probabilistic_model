import unittest

import equinox
import jax
import numpy as np
from jax.experimental.sparse import BCOO
from numpy import dtype
from random_events.interval import closed, singleton
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
import jax.numpy as jnp
from sortedcontainers import SortedSet

from probabilistic_model.probabilistic_circuit.jax import embed_sparse_array_in_nan_array
from probabilistic_model.probabilistic_circuit.jax.input_layer import DiracDeltaLayer
from probabilistic_model.probabilistic_circuit.jax.inner_layer import ProductLayer
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import ProbabilisticCircuit

import warnings
warnings.filterwarnings("ignore")

class DiracProductTestCase(unittest.TestCase):

    x = Continuous("x")
    y = Continuous("y")
    z = Continuous("z")

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

    def test_cdf(self):
        data = jnp.array([[0, 0, 0], [0, 5, 6], [2, 4, 6], [10, 10, 10]], dtype=jnp.float32)
        cdf = self.product_layer.cdf_of_nodes(data)
        self.assertEqual(cdf.shape, (4, 2))
        result = jnp.array([[0, 0], [1., 0.], [0., 1], [1, 1]], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(cdf, result))

    def test_moment(self):
        order = jnp.array([1, 1, 2], dtype=jnp.int32)
        center = jnp.array([0., 1., 2], dtype=jnp.float32)
        moment = self.product_layer.moment_of_nodes(order, center)

        result = jnp.array([[0, 4., 0.],
                            [2., 3., 0.]], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(moment, result))

    def test_probability(self):
        event = SimpleEvent({self.x: closed(0,2), self.y: singleton(5), self.z: singleton(6.)})
        prob = self.product_layer.probability_of_simple_event(event)
        result = jnp.array([1, 0], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(prob, result))

    def test_conditioning(self):

        event = SimpleEvent({self.x: closed(-1, 1),
                             self.y: closed(4.5, 5.5),
                             self.z: closed(5.5, 6.5)})

        conditional, log_prob = self.product_layer.log_conditional_of_simple_event(event)
        conditional.validate()
        self.assertTrue(jnp.allclose(log_prob, jnp.log(jnp.array([1., 0.]))))
        self.assertEqual(conditional.number_of_nodes, 1)
        self.assertEqual(len(conditional.child_layers), 3)
        self.assertEqual(conditional.child_layers[0].number_of_nodes, 1)
        self.assertEqual(conditional.child_layers[1].number_of_nodes, 1)
        self.assertEqual(conditional.child_layers[2].number_of_nodes, 1)


class PCProductLayerTestCase(unittest.TestCase):

    x = Continuous("x")
    y = Continuous("y")
    z = Continuous("z")

    p1_x = DiracDeltaLayer(0, jnp.array([0., 1.]), jnp.array([1, 1]))
    p2_x = DiracDeltaLayer(0, jnp.array([2., 3.]), jnp.array([1, 1]))
    p_y = DiracDeltaLayer(1, jnp.array([4., 5.]), jnp.array([1, 1]))
    p_z = DiracDeltaLayer(2, jnp.array([6.]), jnp.array([1]))
    model: ProbabilisticCircuit

    def setUp(self):
        indices = jnp.array([[0, 0],
                             [1, 0],
                             [3, 0]])
        values = jnp.array([0, 0, 1])
        edges = BCOO((values, indices), shape=(4, 2)).sum_duplicates(remove_zeros=False).sort_indices()
        product_layer = ProductLayer([self.p_z, self.p1_x, self.p2_x, self.p_y, ], edges)
        self.model = ProbabilisticCircuit(SortedSet([self.x, self.y, self.z]), product_layer)

    def test_sample(self):
        samples = self.model.sample(3)
        result = np.array([[0, 5, 6],
                            [0, 5, 6],
                            [0, 5, 6]])
        self.assertTrue(np.allclose(samples, result))


if __name__ == '__main__':
    unittest.main()
