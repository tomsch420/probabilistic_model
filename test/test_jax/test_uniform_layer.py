import json
import math
import unittest

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from random_events.interval import SimpleInterval, Bound, closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous

from probabilistic_model.probabilistic_circuit.jax import simple_interval_to_open_array, SumLayer
from probabilistic_model.probabilistic_circuit.jax.uniform_layer import UniformLayer
import equinox as eqx


class UniformLayerTestCaste(unittest.TestCase):
    x = Continuous("x")
    p_x = UniformLayer(0, jnp.array([[0, 1], [1, 3]]))
    key = jax.random.PRNGKey(69)

    def test_log_likelihood(self):
        data = jnp.array([0.5, 1.5, 4]).reshape(-1, 1)
        ll = self.p_x.log_likelihood_of_nodes(data)
        self.assertEqual(ll.shape, (3, 2))
        result = [[0., -float("inf")], [-float("inf"), -math.log(2)], [-float("inf"), -float("inf")]]
        self.assertTrue(jnp.allclose(ll, jnp.array(result)))

    def test_from_interval(self):
        ioo = SimpleInterval(0, 1)
        ioc = SimpleInterval(0, 1, right=Bound.CLOSED)
        ico = SimpleInterval(0, 1, left=Bound.CLOSED)
        icc = SimpleInterval(0, 1, left=Bound.CLOSED, right=Bound.CLOSED)

        intervals = jnp.vstack([simple_interval_to_open_array(i) for i in [ioo, ioc, ico, icc]])
        p_x = UniformLayer(0, intervals)

        data = jnp.array([[0], [1]])
        ll = jnp.exp(p_x.log_likelihood_of_nodes(data))
        result = jnp.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        self.assertTrue(jnp.allclose(ll, result))

    def test_cdf(self):
        data = jnp.array([0.5, 1.5, 4]).reshape(-1, 1)
        cdf = self.p_x.cdf_of_nodes(data)
        self.assertEqual(cdf.shape, (3, 2))
        result = jnp.array([[0.5, 0], [1, 0.25], [1, 1]])
        self.assertTrue(jnp.allclose(cdf, result))

    def test_moment(self):
        order = jnp.array([1], dtype=jnp.int32)
        center = jnp.array([1.], dtype=jnp.float32)
        moment = self.p_x.moment_of_nodes(order, center)
        result = jnp.array([[-0.5], [1.]], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(moment, result))

    def test_probability(self):
        event = SimpleEvent({self.x: closed(0.5, 2.5) | closed(3, 5)})
        prob = self.p_x.probability_of_simple_event(event)
        self.assertEqual(prob.shape, (2,))
        result = jnp.array([0.5, 0.75])
        self.assertTrue(jnp.allclose(prob, result))

    def test_to_json(self):
        data = self.p_x.to_json()
        json.dumps(data)
        p_x = UniformLayer.from_json(data)

        self.assertTrue(jnp.allclose(self.p_x.interval, p_x.interval))

    def test_conditional_singleton(self):
        event = SimpleEvent({self.x: closed(0.5, 0.5)})
        layer, ll = self.p_x.log_conditional_of_simple_event(event)
        self.assertEqual(layer.number_of_nodes, 1)
        self.assertTrue(jnp.allclose(jnp.array([0.5]), layer.location))
        self.assertTrue(jnp.allclose(jnp.array([1.]), layer.density_cap))

    def test_conditional_single_truncation(self):
        event = SimpleEvent({self.x: closed(0.5, 2.5)})
        layer, ll = self.p_x.log_conditional_of_simple_event(event)
        layer.validate()
        self.assertEqual(layer.number_of_nodes, 2)
        self.assertTrue(jnp.allclose(layer.interval, jnp.array([[0.5, 1], [1, 2.5]])))
        self.assertTrue(jnp.allclose(jnp.log(jnp.array([0.5, 0.75])), ll))

    def test_conditional_with_node_removal(self):
        event = SimpleEvent({self.x: closed(0.25, 0.5)})
        layer, ll = self.p_x.log_conditional_of_simple_event(event)
        layer.validate()
        self.assertEqual(layer.number_of_nodes, 1)
        self.assertTrue(jnp.allclose(layer.interval, jnp.array([[0.25, 0.5]])))
        self.assertTrue(jnp.allclose(jnp.log(jnp.array([0.25, 0.])), ll))

    def test_conditional_multiple_truncation(self):
        event = closed(-1, 0.5) | closed(0.7, 0.8) | closed(2., 3.) | closed(3.5, 4.)

        layer, ll = self.p_x.log_conditional_from_interval(event)
        self.assertTrue(jnp.allclose(jnp.log(jnp.array([0.6, 0.5])), ll))
        self.assertIsInstance(layer, SumLayer)

        layer.validate()
        self.assertEqual(layer.number_of_nodes, 2)
        self.assertEqual(len(layer.child_layers), 1)
        self.assertTrue(jnp.allclose(layer.child_layers[0].interval, jnp.array([[0., 0.5], [0.7, 0.8], [2., 3.]])))

        log_weights_by_hand = BCOO.fromdense(jnp.array([[0.5, 0.1, 0.], [0., 0., 0.5]]))
        log_weights_by_hand.data = jnp.log(log_weights_by_hand.data)
        self.assertTrue(jnp.allclose(layer.log_weights[0].data, log_weights_by_hand.data))
        self.assertTrue(jnp.allclose(layer.log_weights[0].indices, log_weights_by_hand.indices))
