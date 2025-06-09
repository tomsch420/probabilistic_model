import copy
import unittest

import numpy as np
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous

from probabilistic_model.probabilistic_circuit.tensor_graph.probabilistic_circuit import *

np.random.seed(69)


class SumLayerTestCase(unittest.TestCase):
    x = Continuous("x")

    sum_layer = SumLayer()

    @classmethod
    def setUpClass(cls):
        model = ProbabilisticCircuit()
        p1_x = DiracDeltaLayer(cls.x, np.array([0., 1.]), np.array([1, 2]), probabilistic_circuit=model)
        p2_x = DiracDeltaLayer(cls.x, np.array([2.]), np.array([3]), probabilistic_circuit=model)
        p3_x = DiracDeltaLayer(cls.x, np.array([3., 4., 5.]), np.array([4, 5, 6]), probabilistic_circuit=model)
        p4_x = DiracDeltaLayer(cls.x, np.array([6.]), np.array([1]), probabilistic_circuit=model)

        sum_layer = SumLayer(probabilistic_circuit=model)

        weights_p1 = np.array([[0, 0.1], [0.4, 0]]) * 2
        weights_p1 = np.log(weights_p1)
        sum_layer.add_child_layer(p1_x, weights_p1)

        weights_p2 = np.array([[0.2], [0.3]]) * 2
        weights_p2 = np.log(weights_p2)
        sum_layer.add_child_layer(p2_x, weights_p2)

        weights_p3 = np.array([[0.3, 0, 0.4], [0., 0.1, 0.2]]) * 2
        weights_p3 = np.log(weights_p3)
        sum_layer.add_child_layer(p3_x, weights_p3)

        weights_p4 = np.array([[0], [0]]) * 2
        weights_p4 = np.log(weights_p4)
        sum_layer.add_child_layer(p4_x, weights_p4)

        cls.sum_layer = sum_layer


    def test_normalization_constants(self):
        self.assertEqual(len(self.sum_layer.log_weights), 4)
        self.assertEqual(self.sum_layer.concatenated_log_weights.shape, (2, 7))
        log_normalization_constants = self.sum_layer.log_normalization_constants
        result = np.log(np.array([2, 2]))
        self.assertTrue(np.allclose(log_normalization_constants, result))

    def test_log_conditional(self):
        condition = SimpleEvent({self.x: closed(1, 4)})
        conditional, log_prob = self.sum_layer.probabilistic_circuit.log_conditional(condition.as_composite_set())
        assert np.allclose(log_prob, np.log(np.array([0.6, 0.4])))

    # def test_normalized_weights(self):
    #     normalized_weights = self.sum_layer.normalized_weights.todense()
    #     result = jnp.array([[0, 0.1, 0.2, 0.3, 0, 0.4, 0],
    #                         [0.4, 0, 0.3, 0., 0.1, 0.2, 0]])
    #     self.assertTrue(jnp.allclose(normalized_weights, result))
    #
    # def test_ll(self):
    #     data = jnp.array([0., 1., 2., 3., 4., 5., 6.]).reshape(-1, 1)
    #     # l = self.sum_layer.log_likelihood_of_nodes_single(data[0])
    #
    #     ll = self.sum_layer.log_likelihood_of_nodes(data)
    #     result = jnp.log(jnp.array([[0., 0.4,],
    #                            [0.1 * 2, 0.,],
    #                            [0.2 * 3, 0.3 * 3,],
    #                            [0.3 * 4, 0.,],
    #                            [0., 0.1 * 5,],
    #                            [0.4 * 6, 0.2 * 6,],
    #                            [0., 0.,]]))
    #     assert jnp.allclose(ll, result)
    #
    # def test_ll_single(self):
    #     data = jnp.array([0])
    #     l = self.sum_layer.log_likelihood_of_nodes_single(data)
    #     result = jnp.log(jnp.array([0., 0.4]))
    #     assert jnp.allclose(l, result)
    #
    # def test_set_variables(self):
    #     self.sum_layer.reset_variables()
    #     self.assertEqual(self.sum_layer.variables.item(), 0)


class DiracProductTestCase(unittest.TestCase):

    x = Continuous("x")
    y = Continuous("y")
    z = Continuous("z")

    @classmethod
    def setUpClass(cls):
        model = ProbabilisticCircuit()
        cls.product_layer = ProductLayer(probabilistic_circuit=model)
        p1_x = DiracDeltaLayer(cls.x, np.array([0., 1.]), np.array([1, 1]))
        p2_x = DiracDeltaLayer(cls.y, np.array([2., 3.]), np.array([1, 1]))
        p_y = DiracDeltaLayer(cls.y, np.array([4., 5.]), np.array([1, 1]))
        p_z = DiracDeltaLayer(cls.z, np.array([6.]), np.array([1]))

        edges_p1_x = np.array([[True, False],[True, False]])
        cls.product_layer.add_child_layer(p1_x, edges_p1_x)

        edges_p2_x = np.array([[True, False],[False, False]])
        cls.product_layer.add_child_layer(p2_x, edges_p2_x)

        edges_p_y = np.array([[False, False],[True, False]])
        cls.product_layer.add_child_layer(p_y, edges_p_y)

        edges_p_z = np.array([[False, True],[True, False]])
        cls.product_layer.add_child_layer(p_z, edges_p_z)

    def test_variables(self):
        self.assertEqual(self.product_layer.variables, SortedSet([self.x, self.y, self.z]))

    def test_likelihood(self):
        data = np.array([[0., 5., 6.],
                             [2, 4, 6]])
        likelihood = self.product_layer.log_likelihood_of_nodes(data)
        self.assertTrue(likelihood[0, 0] > -np.inf)
        self.assertTrue(likelihood[1, 1] > -np.inf)
        self.assertTrue(likelihood[0, 1] == -np.inf)
        self.assertTrue(likelihood[1, 0] == -np.inf)

class DiracLayerTestCase(unittest.TestCase):
    x = Continuous("x")

    p_x: DiracDeltaLayer

    @classmethod
    def setUpClass(cls):
        model = ProbabilisticCircuit()
        cls.p_x = DiracDeltaLayer(cls.x, np.array([0., 1.]), np.array([1, 2.]))
        model.add_layer(cls.p_x)


    def test_univariate_log_conditional_of_simple_interval_in_place(self):
        simple_interval = SimpleInterval(0, 0.5)
        p_x = copy.deepcopy(self.p_x)
        self.assertEqual(p_x.number_of_nodes, 2)
        p_x.univariate_log_conditional_of_simple_interval_in_place(simple_interval)
        self.assertEqual(p_x.number_of_nodes, 2)

        self.assertEqual(p_x.location[0], 0.)
        self.assertTrue(np.isnan(p_x.location[1]))
        self.assertTrue(np.allclose(p_x.result_of_current_query, np.array([0., -np.inf])))


if __name__ == '__main__':
    unittest.main()























