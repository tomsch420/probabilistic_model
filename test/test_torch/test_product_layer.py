import math
import unittest

import numpy as np
import torch
from numpy.testing import assert_almost_equal
from random_events.interval import SimpleInterval, closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
from torch.testing import assert_close

from probabilistic_model.learning.torch.pc import SumLayer, ProductLayer
from probabilistic_model.learning.torch.uniform_layer import UniformLayer
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import SumUnit, ProductUnit


class ProductTestCase(unittest.TestCase):
    x = Continuous("x")
    y = Continuous("y")
    p1_x_by_hand = UniformDistribution(x, SimpleInterval(0, 1))
    p1_y_by_hand = UniformDistribution(y, SimpleInterval(0.5, 1))
    p2_y_by_hand = UniformDistribution(y, SimpleInterval(5, 6))

    product_1 = ProductUnit()
    product_1.add_subcircuit(p1_x_by_hand)
    product_1.add_subcircuit(p1_y_by_hand)

    product_2 = ProductUnit()
    product_2.probabilistic_circuit = product_1.probabilistic_circuit
    product_2.add_subcircuit(p1_x_by_hand)
    product_2.add_subcircuit(p2_y_by_hand)

    p1_x = UniformLayer(x, torch.Tensor([[0, 1]]))
    p1_y = UniformLayer(y, torch.Tensor([[0.5, 1], [5, 6]]))

    product = ProductLayer(child_layers=[p1_x, p1_y], edges=torch.tensor([[0, 0], [0, 1]]).long())

    def test_log_likelihood(self):
        data = [[0.5, 0.75], [0.9, 0.7], [0.5, 5.5]]
        ll_p1_by_hand = self.product_1.log_likelihood(np.array(data))
        ll_p2_by_hand = self.product_2.log_likelihood(np.array(data))
        ll = self.product.log_likelihood(torch.tensor(data))
        self.assertEqual(ll.shape, (3, 2))
        assert_almost_equal(ll_p1_by_hand.tolist(), ll[:, 0].tolist())
        assert_almost_equal(ll_p2_by_hand.tolist(), ll[:, 1].tolist())

    def test_probability(self):
        event = SimpleEvent({self.x: closed(0.5, 2.5) | closed(3, 5), self.y: closed(0.5, 2.5) | closed(3, 5)})
        prob = self.product.probability_of_simple_event(event)
        self.assertEqual(prob.shape, (2, 1))
        p_by_hand_1 = self.product_1.probability_of_simple_event(event)
        p_by_hand_2 = self.product_2.probability_of_simple_event(event)
        assert_almost_equal([p_by_hand_1, p_by_hand_2], prob[:, 0].tolist())

    def test_conditional_of_simple_event(self):
        event = SimpleEvent({self.x: closed(0.5, 2.), self.y: closed(4, 5.5)})
        c, lp = self.product.log_conditional_of_simple_event(event)
        c.validate()
        self.assertEqual(c.number_of_nodes, 1)
        self.assertEqual(len(c.child_layers), 2)
        self.assertEqual(c.child_layers[0].number_of_nodes, 1)
        self.assertEqual(c.child_layers[1].number_of_nodes, 1)
        assert_close(lp, torch.tensor([0., 0.25]).reshape(-1, 1).log())

    def test_remove_nodes_inplace(self):
        product = self.product.__deepcopy__()
        remove_mask = torch.tensor([1, 0]).bool()
        product.remove_nodes_inplace(remove_mask)
        self.assertEqual(product.number_of_nodes, 1)
        product.validate()
        self.assertEqual(len(product.child_layers), 2)
        self.assertTrue((product.edges == 0).all())


class CleanUpTestCase(unittest.TestCase):
    x = Continuous("x")
    y = Continuous("y")

    p1_x = UniformLayer(x, torch.tensor([[0, 1]], dtype=torch.double))
    p1_y = UniformLayer(y, torch.tensor([[0.5, 1], [5, 6]], dtype=torch.double))

    model = ProductLayer(child_layers=[p1_x, p1_y], edges=torch.tensor([[0, 0], [1, 1]]).long())

    def test_clean_up_inplace(self):
        model = self.model.__deepcopy__()
        model.clean_up_orphans_inplace()
        self.assertEqual(model.number_of_nodes, 2)
        self.assertEqual(len(model.child_layers), 2)
        cleaned_child_layer = model.child_layers[1]
        self.assertEqual(cleaned_child_layer.number_of_nodes, 1)
        cleaned_child_layer.validate()
        self.assertTrue((model.edges == 0).all())
