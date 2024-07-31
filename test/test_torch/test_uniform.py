import math
import unittest

import torch
from random_events.interval import closed, open
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
from torch.testing import assert_close

from probabilistic_model.learning.torch import SumLayer
from probabilistic_model.learning.torch.uniform_layer import UniformLayer


class UniformTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")

    p_x = UniformLayer(x, torch.Tensor([[0, 1], [1, 3]]))

    def test_log_likelihood(self):
        data = torch.tensor([0.5, 1.5, 4]).reshape(-1, 1)
        ll = self.p_x.log_likelihood(data)
        self.assertEqual(ll.shape, (3, 2))
        result = [[0., -float("inf")], [-float("inf"), -math.log(2)], [-float("inf"), -float("inf")]]
        assert_close(ll, torch.tensor(result))

    def test_cdf(self):
        data = torch.tensor([0.5, 1.5, 4]).reshape(-1, 1)
        cdf = self.p_x.cdf(data)
        self.assertEqual(cdf.shape, (3, 2))
        result = [[0.5, 0], [1, 0.25], [1, 1]]
        assert_close(cdf, torch.tensor(result))

    def test_probability(self):
        event = SimpleEvent({self.x: closed(0.5, 2.5) | closed(3, 5)})
        prob = self.p_x.probability_of_simple_event(event)
        self.assertEqual(prob.shape, (2, 1))
        result = [0.5, 0.75]
        assert_close(prob, torch.tensor(result, dtype=prob.dtype).reshape(-1, 1))

    def test_support_per_node(self):
        support = self.p_x.support_per_node
        result = [SimpleEvent({self.x: open(0, 1)}).as_composite_set(),
                  SimpleEvent({self.x: open(1, 3)}).as_composite_set()]
        self.assertEqual(support, result)

    def test_conditional_singleton(self):
        event = SimpleEvent({self.x: closed(0.5, 0.5)})
        layer, ll = self.p_x.log_conditional_of_simple_event(event)
        self.assertEqual(layer.number_of_nodes, 1)
        assert_close(torch.tensor([0.5]), layer.location)
        assert_close(torch.tensor([1.]), layer.density_cap)

    def test_conditional_single_truncation(self):
        event = SimpleEvent({self.x: closed(0.5, 2.5)})
        layer, ll = self.p_x.log_conditional_of_simple_event(event)
        self.assertEqual(layer.number_of_nodes, 2)
        assert_close(layer.interval, torch.tensor([[0.5, 1], [1, 2.5]]))
        assert_close(torch.tensor([0.5, 0.75]).reshape(-1, 1).double(), ll)

    def test_conditional_multiple_truncation(self):
        event = closed(-1, 0.5) | closed(0.7, 0.8) | closed(2., 3.) | closed(3.5, 4.)

        layer, ll = self.p_x.log_conditional_from_interval(event)
        assert_close(torch.tensor([0.6, 0.5]).log().reshape(-1, 1).double(), ll)
        self.assertIsInstance(layer, SumLayer)

        layer.validate()
        self.assertEqual(layer.number_of_nodes, 2)
        self.assertEqual(len(layer.child_layers), 1)
        assert_close(layer.child_layers[0].interval, torch.tensor([[0., 0.5], [0.7, 0.8], [2., 3.]]))

        log_weights_by_hand = torch.tensor([[0.5, 0.1, 0.], [0., 0., 0.5]]).to_sparse_coo().double()
        log_weights_by_hand.values().log_()
        assert_close(layer.log_weights[0], log_weights_by_hand)

    def test_conditional_row_remove(self):
        event = closed(-1, 0.5) | closed(0.7, 0.8)
        layer, ll = self.p_x.log_conditional_from_interval(event)
        assert_close(torch.tensor([0.6, 0.]).log().reshape(-1, 1).double(), ll)
        self.assertIsInstance(layer, SumLayer)
        layer.validate()
        self.assertEqual(layer.number_of_nodes, 1)


if __name__ == '__main__':
    unittest.main()
