import math
import unittest

import torch
from random_events.interval import closed, singleton
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
from torch.testing import assert_close

from probabilistic_model.learning.torch.input_layer import DiracDeltaLayer


class DiracDeltaLayerTestCase(unittest.TestCase):

    x: Continuous = Continuous("x")
    p_x = DiracDeltaLayer(x, torch.tensor([0., 1.]), torch.tensor([1., 2.]))

    def test_likelihood(self):
        data = torch.tensor([0, 1, 2]).reshape(-1, 1)
        ll = self.p_x.log_likelihood(data)
        self.assertEqual(ll.shape, torch.Size((len(data), self.p_x.number_of_nodes)))
        result = [[0, -torch.inf],
                  [-torch.inf, math.log(2)],
                  [-torch.inf, -torch.inf]]
        assert_close(ll, torch.tensor(result))

    def test_support_per_node(self):
        support = self.p_x.support_per_node
        result = [SimpleEvent({self.x: singleton(0)}).as_composite_set(),
                  SimpleEvent({self.x: singleton(1)}).as_composite_set()]
        self.assertEqual(support, result)

    def test_conditional_of_simple_interval(self):
        interval = closed(-0.5, 0.5).simple_sets[0]
        layer, ll = self.p_x.log_conditional_from_simple_interval(interval)
        result = torch.tensor([1, 0]).log()
        assert_close(ll, result)
        layer.validate()
        self.assertEqual(layer.number_of_nodes, 1)
        assert_close(layer.location, torch.tensor([0.]))
        assert_close(layer.density_cap, torch.tensor([1.]))

    def test_sample(self):
        s = self.p_x.sample_from_frequencies(torch.tensor([10, 5]))
        self.assertTrue(torch.all(s.values()[:10] == 0.))
        self.assertTrue(torch.all(s.values()[10:] == 1.))

    def test_cdf(self):
        data = torch.tensor([-1, 0, 1, 2]).unsqueeze(-1).double()
        cdf = self.p_x.cdf(data)
        result = torch.tensor([[0, 0], [1, 0], [1, 1], [1, 1]]).double()
        assert_close(cdf, result)

if __name__ == '__main__':
    unittest.main()
