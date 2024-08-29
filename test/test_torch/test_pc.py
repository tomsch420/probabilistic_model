import math
import unittest

import torch
from random_events.variable import Continuous

from probabilistic_model.learning.torch.pc import ProductLayer, SumLayer
from probabilistic_model.learning.torch.uniform_layer import UniformLayer
from random_events.product_algebra import Event, SimpleEvent
from random_events.interval import closed

@unittest.skip("This test is not working yet.")
class FullCircuitTestCase(unittest.TestCase):
    x = Continuous("x")
    y = Continuous("y")

    p_x = UniformLayer(x, torch.Tensor([[0, 1],
                                        [1, 3]]).double())
    p_y = UniformLayer(y, torch.Tensor([[2., 3],
                                        [4, 5]]).double())
    product_layer = ProductLayer([p_x, p_y], torch.tensor([[0, 1], [0, 1]]).long())

    model = SumLayer([product_layer], log_weights=[torch.tensor([[1, 1]]).double().to_sparse_coo()])
    model.validate()

    def test_log_likelihood(self):
        data = torch.tensor([[0.5, 2.5], [1.5, 4.5]]).double()
        ll = self.model.log_likelihood_of_nodes(data)
        self.assertEqual(ll.shape, (2, 1))
        self.assertAlmostEqual(ll[0, 0].item(), math.log(0.5))
        self.assertAlmostEqual(ll[1, 0].item(), math.log(0.25))

    def test_conditional(self):
        event = SimpleEvent({self.x: closed(0.5, 2.5),
                             self.y: closed(2.5, 4.5)}).as_composite_set().complement()
        conditional, log_prob = self.model.conditional(event)
        self.assertAlmostEqual(log_prob, 1 - (0.5 * 0.25 + 0.5 * 0.375))
        conditional.conditional(event)

    def test_merging(self):
        event = SimpleEvent({self.x: closed(0.5, 2.5),
                             self.y: closed(2.5, 4.5)}).as_composite_set().complement()
        conditional, log_prob = self.model.conditional(event)

        c1: SumLayer = conditional.child_layers[0]
        c2: SumLayer = conditional.child_layers[1]
        c1.merge_with_one_layer_inplace(c2)
        self.assertEqual(c1.number_of_nodes, 2)
        c1.validate()




if __name__ == '__main__':
    unittest.main()
