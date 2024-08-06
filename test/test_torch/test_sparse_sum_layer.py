import math
import unittest

import numpy as np
import torch
from numpy.testing import assert_almost_equal
from random_events.interval import SimpleInterval, closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
from torch.testing import assert_close

from probabilistic_model.learning.torch.pc import SparseSumLayer, ProductLayer
from probabilistic_model.learning.torch.uniform_layer import UniformLayer


class SparseSumUnitTestCase(unittest.TestCase):
    x = Continuous("x")
    p1_x = UniformLayer(x, torch.Tensor([[0, 1]]).double())
    p2_x = UniformLayer(x, torch.Tensor([[1, 3], [1, 1.5]]).double())
    s1 = SparseSumLayer([p1_x, p2_x],
                  log_weights=[torch.tensor([[2], [0]]).log().to_sparse_coo().coalesce().double(),
                               torch.tensor([[0, 2], [2, 2]]).to_sparse_coo().coalesce().double()])

    def test_conditional(self):
        event = SimpleEvent({self.x: closed(2., 3.)})
        c, lp = self.s1.log_conditional_of_simple_event(event)
        c.validate()
        self.assertEqual(c.number_of_nodes, 1)
        self.assertEqual(len(c.child_layers), 1)
        self.assertEqual(c.child_layers[0].number_of_nodes, 1)
        assert_close(lp, torch.tensor([0., 0.25]).reshape(-1, 1).double().log())

    def test_remove_nodes_inplace(self):
        s1 = self.s1.__deepcopy__()
        remove_mask = torch.tensor([1, 0]).bool()
        s1.remove_nodes_inplace(remove_mask)
        self.assertEqual(s1.number_of_nodes, 1)
        s1.validate()
        self.assertEqual(len(s1.child_layers), 1)



if __name__ == '__main__':
    unittest.main()
