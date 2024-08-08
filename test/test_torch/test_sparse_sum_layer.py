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
        assert_close(lp, torch.tensor([0., 0.25]).double().log())

    def test_remove_nodes_inplace(self):
        s1 = self.s1.__deepcopy__()
        remove_mask = torch.tensor([1, 0]).bool()
        s1.remove_nodes_inplace(remove_mask)
        self.assertEqual(s1.number_of_nodes, 1)
        s1.validate()
        self.assertEqual(len(s1.child_layers), 1)

    def test_log_normalization_constants(self):
        log_normalization_constants = self.s1.log_normalization_constants
        result = torch.tensor([2. + math.exp(2.), 2 * math.exp(2)]).log().double()
        assert_close(log_normalization_constants, result)

    def test_normalized_log_weights(self):
        log_normalized_weights = self.s1.normalized_log_weights

        dense_weights_1 = torch.tensor([[2], [0]])
        dense_weights_2 = torch.tensor([[-torch.inf, 2], [2, 2]]).exp()
        catted = torch.cat([dense_weights_1, dense_weights_2], dim=1).double()
        normalized = catted / catted.sum(dim=1, keepdim=True)

        catted_lnw = torch.cat(log_normalized_weights, dim=1).coalesce()
        catted_lnw.values().exp_()
        assert_close(catted_lnw.to_dense(), normalized)




class SparseMergingTestCase(unittest.TestCase):
    x = Continuous("x")
    y = Continuous("y")

    u1 = UniformLayer(x, torch.Tensor([[0, 1]]).double())
    u2 = UniformLayer(y, torch.Tensor([[0, 1]]).double())
    u3 = UniformLayer(x, torch.Tensor([[1, 2]]).double())
    s1 = SparseSumLayer([u1], log_weights=[torch.tensor([[1]]).double().to_sparse_coo().coalesce()])
    s2 = SparseSumLayer([u2], log_weights=[torch.tensor([[1]]).double().to_sparse_coo().coalesce()])
    s3 = SparseSumLayer([u3], log_weights=[torch.tensor([[3]]).double().to_sparse_coo().coalesce()])

    def test_merge_s1s2(self):
        s1 = self.s1.__deepcopy__()
        s2 = self.s2.__deepcopy__()
        s1.merge_with_one_layer_inplace(s2)
        self.assertEqual(s1.number_of_nodes, 2)
        self.assertEqual(len(s1.child_layers), 2)
        s1.validate()
        s1.child_layers[0].validate()
        s1.child_layers[1].validate()

    def test_merge_s1s3(self):
        s1 = self.s1.__deepcopy__()
        s3 = self.s3.__deepcopy__()
        s1.merge_with_one_layer_inplace(s3)
        self.assertEqual(s1.number_of_nodes, 2)
        self.assertEqual(len(s1.child_layers), 1)
        self.assertEqual(s1.child_layers[0].number_of_nodes, 2)
        s1.validate()
        s1.child_layers[0].validate()




if __name__ == '__main__':
    unittest.main()
