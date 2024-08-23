import math
import unittest

import numpy as np
import torch
from numpy.testing import assert_almost_equal
from random_events.interval import SimpleInterval, closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
from torch.testing import assert_close

from probabilistic_model.learning.torch import DiracDeltaLayer
from probabilistic_model.learning.torch.pc import SumLayer, ProductLayer
from probabilistic_model.learning.torch.uniform_layer import UniformLayer
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import SumUnit, ProductUnit
from probabilistic_model.utils import embed_sparse_tensor_in_nan_tensor


class UniformProductTestCase(unittest.TestCase):
    x = Continuous("x")
    y = Continuous("y")

    p1_x = UniformLayer(x, torch.Tensor([[0, 1],
                                        [1, 3]]).double())
    p2_x = UniformLayer(x, torch.Tensor([[2, 4],
                                        [4, 5]]).double())
    p_y = UniformLayer(y, torch.Tensor([[2., 3],
                                        [4, 6]]).double())

    indices = torch.tensor([[0, 1, 2, 2, ],
                            [0, 1, 0, 1, ]])
    values = torch.tensor([0, 0, 0, 1])
    edges = torch.sparse_coo_tensor(indices, values, (3, 2), is_coalesced=True)

    product_layer = ProductLayer([p1_x, p2_x, p_y], edges)

    def test_log_likelihood(self):
        data = [[0.5, 2.5], [3.5, 4.5], [0, 5]]
        ll = self.product_layer.log_likelihood(torch.tensor(data))
        self.assertEqual(ll.shape, (len(data), self.product_layer.number_of_nodes))
        result = torch.tensor([[0, -torch.inf],
                               [-torch.inf, math.log(0.25)],
                               [-torch.inf, -torch.inf]]).double()
        assert_close(ll, result)

    def test_probability(self):
        event = SimpleEvent({self.x: closed(0.5, 2.5) | closed(3, 5), self.y: closed(0.5, 2.5) | closed(3, 5)})
        prob = self.product_layer.probability_of_simple_event(event)
        self.assertEqual(prob.shape, (self.product_layer.number_of_nodes,))
        result = torch.tensor([0.25, 3./8.]).double()
        assert_close(result, prob)

    def test_conditional_of_simple_event(self):
        event = SimpleEvent({self.x: closed(0.5, 2.), self.y: closed(2, 2.5)})
        c, lp = self.product_layer.log_conditional_of_simple_event(event)
        c.validate()
        self.assertEqual(c.number_of_nodes, 1)
        self.assertEqual(len(c.child_layers), 2)
        self.assertEqual(c.child_layers[0].number_of_nodes, 1)
        self.assertEqual(c.child_layers[1].number_of_nodes, 1)
        assert_close(lp, torch.tensor([0.25, 0.]).log())

    def test_sample_from_frequencies(self):
        torch.random.manual_seed(69)
        frequencies = torch.tensor([5, 3])
        samples = self.product_layer.sample_from_frequencies(frequencies)
        for index, sample_row in enumerate(samples):
            sample_row = sample_row.coalesce().values()
            self.assertEqual(len(sample_row), frequencies[index])
            likelihood = self.product_layer.likelihood(sample_row)
            self.assertTrue(all(likelihood[:, index] > 0.))

    def test_cdf(self):
        data = torch.tensor([[1., 1.], [1, 2.5], [3, 5], [6, 6]]).double()
        cdf = self.product_layer.cdf(data)
        self.assertEqual(cdf.shape, (4, 2))
        result = torch.tensor([[0, 0], [0.5, 0.0], [1, 0.25], [1, 1]]).double()
        assert_close(cdf, result)


class DiracProductTestCase(unittest.TestCase):
    x = Continuous("x")
    y = Continuous("y")
    z = Continuous("z")

    p1_x = DiracDeltaLayer(x, torch.tensor([0., 1.]).double(), torch.tensor([1, 1]).double())
    p2_x = DiracDeltaLayer(x, torch.tensor([2., 3.]).double(), torch.tensor([1, 1]).double())
    p_y = DiracDeltaLayer(y, torch.tensor([4., 5.]).double(), torch.tensor([1, 1]).double())
    p_z = DiracDeltaLayer(z, torch.tensor([6.]).double(), torch.tensor([1]).double())

    indices = torch.tensor([[1, 2, 3, 3, 0, 0],
                            [0, 1, 0, 1, 0, 1]])
    values = torch.tensor([0, 0, 1, 0, 0, 0])
    edges = torch.sparse_coo_tensor(indices, values).coalesce()

    product_layer = ProductLayer([p_z, p1_x, p2_x, p_y, ], edges)

    def test_likelihood(self):
        data = torch.tensor([[0., 5., 6.],
                             [2, 4, 6]]).double()
        likelihood = self.product_layer.likelihood(data)
        self.assertTrue(likelihood[0, 0] > 0)
        self.assertTrue(likelihood[1, 1] > 0)
        self.assertTrue(likelihood[0, 1] == 0)
        self.assertTrue(likelihood[1, 0] == 0)

    def test_sample_from_frequencies(self):
        torch.random.manual_seed(69)
        frequencies = torch.tensor([5, 3])
        samples = self.product_layer.sample_from_frequencies(frequencies)

        samples_n0 = samples[0].to_dense()
        samples_n1 = samples[1].to_dense()

        self.assertEqual(samples_n0.shape, torch.Size((5, 3)))
        self.assertEqual(samples_n1.shape, torch.Size((5, 3)))
        self.assertEqual(len(samples[1].coalesce().values()), 3)
        self.assertTrue(torch.all(samples_n0 == torch.tensor([0, 5 ,6])))
        self.assertTrue(torch.all(samples_n1[:3] == torch.tensor([2, 4 ,6])))



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
