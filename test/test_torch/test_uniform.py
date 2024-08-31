import math
import unittest

import torch
from random_events.interval import closed, open, SimpleInterval, Bound
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
from torch.testing import assert_close

from probabilistic_model.probabilistic_circuit.nx.distributions import UniformDistribution
from probabilistic_model.probabilistic_circuit.torch import SumLayer
from probabilistic_model.probabilistic_circuit.torch.uniform_layer import UniformLayer
from probabilistic_model.utils import simple_interval_to_open_tensor, timeit


class UniformTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")

    p_x = UniformLayer(x, torch.Tensor([[0, 1], [1, 3]]).double())

    def test_log_likelihood(self):
        data = torch.tensor([0.5, 1.5, 4]).reshape(-1, 1)
        ll = self.p_x.log_likelihood_of_nodes(data)
        self.assertEqual(ll.shape, (3, 2))
        result = [[0., -float("inf")], [-float("inf"), -math.log(2)], [-float("inf"), -float("inf")]]
        assert_close(ll, torch.tensor(result).double())

    def test_cdf(self):
        data = torch.tensor([0.5, 1.5, 4]).reshape(-1, 1)
        cdf = self.p_x.cdf_of_nodes(data)
        self.assertEqual(cdf.shape, (3, 2))
        result = [[0.5, 0], [1, 0.25], [1, 1]]
        assert_close(cdf, torch.tensor(result).double())

    def test_probability(self):
        event = SimpleEvent({self.x: closed(0.5, 2.5) | closed(3, 5)})
        prob = self.p_x.probability_of_simple_event(event)
        self.assertEqual(prob.shape, (2,))
        result = [0.5, 0.75]
        assert_close(prob, torch.tensor(result, dtype=prob.dtype))

    def test_support_per_node(self):
        support = self.p_x.support_per_node
        result = [SimpleEvent({self.x: open(0, 1)}).as_composite_set(),
                  SimpleEvent({self.x: open(1, 3)}).as_composite_set()]
        self.assertEqual(support, result)

    def test_conditional_singleton(self):
        event = SimpleEvent({self.x: closed(0.5, 0.5)})
        layer, ll = self.p_x.log_conditional_of_simple_event(event)
        self.assertEqual(layer.number_of_nodes, 1)
        assert_close(torch.tensor([0.5]).double(), layer.location)
        assert_close(torch.tensor([1.]).double(), layer.density_cap)

    def test_conditional_single_truncation(self):
        event = SimpleEvent({self.x: closed(0.5, 2.5)})
        layer, ll = self.p_x.log_conditional_of_simple_event(event)
        self.assertEqual(layer.number_of_nodes, 2)
        assert_close(layer.interval, torch.tensor([[0.5, 1], [1, 2.5]]))
        assert_close(torch.tensor([0.5, 0.75]).double().log(), ll)

    def test_conditional_multiple_truncation(self):
        event = closed(-1, 0.5) | closed(0.7, 0.8) | closed(2., 3.) | closed(3.5, 4.)

        layer, ll = self.p_x.log_conditional_from_interval(event)
        assert_close(torch.tensor([0.6, 0.5]).log().double(), ll)
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
        assert_close(torch.tensor([0.6, 0.]).log().double(), ll)
        self.assertIsInstance(layer, SumLayer)
        layer.validate()
        self.assertEqual(layer.number_of_nodes, 1)

    def test_sampling(self):
        samples = self.p_x.sample_from_frequencies(torch.tensor([20, 10]))
        self.assertEqual(samples.shape, (2, 20, 1))
        samples = samples.values()
        self.assertEqual(samples.shape, torch.Size((30, 1)))
        samples_n0 = samples[:20]
        samples_n1 = samples[20:30]

        l_n0 = self.p_x.log_likelihood_of_nodes(samples_n0)[:, 0]
        l_n1 = self.p_x.log_likelihood_of_nodes(samples_n1)[:, 1]

        self.assertTrue(all(l_n0 > -torch.inf))
        self.assertTrue(all(l_n1 > -torch.inf))

    def test_moment(self):
        order = torch.tensor([1]).long()
        center = torch.tensor([1.]).double()
        moment = self.p_x.moment_of_nodes(order, center)
        result = torch.tensor([[-0.5], [1.]]).double()
        assert_close(moment, result)

    def test_log_mode(self):
        mode, ll = self.p_x.log_mode_of_nodes()
        result_ll = torch.tensor([1, 0.5]).log().double()
        assert_close(result_ll, ll)
        self.assertEqual(mode, self.p_x.support_per_node)


class CompiledSpeedTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")
    number_of_nodes = 1000

    def setUp(self):
        intervals = [SimpleInterval(0, 1, Bound.OPEN, Bound.OPEN) for _ in range(self.number_of_nodes)]
        self.nx_uniform_distributions = [UniformDistribution(self.x, interval) for interval in intervals]
        self.torch_uniform_distributions = UniformLayer(self.x, torch.stack([torch.zeros(self.number_of_nodes), torch.ones(self.number_of_nodes)]).double().T)

    def test_log_likelihood(self):
        self.assertEqual(len(self.nx_uniform_distributions), self.torch_uniform_distributions.number_of_nodes)
        samples = torch.rand(10000, 1).double()

        @timeit
        def time_nx():
            return [node.log_likelihood(samples) for node in self.nx_uniform_distributions]

        @timeit
        def time_torch():
            return self.torch_uniform_distributions.log_likelihood_of_nodes(samples)

        nx_result = time_nx()
        for i in range(10):
            time_torch()

        torch_result = self.torch_uniform_distributions.log_likelihood_of_nodes(samples)
        for tr, nxr in zip(torch_result.T, nx_result):
            assert_close(tr, torch.tensor(nxr).double())

    def test_sampling(self):
        samples_per_node = 1000
        frequencies = torch.full((self.number_of_nodes,), samples_per_node).long()

        @timeit
        def time_torch():
            return self.torch_uniform_distributions.sample_from_frequencies(frequencies)

        @timeit
        def time_nx():
            return [node.sample(samples_per_node) for node in self.nx_uniform_distributions]

        nx_result = time_nx()
        for i in range(10):
            time_torch()

    def test_conditioning(self):
        event = SimpleEvent({self.x: closed(0.5, 1.5)})

        @timeit
        def time_torch():
            return self.torch_uniform_distributions.log_conditional_of_simple_event(event)

        @timeit
        def time_nx():
            return [node.log_conditional_of_simple_event(event) for node in self.nx_uniform_distributions]

        nx_result = time_nx()
        for i in range(10):
            time_torch()


if __name__ == '__main__':
    unittest.main()
