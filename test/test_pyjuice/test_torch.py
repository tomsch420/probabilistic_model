import math
import time
import unittest

import numpy as np
import pandas as pd
import torch
from numpy.testing import assert_almost_equal
from random_events.interval import SimpleInterval
from random_events.variable import Continuous, Integer, Symbolic
from torch.testing import assert_close

from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.learning.nyga_distribution import NygaDistribution
from probabilistic_model.learning.torch.pc import UniformLayer, SumLayer, ProductLayer, Layer
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import SumUnit, ProductUnit
import plotly.graph_objects as go


class UniformTestCase(unittest.TestCase):
    x = Continuous("x")
    p_x = UniformLayer(x, torch.Tensor([0, 1]), torch.tensor([1, 3]))

    def test_log_likelihood(self):
        ll = self.p_x.log_likelihood(torch.tensor([0.5, 1.5, 4]).reshape(-1, 1))
        self.assertEqual(ll.shape, (3, 2))
        result = [[-0.0, -torch.inf],
                  [-torch.inf, -torch.log(torch.tensor(2.))],
                  [-torch.inf, -torch.inf]]
        self.assertEqual(ll.tolist(), result)


class SumTestCase(unittest.TestCase):
    x = Continuous("x")
    p1_x = UniformLayer(x, torch.Tensor([0]), torch.tensor([1]))
    p2_x = UniformLayer(x, torch.Tensor([1, 1]), torch.tensor([3, 1.5]))
    s1 = SumLayer([p1_x, p2_x], log_weights=[torch.tensor([[math.log(2)], [1]]),
                                             torch.tensor([[0, 0], [1, 1]])])

    p1_x_by_hand = UniformDistribution(x, SimpleInterval(0, 1))
    p2_x_by_hand = UniformDistribution(x, SimpleInterval(1, 3))
    p3_x_by_hand = UniformDistribution(x, SimpleInterval(1, 1.5))
    s1_by_hand = SumUnit()
    s1_by_hand.add_subcircuit(p1_x_by_hand, 1 / 2)
    s1_by_hand.add_subcircuit(p2_x_by_hand, 1 / 4)
    s1_by_hand.add_subcircuit(p3_x_by_hand, 1 / 4)

    s2_by_hand = SumUnit()
    s2_by_hand.probabilistic_circuit = s1_by_hand.probabilistic_circuit
    s2_by_hand.add_subcircuit(p1_x_by_hand, 1 / 3)
    s2_by_hand.add_subcircuit(p2_x_by_hand, 1 / 3)
    s2_by_hand.add_subcircuit(p3_x_by_hand, 1 / 3)

    def test_stack(self):
        self.assertEqual(self.s1.concatenated_weights.shape, (2, 3))

    def test_normalizing_constant(self):
        assert_close(self.s1.log_normalization_constants, torch.tensor([torch.log(torch.tensor(4.)),
                                                                        torch.log(torch.exp(torch.tensor(1)) * 3)]))

    def test_log_likelihood(self):
        input = torch.tensor([0.5, 1.5, 2.5]).reshape(-1, 1)

        p_by_hand_1 = self.s1_by_hand.log_likelihood(input)
        p_by_hand_2 = self.s2_by_hand.log_likelihood(input)

        self.assertEqual(input.shape, (3, 1))

        ll = self.s1.log_likelihood(input)
        self.assertEqual(ll.shape, (3, 2))
        assert_almost_equal(p_by_hand_1.tolist(), ll[:, 0].tolist())
        assert_almost_equal(p_by_hand_2.tolist(), ll[:, 1].tolist())


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

    p1_x = UniformLayer(x, torch.Tensor([0]), torch.tensor([1]))
    p1_y = UniformLayer(y, torch.Tensor([0.5, 5]), torch.tensor([1, 6]))

    product = ProductLayer(child_layers=[p1_x, p1_y], edges=[torch.tensor([0, 0]),
                                                             torch.tensor([0, 1])])

    def test_log_likelihood(self):
        data = [[0.5, 0.75], [0.9, 0.7], [0.5, 5.5]]
        ll_p1_by_hand = self.product_1.log_likelihood(np.array(data))
        ll_p2_by_hand = self.product_2.log_likelihood(np.array(data))
        ll = self.product.log_likelihood(torch.tensor(data))
        self.assertEqual(ll.shape, (3, 2))
        assert_almost_equal(ll_p1_by_hand.tolist(), ll[:, 0].tolist())
        assert_almost_equal(ll_p2_by_hand.tolist(), ll[:, 1].tolist())


class FromNygaDistributionTestCase(unittest.TestCase):

    x = Continuous("x")
    nyga_distribution = NygaDistribution(x, min_likelihood_improvement=0.001, min_samples_per_quantile=10)
    data = np.random.normal(0, 1, 100000)
    nyga_distribution.fit(data)

    def test_from_pc(self):
        print(self.nyga_distribution.probabilistic_circuit)
        model = Layer.from_probabilistic_circuit(self.nyga_distribution.probabilistic_circuit)
        self.assertIsInstance(model, SumLayer)
        self.assertEqual(model.number_of_nodes, 1)
        self.assertEqual(len(model.log_weights), 1)
        self.assertEqual(len(model.child_layers), 1)
        self.assertEqual(model.log_weights[0].shape, (1, len(self.nyga_distribution.subcircuits)))

        uniform_layer = model.child_layers[0]
        # print(uniform_layer.number_of_nodes)
        self.assertEqual(uniform_layer.number_of_nodes, len(self.nyga_distribution.subcircuits))

        tensor_data = torch.tensor(self.data).unsqueeze(1)
        ll_m_begin_time = time.time_ns()
        ll_m = model.log_likelihood(tensor_data)
        ll_m_time_total = time.time_ns() - ll_m_begin_time
        print(f"Time for log likelihood calculation: {ll_m_time_total}")

        numpy_data = self.data.reshape(-1, 1)
        ll_n_begin_time = time.time_ns()
        ll_n = self.nyga_distribution.log_likelihood(numpy_data)
        ll_n_time_total = time.time_ns() - ll_n_begin_time
        print(f"Time for log likelihood calculation: {ll_n_time_total}")
        print("Speedup: ", ll_m_time_total / ll_n_time_total)
        assert_almost_equal(ll_m.squeeze().tolist(), ll_n.tolist(), decimal=4)


class FromJPTTestCase(unittest.TestCase):
    data: pd.DataFrame
    x: Continuous
    y: Continuous
    integer: Integer
    symbol: Symbolic
    model: JPT

    def setUp(self):
        np.random.seed(69)
        data = pd.DataFrame()
        size = 100
        data["x"] = np.random.normal(2, 4, size)
        data["y"] = np.random.normal(2, 4, size)
        data["integer"] = np.concatenate((np.random.randint(low=0, high=4, size=int(size/2)),
                                          np.random.randint(7, 10, int(size/2))))
        data["symbol"] = np.random.randint(0, 4, size).astype(str)

        self.x, self.y, self.integer, self.symbol = infer_variables_from_dataframe(data)

        self.model = JPT([self.x, self.y,], min_samples_leaf=10)
        self.data = data[[v.name for v in self.model.variables_from_init]]
        self.model.fit(self.data)
        # fig = go.Figure(self.model.plot())
        # fig.show()

    def test_from_pc(self):
        lc = Layer.from_probabilistic_circuit(self.model.probabilistic_circuit)
        tensor_data = torch.tensor(self.data.values)
        lc_ll_begin_time = time.time_ns()
        lc_ll = lc.log_likelihood(tensor_data)
        lc_ll_time_total = time.time_ns() - lc_ll_begin_time
        print(f"Time for log likelihood calculation: {lc_ll_time_total}")

        numpy_data = self.data.to_numpy()
        model_ll_begin_time = time.time_ns()
        model_ll = self.model.log_likelihood(numpy_data)
        model_ll_time_total = time.time_ns() - model_ll_begin_time
        print(f"Time for log likelihood calculation: {model_ll_time_total}")
        print("Speedup: ", lc_ll_time_total / model_ll_time_total)

        assert_almost_equal(lc_ll.squeeze().tolist(), model_ll.tolist(), decimal=4)


if __name__ == '__main__':
    unittest.main()
