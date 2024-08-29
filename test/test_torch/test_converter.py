import time
import time
import unittest

import numpy as np
import pandas as pd
import torch
from numpy.testing import assert_almost_equal
from random_events.variable import Continuous, Integer, Symbolic

from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.learning.nyga_distribution import NygaDistribution
from probabilistic_model.learning.torch.pc import SumLayer, Layer


@unittest.skip("Outdated")
class FromNygaDistributionTestCase(unittest.TestCase):
    x = Continuous("x")
    nyga_distribution = NygaDistribution(x, min_likelihood_improvement=0.001, min_samples_per_quantile=10)
    data = np.random.normal(0, 1, 10000)
    nyga_distribution.fit(data)

    def test_from_pc(self):
        model = Layer.from_probabilistic_circuit(self.nyga_distribution.probabilistic_circuit).eval()
        self.assertIsInstance(model, SumLayer)
        self.assertEqual(model.number_of_nodes, 1)
        self.assertEqual(len(model.log_weights), 1)
        self.assertEqual(len(model.child_layers), 1)
        self.assertEqual(model.log_weights[0].shape, (1, len(self.nyga_distribution.subcircuits)))

        uniform_layer = model.child_layers[0]
        self.assertEqual(uniform_layer.number_of_nodes, len(self.nyga_distribution.subcircuits))

        tensor_data = torch.tensor(self.data).unsqueeze(1)

        ll_m = model.log_likelihood_of_nodes(tensor_data)
        numpy_data = self.data.reshape(-1, 1)
        ll_n = self.nyga_distribution.log_likelihood(numpy_data)

        assert_almost_equal(ll_m.squeeze().tolist(), ll_n.tolist(), decimal=4)


@unittest.skip("Outdated")
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
        data["integer"] = np.concatenate(
            (np.random.randint(low=0, high=4, size=int(size / 2)), np.random.randint(7, 10, int(size / 2))))
        data["symbol"] = np.random.randint(0, 4, size).astype(str)

        self.x, self.y, self.integer, self.symbol = infer_variables_from_dataframe(data, min_samples_per_quantile=4)

        self.model = JPT([self.x, self.y, ], min_samples_leaf=10)
        self.data = data[[v.name for v in self.model.variables_from_init]]
        self.model.fit(self.data)  # fig = go.Figure(self.model.plot())  # fig.show()

    def test_from_pc(self):
        lc = Layer.from_probabilistic_circuit(self.model.probabilistic_circuit).eval()

        tensor_data = torch.tensor(self.data.values)
        lc.log_likelihood_of_nodes(tensor_data[:10])
        lc.log_likelihood_of_nodes(tensor_data[:10])
        lc_ll_begin_time = time.time_ns()
        lc_ll = lc.log_likelihood_of_nodes(tensor_data)
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
