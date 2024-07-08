import unittest

import numpy as np
from random_events.variable import Continuous

from probabilistic_model.learning.nyga_distribution import NygaDistribution
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import SumUnit
from probabilistic_model.probabilistic_circuit.distributions.distributions import GaussianDistribution
import plotly.graph_objects as go
from probabilistic_model.learning.converter import TensorProbabilisticCircuit

class InterfaceTestCase(unittest.TestCase):

    x: Continuous = Continuous("x")
    model: SumUnit

    def setUp(self) -> None:
        model = NygaDistribution(self.x, min_likelihood_improvement=0.001, min_samples_per_quantile=300)
        data = np.random.normal(0, 1, 1000).tolist()
        model.fit(data)

        self.model = SumUnit()

        for weight, subcircuit in model.weighted_subcircuits:
            mean = subcircuit.expectation()[self.x]
            variance = subcircuit.variance()[self.x]
            normal_child = GaussianDistribution(self.x, mean, variance**0.5)
            self.model.add_subcircuit(normal_child, weight)

    def show(self):
        fig = go.Figure(self.model.plot())
        fig.show()

    def test_something(self):
        # self.show()
        result = TensorProbabilisticCircuit.from_pc(self.model.probabilistic_circuit)
        return
        data = self.model.sample(1000)
        log_likelihoods = self.model.log_likelihood(data)


if __name__ == '__main__':
    unittest.main()
