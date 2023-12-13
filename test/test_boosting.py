import unittest

import torch
from random_events.variables import Continuous
from torch.testing import assert_close

from probabilistic_model.learning.boosting.boosting import BoostedCircuit, NormalDistribution


class Boosting1DTestCase(unittest.TestCase):
    data = torch.cat(
        (torch.normal(0, 1, (100, 1)), torch.normal(3, 1, (100, 1)), torch.normal(6, 1, (100, 1)))).double()
    variable = Continuous("real")

    def test_normal_creation(self):
        distribution = NormalDistribution(self.variable, torch.tensor(0.), torch.tensor(1.))
        self.assertEqual(distribution.loc, 0.)
        self.assertEqual(distribution.scale, 1.)

    def test_boosted_circuit_creation(self):
        model = BoostedCircuit([self.variable])
        self.assertIsInstance(model.weights, torch.Tensor)

    def test_log_likelihood(self):
        model = BoostedCircuit([self.variable])
        model.weights = torch.tensor([1.])
        model.children = [NormalDistribution(self.variable, torch.tensor(0.), torch.tensor(1.))]
        log_likelihood = model.log_likelihood(torch.tensor([[0.]]))
        self.assertEqual(log_likelihood, torch.distributions.Normal(0, 1).log_prob(torch.tensor([[0.]])))

    def test_fit_with_single_component(self):
        model = BoostedCircuit([self.variable], number_of_components=1)
        model.fit(self.data)
        self.assertEqual(len(model.children), 1)
        self.assertAlmostEqual(model.children[0].children[0].loc.item(), torch.mean(self.data).item())
        self.assertAlmostEqual(model.children[0].children[0].scale.item(), torch.std(self.data).item(), delta=0.01)

    def test_fit_with_multiple_components(self):
        model = BoostedCircuit([self.variable], number_of_components=3)
        model.fit(self.data)
        self.assertEqual(len(model.children), 3)


if __name__ == '__main__':
    unittest.main()
