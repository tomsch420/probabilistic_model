import unittest
from probabilistic_model.bayesian_network import BayesianNetwork, ConditionalMultinomialDistribution
from probabilistic_model.distributions.multinomial import MultinomialDistribution
from random_events.variables import Symbolic
import numpy as np


class MinimalBayesianNetworkTestCase(unittest.TestCase):

    model: BayesianNetwork
    x: Symbolic = Symbolic('x', [0, 1, 2])
    y: Symbolic = Symbolic('y', [0, 1])
    d_x: ConditionalMultinomialDistribution
    d_xy: ConditionalMultinomialDistribution

    def setUp(self):
        np.random.seed(69)

        self.model = BayesianNetwork()

        self.d_x = ConditionalMultinomialDistribution([self.x])
        self.d_xy = ConditionalMultinomialDistribution([self.y])
        self.model.add_nodes_from([self.d_x, self.d_xy])
        self.model.add_edge(self.d_x, self.d_xy)

        self.d_x.probabilities = np.array([0.5, 0.3, 0.2])
        self.d_xy.probabilities = np.array([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]])

    def test_setup(self):
        self.assertEqual(len(self.model.nodes), 2)
        self.assertEqual(len(self.model.edges), 1)

    def test_parents(self):
        self.assertEqual(list(self.model.nodes)[1].parents, [list(self.model.nodes)[0]])

    def test_variables(self):
        self.assertEqual(self.model.variables, (self.x, self.y))

    def test_normalize(self):
        probabilities_before = self.d_xy.probabilities
        self.d_xy.probabilities = np.array([[5, 5], [6, 4], [7, 3]])
        self.d_xy.normalize()
        self.assertTrue(np.all(probabilities_before == self.d_xy.probabilities))


if __name__ == '__main__':
    unittest.main()
