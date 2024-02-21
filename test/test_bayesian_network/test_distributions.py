import unittest

from random_events.events import Event
from random_events.variables import Symbolic

from probabilistic_model.bayesian_network.distributions import ConditionalProbabilityTable, RootDistribution
from probabilistic_model.bayesian_network.bayesian_network import BayesianNetwork
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.probabilistic_circuit.distributions import DiscreteDistribution as PCDiscreteDistribution

import tabulate


class DistributionTestCase(unittest.TestCase):

    x = Symbolic("x", [0, 1, 2])
    y = Symbolic("y", [0, 1])

    p_x = ConditionalProbabilityTable(x)
    p_yx = ConditionalProbabilityTable(y)

    def setUp(self):
        bayesian_network = BayesianNetwork()

        # create the root distribution for x
        self.p_x = RootDistribution(self.x, [0.5, 0.3, 0.2])

        # create the conditional probability table for y
        self.p_yx.conditional_probability_distributions[(0,)] = SymbolicDistribution(self.y, [0.5, 0.5])
        self.p_yx.conditional_probability_distributions[(1,)] = SymbolicDistribution(self.y, [0.3, 0.7])
        self.p_yx.conditional_probability_distributions[(2,)] = SymbolicDistribution(self.y, [0.1, 0.9])

        # add the distributions to the bayesian network
        bayesian_network.add_node(self.p_x)
        bayesian_network.add_node(self.p_yx)

        # add the edge between x and y
        bayesian_network.add_edge(self.p_x, self.p_yx)

    def test_to_tabulate(self):
        table = tabulate.tabulate(self.p_yx.to_tabulate())
        self.assertIsInstance(table, str)
        # print(table)

    def test_likelihood(self):
        self.assertEqual(self.p_yx.likelihood([0, 1]), 0.5)

    def test_forward_pass(self):
        event = Event({self.x: [0, 1], self.y: [0]})
        event = self.p_x.bayesian_network.preprocess_event(event)
        self.p_x.forward_pass(event)

        self.assertEqual(self.p_x.forward_message.weights, [0.5/0.8, 0.3/0.8, 0.])
        self.assertEqual(self.p_x.forward_probability, 0.8)

        self.p_yx.forward_pass(event)
        self.assertEqual(self.p_yx.forward_message.weights, [1., 0.])
        self.assertEqual(self.p_yx.forward_probability, 0.5/0.8 * 0.5 + 0.3/0.8 * 0.3)

    def test_forward_pass_impossible_event(self):
        self.p_x.weights = [1, 0, 0]
        event = Event({self.x: 2})
        event = self.p_x.bayesian_network.preprocess_event(event)

        self.p_x.forward_pass(event)
        self.assertIsNone(self.p_x.forward_message)
        self.assertEqual(self.p_x.forward_probability, 0)

        self.p_yx.forward_pass(event)
        self.assertIsNone(self.p_yx.forward_message)
        self.assertEqual(self.p_yx.forward_probability, 0)

    def test_joint_distribution_with_parents_root(self):
        event = Event()
        event = self.p_x.bayesian_network.preprocess_event(event)

        self.p_x.forward_pass(event)

        joint_distribution = self.p_x.joint_distribution_with_parents()
        self.assertIsInstance(joint_distribution, PCDiscreteDistribution)

    def test_joint_distribution_with_parents(self):
        event = Event()
        event = self.p_x.bayesian_network.preprocess_event(event)

        self.p_x.bayesian_network.forward_pass(event)

        joint_distribution = self.p_yx.joint_distribution_with_parents()
        self.assertIsInstance(joint_distribution, None)


if __name__ == '__main__':
    unittest.main()
