import unittest

import portion
from matplotlib import pyplot as plt
from random_events.events import Event
from random_events.variables import Symbolic, Continuous

from probabilistic_model.bayesian_network.distributions import (ConditionalProbabilityTable, DiscreteDistribution,
                                                                ConditionalProbabilisticCircuit)
from probabilistic_model.bayesian_network.bayesian_network import BayesianNetwork
from probabilistic_model.probabilistic_circuit.distributions import (DiscreteDistribution as PCDiscreteDistribution,
                                                                     SymbolicDistribution, UniformDistribution)
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import DeterministicSumUnit, \
    DecomposableProductUnit

import tabulate

import networkx as nx


class DistributionTestCase(unittest.TestCase):

    x = Symbolic("x", [0, 1, 2])
    y = Symbolic("y", [0, 1])

    p_x = ConditionalProbabilityTable(x)
    p_yx = ConditionalProbabilityTable(y)

    def setUp(self):
        bayesian_network = BayesianNetwork()

        # create the root distribution for x
        self.p_x = DiscreteDistribution(self.x, [0.5, 0.3, 0.2])

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

        joint_distribution = self.p_x.joint_distribution_with_parent()
        self.assertIsInstance(joint_distribution, DeterministicSumUnit)

    def test_joint_distribution_with_parents(self):
        event = Event()
        event = self.p_x.bayesian_network.preprocess_event(event)

        self.p_x.bayesian_network.forward_pass(event)

        joint_distribution = self.p_yx.joint_distribution_with_parent()
        self.assertIsInstance(joint_distribution, DeterministicSumUnit)

        self.assertEqual(joint_distribution.likelihood([0, 1]), 0.25)
        self.assertEqual(joint_distribution.likelihood([2, 1]), 0.2 * 0.9)


class CircuitDistributionTestCase(unittest.TestCase):

    x: Symbolic = Symbolic("x", [0, 1])
    y: Continuous = Continuous("y")
    z: Continuous = Continuous("z")
    p_x: DiscreteDistribution
    p_yzx = ConditionalProbabilisticCircuit([y, z])
    bayesian_network: BayesianNetwork

    def setUp(self):
        self.bayesian_network = BayesianNetwork()
        self.p_x = DiscreteDistribution(self.x, [0.7, 0.3])

        d1 = DecomposableProductUnit()
        d1.add_subcircuit(UniformDistribution(self.y, portion.closed(0, 1)))
        d1.add_subcircuit(UniformDistribution(self.z, portion.closed(0, 1)))

        d2 = DecomposableProductUnit()
        d2.add_subcircuit(UniformDistribution(self.y, portion.closed(0, 2)))
        d2.add_subcircuit(UniformDistribution(self.z, portion.closed(0, 3)))

        self.p_yzx.conditional_probability_distributions[(0,)] = d1.probabilistic_circuit
        self.p_yzx.conditional_probability_distributions[(1,)] = d2.probabilistic_circuit

        self.bayesian_network.add_nodes_from([self.p_x, self.p_yzx])
        self.bayesian_network.add_edge(self.p_x, self.p_yzx)

    def plot(self):
        nx.draw(self.bayesian_network, with_labels=True)
        plt.show()

    def test_likelihood(self):
        self.assertEqual(self.p_yzx.likelihood([0, 0, 0]), 1)
        self.assertEqual(self.p_yzx.likelihood([1, 0, 0]), 0.5 / 1/3)
        self.assertEqual(self.p_yzx.likelihood([1, -1, -1]), 0)

    def test_forward_pass(self):
        event = self.bayesian_network.preprocess_event(Event())
        self.bayesian_network.forward_pass(event)
        self.assertEqual(self.p_x.forward_probability, 1)
        self.assertEqual(self.p_yzx.forward_probability, 1)

    def test_probability(self):
        event = Event({self.x: 0, self.y: portion.closed(0, 0.5)})
        probability = self.bayesian_network.probability(event)
        self.assertEqual(probability, 0.7 * 0.5)

    def test_joint_distribution_with_parent(self):
        event = self.bayesian_network.preprocess_event(Event())
        self.bayesian_network.forward_pass(event)

        joint_distribution = self.p_yzx.joint_distribution_with_parent()
        event = Event({self.x: 0, self.y: portion.closed(0, 0.5)})
        self.assertEqual(joint_distribution.probability(event), self.bayesian_network.probability(event))


if __name__ == '__main__':
    unittest.main()
