import itertools
import unittest

import portion
from random_events.events import Event

from probabilistic_model.bayesian_network.bayesian_network import BayesianNetwork
from probabilistic_model.bayesian_network.distributions import ConditionalProbabilityTable, RootDistribution

from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution, SymbolicDistribution
from probabilistic_model.distributions.multinomial import MultinomialDistribution
from random_events.variables import Symbolic, Continuous, Integer
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx


class MinimalBayesianNetworkTestCase(unittest.TestCase):

    model: BayesianNetwork
    x: Symbolic = Symbolic('x', [0, 1, 2])
    y: Symbolic = Symbolic('y', [0, 1])
    p_x: RootDistribution
    p_yx: ConditionalProbabilityTable = ConditionalProbabilityTable(y)

    bf_distribution: MultinomialDistribution

    def setUp(self):
        np.random.seed(69)

        self.model = BayesianNetwork()

        # create the root distribution for x
        self.p_x = RootDistribution(self.x, [0.5, 0.3, 0.2])

        # create the conditional probability table for y
        self.p_yx.conditional_probability_distributions[(0,)] = SymbolicDistribution(self.y, [0.5, 0.5])
        self.p_yx.conditional_probability_distributions[(1,)] = SymbolicDistribution(self.y, [0.3, 0.7])
        self.p_yx.conditional_probability_distributions[(2,)] = SymbolicDistribution(self.y, [0.1, 0.9])

        # add the distributions to the bayesian network
        self.model.add_node(self.p_x)
        self.model.add_node(self.p_yx)

        # add the edge between x and y
        self.model.add_edge(self.p_x, self.p_yx)

        self.bf_distribution = self.model.brute_force_joint_distribution()

    def plot(self):
        pos = nx.planar_layout(self.model)
        nx.draw(self.model, pos=pos, with_labels=True, labels={node: repr(node) for node in self.model.nodes})
        plt.show()

    def test_setup(self):
        self.assertEqual(len(self.model.nodes), 2)
        self.assertEqual(len(self.model.edges), 1)

    def test_parents(self):
        self.assertEqual(self.p_yx.parent, self.p_x)

    def test_variables(self):
        self.assertEqual(self.model.variables, (self.x, self.y))

    def test_brute_force_joint_distribution(self):
        distribution = self.model.brute_force_joint_distribution()
        self.assertEqual(distribution.variables, (self.x, self.y))
        self.assertEqual(distribution.probabilities.sum(), 1)

    def test_likelihood(self):
        event = [0, 1]
        likelihood = self.model.likelihood(event)
        self.assertEqual(likelihood, 0.5 * 0.5)

        event = [1, 0]
        likelihood = self.model.likelihood(event)
        self.assertEqual(likelihood, 0.3 * 0.3)

    def test_probability(self):
        event = Event({self.x: [1], self.y: [0, 1]})
        probability = self.model.probability(event)
        self.assertEqual(probability, self.bf_distribution.probability(event))

    def test_probability_empty_x(self):
        event = Event({self.y: [1]})
        probability = self.model.probability(event)
        self.assertEqual(probability, self.bf_distribution.probability(event))

    def test_as_probabilistic_circuit(self):
        circuit = self.model.as_probabilistic_circuit()
        nx.draw(circuit.probabilistic_circuit.simplify(), with_labels=True)
        plt.show()
        for event in itertools.product(self.x.domain, self.y.domain):
            self.assertAlmostEqual(self.model.likelihood(event),
                                   circuit.likelihood(event))


# class ComplexBayesianNetworkTestCase(unittest.TestCase):
#
#     model: BayesianNetwork
#     x: Symbolic = Symbolic('x', [0, 1, 2])
#     y: Symbolic = Symbolic('y', [0, 1])
#     z: Symbolic = Symbolic('z', [0, 1])
#     a: Symbolic = Symbolic('a', [0, 1])
#
#     d_x: ConditionalMultinomialDistribution
#     d_yx: ConditionalMultinomialDistribution
#     d_zx: ConditionalMultinomialDistribution
#     d_az: ConditionalMultinomialDistribution
#
#     bf_distribution: MultinomialDistribution
#
#     def setUp(self):
#         np.random.seed(69)
#
#         self.model = BayesianNetwork()
#
#         self.d_x = ConditionalMultinomialDistribution([self.x])
#         self.d_yx = ConditionalMultinomialDistribution([self.y])
#         self.d_zx = ConditionalMultinomialDistribution([self.z])
#         self.d_az = ConditionalMultinomialDistribution([self.a])
#
#         self.model.add_nodes_from([self.d_x, self.d_yx, self.d_zx, self.d_az])
#         self.model.add_edge(self.d_x, self.d_yx)
#         self.model.add_edge(self.d_x, self.d_zx)
#         self.model.add_edge(self.d_zx, self.d_az)
#
#         self.d_x.probabilities = np.array([0.5, 0.3, 0.2])
#         self.d_yx.probabilities = np.array([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]])
#         self.d_zx.probabilities = np.array([[0.1, 0.9], [1., 0.], [0.8, 0.2]])
#         self.d_az.probabilities = np.array([[0.3, 0.7], [0.6, 0.4]])
#
#         self.bf_distribution = self.model.brute_force_joint_distribution()
#
#     def plot(self):
#         pos = nx.planar_layout(self.model)
#         nx.draw(self.model, pos=pos, with_labels=True, labels={node: repr(node) for node in self.model.nodes})
#         plt.show()
#
#     def test_likelihood(self):
#         for event in itertools.product(*[variable.domain for variable in self.model.variables]):
#             self.assertEqual(self.model.likelihood(event), self.bf_distribution.likelihood(event))
#
#     def test_probability(self):
#         event = Event({self.x: [1], self.y: [0, 1], self.z: [0]})
#         probability = self.model.probability(event)
#         self.assertEqual(probability, self.bf_distribution.probability(event))
#
#     def test_as_probabilistic_circuit(self):
#         circuit = self.model.as_probabilistic_circuit()
#         for event in itertools.product(self.a.domain, self.x.domain, self.y.domain, self.z.domain):
#             self.assertAlmostEqual(self.model.likelihood(event),
#                                    circuit.likelihood(event))
#
#
# class BayesianNetworkWithCircuitTestCase(unittest.TestCase):
#     model: BayesianNetwork
#     x: Integer = Integer('x', [0, 1, 2])
#     y: Continuous = Continuous('y')
#     d_x: ConditionalMultinomialDistribution
#     d_xy: ConditionalProbabilisticCircuit
#
#     def setUp(self):
#         np.random.seed(69)
#
#         self.model = BayesianNetwork()
#
#         self.d_x = ConditionalMultinomialDistribution([self.x])
#         self.d_xy = ConditionalProbabilisticCircuit()
#
#         for x_value in self.x.domain:
#             distribution = UniformDistribution(self.y, portion.closed(x_value, 5.))
#             self.d_xy.circuits[(x_value, )] = distribution
#
#         self.model.add_nodes_from([self.d_x, self.d_xy])
#         self.model.add_edge(self.d_x, self.d_xy)
#
#         self.d_x.probabilities = np.array([0.5, 0.3, 0.2])
#
#     def plot(self):
#         pos = nx.planar_layout(self.model)
#         nx.draw(self.model, pos=pos, with_labels=True, labels={node: repr(node) for node in self.model.nodes})
#         plt.show()
#
#     def test_likelihood(self):
#         event = [0, 1]
#         likelihood = self.model.likelihood(event,)
#         self.assertEqual(likelihood, 0.5 * 1 / 5)
#
#         event = [1, 2]
#         likelihood = self.model.likelihood(event)
#         self.assertEqual(likelihood, 0.3 * 1 / 4)
#
#         event = [1, 0]
#         likelihood = self.model.likelihood(event)
#         self.assertEqual(likelihood, 0.)
#
#     def test_probability(self):
#         event = Event({self.x: [1, 2], self.y: portion.closed(0, 2)})
#         probability = self.model.probability(event)
#         self.assertEqual(probability, 0.3 * 1 / 4)
#
#     def test_as_probabilistic_circuit(self):
#         circuit = self.model.as_probabilistic_circuit()
#         self.assertEqual(circuit.probabilistic_circuit.probability(Event()), 1.)
#
#         nx.draw(circuit.probabilistic_circuit.simplify(), with_labels=True)
#         plt.show()
#
#         event = Event({self.x: [1, 2], self.y: portion.closed(0, 2)})
#         self.assertAlmostEqual(self.model.probability(event), circuit.probability(event))



if __name__ == '__main__':
    unittest.main()
