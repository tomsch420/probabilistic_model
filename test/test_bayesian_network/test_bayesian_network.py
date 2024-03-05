import itertools
import math
import unittest

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import portion
from random_events.events import Event
from random_events.variables import Symbolic, Continuous

from probabilistic_model.bayesian_network.bayesian_network import BayesianNetwork
from probabilistic_model.bayesian_network.distributions import (ConditionalProbabilityTable, SymbolicDistribution,
                                                                ConditionalProbabilisticCircuit, DiscreteDistribution)
from probabilistic_model.distributions.multinomial import MultinomialDistribution
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import DecomposableProductUnit


class MinimalBayesianNetworkTestCase(unittest.TestCase):
    model: BayesianNetwork
    x: Symbolic = Symbolic('x', [0, 1, 2])
    y: Symbolic = Symbolic('y', [0, 1])
    p_x: SymbolicDistribution
    p_yx: ConditionalProbabilityTable = ConditionalProbabilityTable(y)

    bf_distribution: MultinomialDistribution

    def setUp(self):
        np.random.seed(69)

        self.model = BayesianNetwork()

        # create the root distribution for x
        self.p_x = SymbolicDistribution(self.x, [0.5, 0.3, 0.2])

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
        for event in itertools.product(self.x.domain, self.y.domain):
            self.assertAlmostEqual(self.model.likelihood(event), circuit.likelihood(event))


class ComplexBayesianNetworkTestCase(unittest.TestCase):
    model: BayesianNetwork
    x: Symbolic = Symbolic('x', [0, 1, 2])
    y: Symbolic = Symbolic('y', [0, 1])
    z: Symbolic = Symbolic('z', [0, 1])
    a: Symbolic = Symbolic('a', [0, 1])

    d_x: SymbolicDistribution
    d_yx = ConditionalProbabilityTable(y)
    d_zx = ConditionalProbabilityTable(z)
    d_az = ConditionalProbabilityTable(a)

    bf_distribution: MultinomialDistribution

    def setUp(self):
        np.random.seed(69)

        self.model = BayesianNetwork()

        self.d_x = SymbolicDistribution(self.x, [0.5, 0.3, 0.2])

        self.model.add_nodes_from([self.d_x, self.d_yx, self.d_zx, self.d_az])
        self.model.add_edge(self.d_x, self.d_yx)
        self.model.add_edge(self.d_x, self.d_zx)
        self.model.add_edge(self.d_zx, self.d_az)

        self.d_yx.conditional_probability_distributions[(0,)] = SymbolicDistribution(self.y, [0.5, 0.5])
        self.d_yx.conditional_probability_distributions[(1,)] = SymbolicDistribution(self.y, [0.6, 0.4])
        self.d_yx.conditional_probability_distributions[(2,)] = SymbolicDistribution(self.y, [0.7, 0.3])

        self.d_zx.conditional_probability_distributions[(0,)] = SymbolicDistribution(self.z, [0.1, 0.9])
        self.d_zx.conditional_probability_distributions[(1,)] = SymbolicDistribution(self.z, [1., 0.])
        self.d_zx.conditional_probability_distributions[(2,)] = SymbolicDistribution(self.z, [0.8, 0.2])

        self.d_az.conditional_probability_distributions[(0,)] = SymbolicDistribution(self.a, [0.3, 0.7])
        self.d_az.conditional_probability_distributions[(1,)] = SymbolicDistribution(self.a, [0.6, 0.4])

        self.bf_distribution = self.model.brute_force_joint_distribution()

    def plot(self):
        pos = nx.planar_layout(self.model)
        nx.draw(self.model, pos=pos, with_labels=True, labels={node: repr(node) for node in self.model.nodes})
        plt.show()

    def test_likelihood(self):
        for event in itertools.product(*[variable.domain for variable in self.model.variables]):
            self.assertEqual(self.model.likelihood(event), self.bf_distribution.likelihood(event))

    def test_probability(self):
        event = Event({self.x: [1], self.y: [0, 1], self.z: [0]})
        probability = self.model.probability(event)
        self.assertEqual(probability, self.bf_distribution.probability(event))

    def test_as_probabilistic_circuit(self):
        circuit = self.model.as_probabilistic_circuit().simplify()
        self.assertLess(len(circuit.weighted_edges), math.prod([len(v.domain) for v in circuit.variables]))
        for event in itertools.product(self.a.domain, self.x.domain, self.y.domain, self.z.domain):
            self.assertAlmostEqual(self.model.likelihood(event), circuit.likelihood(event))


class BayesianNetworkWithCircuitTestCase(unittest.TestCase):
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
        pos = nx.planar_layout(self.bayesian_network)
        nx.draw(self.bayesian_network, pos=pos, with_labels=True)
        plt.show()

    def test_likelihood(self):
        event = [0, 1, 1]
        likelihood = self.bayesian_network.likelihood(event, )
        self.assertEqual(likelihood, 0.7)

        event = [1, 2, 2]
        likelihood = self.bayesian_network.likelihood(event)
        self.assertEqual(likelihood, 0.3 * 0.5 * 1 / 3)

    def test_probability(self):
        event = Event({self.x: [0, 1], self.y: portion.closed(1.5, 2)})
        probability = self.bayesian_network.probability(event)
        self.assertEqual(probability, 0.3 * 0.25)

    def test_as_probabilistic_circuit(self):
        circuit = self.bayesian_network.as_probabilistic_circuit().simplify()
        self.assertEqual(circuit.probability(Event()), 1.)
        event = Event({self.x: [0, 1], self.y: portion.closed(1.5, 2)})
        self.assertAlmostEqual(self.bayesian_network.probability(event), circuit.probability(event))


class BayesianNetworkWrongOrderTestCase(unittest.TestCase):

    x: Symbolic = Symbolic("x", [0, 1])
    y: Symbolic = Symbolic("y", [0, 1])

    p_y: DiscreteDistribution
    p_x_y: ConditionalProbabilityTable

    model: BayesianNetwork

    def setUp(self):
        self.p_y = DiscreteDistribution(self.y, [0.5, 0.5])
        self.p_x_y = ConditionalProbabilityTable(self.x)
        self.p_x_y.conditional_probability_distributions[(0,)] = DiscreteDistribution(self.x, [0.7, 0.3])
        self.p_x_y.conditional_probability_distributions[(1,)] = DiscreteDistribution(self.x, [0.3, 0.7])
        self.model = BayesianNetwork()
        self.model.add_node(self.p_x_y)
        self.model.add_node(self.p_y)

        self.model.add_edge(self.p_y, self.p_x_y)

    def test_forward_pass(self):
        event = self.model.preprocess_event(Event())
        self.model.forward_pass(event)

if __name__ == '__main__':
    unittest.main()
