import itertools
import unittest

from random_events.events import Event

from probabilistic_model.bayesian_network import BayesianNetwork, ConditionalMultinomialDistribution
from probabilistic_model.distributions.multinomial import MultinomialDistribution
from random_events.variables import Symbolic
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx


class MinimalBayesianNetworkTestCase(unittest.TestCase):

    model: BayesianNetwork
    x: Symbolic = Symbolic('x', [0, 1, 2])
    y: Symbolic = Symbolic('y', [0, 1])
    d_x: ConditionalMultinomialDistribution
    d_xy: ConditionalMultinomialDistribution

    bf_distribution: MultinomialDistribution

    def setUp(self):
        np.random.seed(69)

        self.model = BayesianNetwork()

        self.d_x = ConditionalMultinomialDistribution([self.x])
        self.d_xy = ConditionalMultinomialDistribution([self.y])
        self.model.add_nodes_from([self.d_x, self.d_xy])
        self.model.add_edge(self.d_x, self.d_xy)

        self.d_x.probabilities = np.array([0.5, 0.3, 0.2])
        self.d_xy.probabilities = np.array([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]])

        self.bf_distribution = self.model.brute_force_joint_distribution()

    def plot(self):
        pos = nx.planar_layout(self.model)
        nx.draw(self.model, pos=pos, with_labels=True, labels={node: repr(node) for node in self.model.nodes})
        plt.show()

    def test_setup(self):
        self.assertEqual(len(self.model.nodes), 2)
        self.assertEqual(len(self.model.edges), 1)

    def test_parents(self):
        self.assertEqual(list(self.model.nodes)[1].parents, [list(self.model.nodes)[0]])

    def test_variables(self):
        self.assertEqual(self.model.variables, (self.x, self.y))

    def test_parent_variables(self):
        self.assertEqual(self.d_xy.parent_variables, (self.x,))

    def test_normalize(self):
        probabilities_before = self.d_xy.probabilities
        self.d_xy.probabilities = np.array([[5, 5], [6, 4], [7, 3]])
        self.d_xy.normalize()
        self.assertTrue(np.all(probabilities_before == self.d_xy.probabilities))

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
        self.assertEqual(likelihood, 0.3 * 0.6)

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
            self.assertAlmostEqual(self.model.likelihood(event),
                                   circuit.likelihood(event))


class ComplexBayesianNetworkTestCase(unittest.TestCase):

    model: BayesianNetwork
    x: Symbolic = Symbolic('x', [0, 1, 2])
    y: Symbolic = Symbolic('y', [0, 1])
    z: Symbolic = Symbolic('z', [0, 1])
    a: Symbolic = Symbolic('a', [0, 1])

    d_x: ConditionalMultinomialDistribution
    d_yx: ConditionalMultinomialDistribution
    d_zx: ConditionalMultinomialDistribution
    d_az: ConditionalMultinomialDistribution

    bf_distribution: MultinomialDistribution

    def setUp(self):
        np.random.seed(69)

        self.model = BayesianNetwork()

        self.d_x = ConditionalMultinomialDistribution([self.x])
        self.d_yx = ConditionalMultinomialDistribution([self.y])
        self.d_zx = ConditionalMultinomialDistribution([self.z])
        self.d_az = ConditionalMultinomialDistribution([self.a])

        self.model.add_nodes_from([self.d_x, self.d_yx, self.d_zx, self.d_az])
        self.model.add_edge(self.d_x, self.d_yx)
        self.model.add_edge(self.d_x, self.d_zx)
        self.model.add_edge(self.d_zx, self.d_az)

        self.d_x.probabilities = np.array([0.5, 0.3, 0.2])
        self.d_yx.probabilities = np.array([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]])
        self.d_zx.probabilities = np.array([[0.1, 0.9], [1., 0.], [0.8, 0.2]])
        self.d_az.probabilities = np.array([[0.3, 0.7], [0.6, 0.4]])

        self.bf_distribution = self.model.brute_force_joint_distribution()

    def plot(self):
        pos = nx.planar_layout(self.model)
        nx.draw(self.model, pos=pos, with_labels=True, labels={node: repr(node) for node in self.model.nodes})
        plt.show()

    def test_likelihood(self):
        for event in itertools.product(*[variable.domain for variable in self.model.variables]):
            self.assertEqual(self.model.likelihood(event), self.bf_distribution.likelihood(event))

    def test_probability(self):
        self.plot()
        event = Event({self.x: [1], self.y: [0, 1], self.z: [0]})
        probability = self.model.probability(event)
        self.assertEqual(probability, self.bf_distribution.probability(event))

    def test_as_probabilistic_circuit(self):
        circuit = self.model.as_probabilistic_circuit()
        for event in itertools.product(self.a.domain, self.x.domain, self.y.domain, self.z.domain):
            self.assertAlmostEqual(self.model.likelihood(event),
                                   circuit.likelihood(event))


if __name__ == '__main__':
    unittest.main()
