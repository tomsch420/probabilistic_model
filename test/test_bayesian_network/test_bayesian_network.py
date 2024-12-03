import math
import unittest

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.set import SetElement, Set
from random_events.variable import Symbolic, Continuous

from probabilistic_model.bayesian_network.bayesian_network import BayesianNetwork
from probabilistic_model.bayesian_network.distributions import (ConditionalProbabilityTable, RootDistribution,
                                                                ConditionalProbabilisticCircuit)
from probabilistic_model.distributions.multinomial import MultinomialDistribution
from probabilistic_model.distributions import UniformDistribution, SymbolicDistribution
from probabilistic_model.probabilistic_circuit.nx.distributions import UnivariateContinuousLeaf
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import ProductUnit
from probabilistic_model.utils import MissingDict


class YEnum(SetElement):
    EMPTY_SET = -1
    ZERO = 0
    ONE = 1


class XEnum(SetElement):
    EMPTY_SET = -1
    ZERO = 0
    ONE = 1
    TWO = 2


class MinimalBayesianNetworkTestCase(unittest.TestCase):
    model: BayesianNetwork
    x: Symbolic = Symbolic('x', XEnum)
    y: Symbolic = Symbolic('y', YEnum)
    p_x: RootDistribution
    p_yx: ConditionalProbabilityTable = ConditionalProbabilityTable(y)

    bf_distribution: MultinomialDistribution

    def setUp(self):
        np.random.seed(69)

        self.model = BayesianNetwork()

        # create the root distribution for x
        self.p_x = RootDistribution(self.x, MissingDict(float, zip([0, 1, 2], [0.5, 0.3, 0.2])))

        # create the conditional probability table for y
        self.p_yx.conditional_probability_distributions[0] = (
            SymbolicDistribution(self.y, MissingDict(float, zip([0, 1], [0.5, 0.5]))))
        self.p_yx.conditional_probability_distributions[1] = (
            SymbolicDistribution(self.y, MissingDict(float, zip([0, 1], [0.3, 0.7]))))
        self.p_yx.conditional_probability_distributions[2] = (
            SymbolicDistribution(self.y, MissingDict(float, zip([0, 1], [0.1, 0.9]))))

        # add the distributions to the bayesian network
        self.model.add_node(self.p_x)
        self.model.add_node(self.p_yx)

        # add the edge between x and y
        self.model.add_edge(self.p_x, self.p_yx)

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

    def test_as_probabilistic_circuit(self):
        circuit = self.model.as_probabilistic_circuit()
        self.assertIsNotNone(circuit)


class ComplexBayesianNetworkTestCase(unittest.TestCase):
    model: BayesianNetwork
    x: Symbolic = Symbolic('x', XEnum)
    y: Symbolic = Symbolic('y', YEnum)
    z: Symbolic = Symbolic('z', YEnum)
    a: Symbolic = Symbolic('a', YEnum)

    d_x: RootDistribution
    d_yx = ConditionalProbabilityTable(y)
    d_zx = ConditionalProbabilityTable(z)
    d_az = ConditionalProbabilityTable(a)

    def setUp(self):
        np.random.seed(69)

        self.model = BayesianNetwork()

        self.d_x = RootDistribution(self.x, MissingDict(float, zip([0, 1, 2], [0.5, 0.3, 0.2])))

        self.model.add_nodes_from([self.d_x, self.d_yx, self.d_zx, self.d_az])
        self.model.add_edge(self.d_x, self.d_yx)
        self.model.add_edge(self.d_x, self.d_zx)
        self.model.add_edge(self.d_zx, self.d_az)

        self.d_yx.conditional_probability_distributions[0] = (
            SymbolicDistribution(self.y, MissingDict(float, zip([0, 1], [0.5, 0.5]))))
        self.d_yx.conditional_probability_distributions[1] = (
            SymbolicDistribution(self.y, MissingDict(float, zip([0, 1], [0.6, 0.4]))))
        self.d_yx.conditional_probability_distributions[2] = (
            SymbolicDistribution(self.y, MissingDict(float, zip([0, 1], [0.7, 0.3]))))

        self.d_zx.conditional_probability_distributions[0] = (
            SymbolicDistribution(self.z, MissingDict(float, zip([0, 1], [0.1, 0.9]))))
        self.d_zx.conditional_probability_distributions[1] = (
            SymbolicDistribution(self.z, MissingDict(float, zip([0, 1], [1, 0]))))
        self.d_zx.conditional_probability_distributions[2] = (
            SymbolicDistribution(self.z, MissingDict(float, zip([0, 1], [0.8, 0.2]))))

        self.d_az.conditional_probability_distributions[0] = (
            SymbolicDistribution(self.a, MissingDict(float, zip([0, 1], [0.3, 0.7]))))
        self.d_az.conditional_probability_distributions[1] = (
            SymbolicDistribution(self.a, MissingDict(float, zip([0, 1], [0.6, 0.4]))))

    def plot(self):
        pos = nx.planar_layout(self.model)
        nx.draw(self.model, pos=pos, with_labels=True, labels={node: repr(node) for node in self.model.nodes})
        plt.show()

    def test_as_probabilistic_circuit(self):
        circuit = self.model.as_probabilistic_circuit().simplify()
        self.assertLess(len(circuit.weighted_edges), math.prod([len(v.domain.simple_sets) for v in circuit.variables]))


class BayesianNetworkWithCircuitTestCase(unittest.TestCase):
    x: Symbolic = Symbolic("x", YEnum)
    y: Continuous = Continuous("y")
    z: Continuous = Continuous("z")
    p_x: RootDistribution
    p_yzx = ConditionalProbabilisticCircuit([y, z])
    bayesian_network: BayesianNetwork

    def setUp(self):
        self.bayesian_network = BayesianNetwork()
        self.p_x = RootDistribution(self.x, MissingDict(float, zip([0, 1], [0.7, 0.3])))

        d1 = ProductUnit()
        d1.add_subcircuit(UnivariateContinuousLeaf(UniformDistribution(self.y, closed(0, 1).simple_sets[0])))
        d1.add_subcircuit(UnivariateContinuousLeaf(UniformDistribution(self.z, closed(0, 1).simple_sets[0])))

        d2 = ProductUnit()
        d2.add_subcircuit(UnivariateContinuousLeaf(UniformDistribution(self.y, closed(0, 2).simple_sets[0])))
        d2.add_subcircuit(UnivariateContinuousLeaf(UniformDistribution(self.z, closed(0, 3).simple_sets[0])))

        self.p_yzx.conditional_probability_distributions[0] = d1.probabilistic_circuit
        self.p_yzx.conditional_probability_distributions[1] = d2.probabilistic_circuit

        self.bayesian_network.add_nodes_from([self.p_x, self.p_yzx])
        self.bayesian_network.add_edge(self.p_x, self.p_yzx)

    def plot(self):
        pos = nx.planar_layout(self.bayesian_network)
        nx.draw(self.bayesian_network, pos=pos, with_labels=True)
        plt.show()

    def test_as_probabilistic_circuit(self):
        circuit = self.bayesian_network.as_probabilistic_circuit()
        circuit.simplify()
        self.assertEqual(circuit.probability(circuit.universal_simple_event().as_composite_set()), 1.)
        event = SimpleEvent({self.x: Set(XEnum.ZERO, XEnum(1)), self.y: closed(1.5, 2)})
        self.assertAlmostEqual(0.075, circuit.probability(event.as_composite_set()))


if __name__ == '__main__':
    unittest.main()
