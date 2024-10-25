import unittest

import networkx as nx
import numpy as np
import tabulate
from matplotlib import pyplot as plt
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.set import SetElement, Set
from random_events.variable import Symbolic, Continuous

from probabilistic_model.bayesian_network.bayesian_network import BayesianNetwork
from probabilistic_model.bayesian_network.distributions import (ConditionalProbabilityTable, RootDistribution,
                                                                ConditionalProbabilisticCircuit)
from probabilistic_model.distributions import SymbolicDistribution, UniformDistribution
from probabilistic_model.probabilistic_circuit.nx.distributions import UnivariateContinuousLeaf
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import  SumUnit, ProductUnit
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


class DistributionTestCase(unittest.TestCase):
    x = Symbolic("x", XEnum)
    y = Symbolic("y", YEnum)

    p_x = ConditionalProbabilityTable(x)
    p_yx = ConditionalProbabilityTable(y)

    def setUp(self):
        bayesian_network = BayesianNetwork()

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
        bayesian_network.add_node(self.p_x)
        bayesian_network.add_node(self.p_yx)

        # add the edge between x and y
        bayesian_network.add_edge(self.p_x, self.p_yx)

    def test_to_tabulate(self):
        table = tabulate.tabulate(self.p_yx.to_tabulate())
        self.assertIsInstance(table, str)  # print(table)

    # def test_likelihood(self):
    #     self.assertEqual(self.p_yx.likelihood([0, 1]), 0.5)

    def test_forward_pass(self):
        event = SimpleEvent({self.x: Set(XEnum.ZERO, XEnum.ONE), self.y: YEnum.ZERO})
        self.p_x.forward_pass(event)

        self.assertEqual(list(self.p_x.forward_message.probabilities.values()), [0.5 / 0.8, 0.3 / 0.8])
        self.assertEqual(self.p_x.forward_probability, 0.8)

        self.p_yx.forward_pass(event)
        self.assertEqual(list(self.p_yx.forward_message.probabilities.values()), [1.])
        self.assertEqual(self.p_yx.forward_probability, 0.5 / 0.8 * 0.5 + 0.3 / 0.8 * 0.3)

    def test_forward_pass_impossible_event(self):
        self.p_x.probabilities = MissingDict(float, zip([0], [1.]))
        event = SimpleEvent({self.x: Set(XEnum.ONE), self.y: self.y.domain})

        self.p_x.forward_pass(event)
        self.assertIsNone(self.p_x.forward_message)
        self.assertEqual(self.p_x.forward_probability, 0)

        self.p_yx.forward_pass(event)
        self.assertIsNone(self.p_yx.forward_message)
        self.assertEqual(self.p_yx.forward_probability, 0)

    def test_joint_distribution_with_parents_root(self):
        event = SimpleEvent({variable: variable.domain for variable in [self.x, self.y]})

        self.p_x.forward_pass(event)

        joint_distribution = self.p_x.joint_distribution_with_parent()
        self.assertIsInstance(joint_distribution, SumUnit)

    def test_joint_distribution_with_parents(self):
        event = SimpleEvent({variable: variable.domain for variable in [self.x, self.y]})

        self.p_x.bayesian_network.forward_pass(event)

        joint_distribution = self.p_yx.joint_distribution_with_parent()
        self.assertIsInstance(joint_distribution, SumUnit)

        likelihoods = joint_distribution.probabilistic_circuit.likelihood(np.array([[0, 1], [2, 1]]))
        self.assertAlmostEqual(likelihoods[0], 0.25)
        self.assertAlmostEqual(likelihoods[1], 0.2 * 0.9)


class CircuitDistributionTestCase(unittest.TestCase):
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
        nx.draw(self.bayesian_network, with_labels=True)
        plt.show()

    def test_forward_pass(self):
        event = SimpleEvent({variable: variable.domain for variable in self.bayesian_network.variables})
        self.bayesian_network.forward_pass(event)
        self.assertEqual(self.p_x.forward_probability, 1)
        self.assertEqual(self.p_yzx.forward_probability, 1)

    def test_joint_distribution_with_parent(self):
        event = SimpleEvent({variable: variable.domain for variable in self.bayesian_network.variables})
        self.bayesian_network.forward_pass(event)

        joint_distribution = self.p_yzx.joint_distribution_with_parent().probabilistic_circuit
        event = SimpleEvent({self.x: YEnum.ZERO, self.y: closed(0, 0.5)}).as_composite_set()
        self.assertEqual(joint_distribution.probability(event), 0.35)


if __name__ == '__main__':
    unittest.main()
