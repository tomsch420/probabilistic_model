import unittest

import portion
from matplotlib import pyplot as plt
from random_events.events import Event
from random_events.variables import Integer, Symbolic, Continuous

from probabilistic_model.probabilistic_circuit.probabilistic_circuit import *

from probabilistic_model.probabilistic_circuit.distributions.distributions import (UniformDistribution,
                                                                                   SymbolicDistribution,
                                                                                   IntegerDistribution)


class UniformDistributionTestCase(unittest.TestCase):

    variable = Continuous("x")
    model: UniformDistribution

    def setUp(self):
        self.model = UniformDistribution(self.variable, portion.closed(0, 1))

    def show(self):
        nx.draw(self.model.probabilistic_circuit, with_labels=True)
        plt.show()

    def test_conditional_from_simple_event(self):
        event = Event({self.variable: portion.closed(0.5, 2)})
        conditional, probability = self.model.conditional(event)
        self.assertEqual(len(list(conditional.probabilistic_circuit.nodes)), 1)
        self.assertEqual(probability, 0.5)
        self.assertEqual(conditional.interval, portion.closed(0.5, 1))

    def test_conditional_from_singleton_event(self):
        event = Event({self.variable: portion.singleton(0.3)})
        conditional, probability = self.model.conditional(event)
        self.assertEqual(len(conditional.probabilistic_circuit.nodes), 1)
        self.assertEqual(probability, 1.)
        self.assertEqual(conditional.location, 0.3)

    def test_conditional_from_complex_event(self):
        interval = portion.closed(0., 0.2) | portion.closed(0.5, 1.) | portion.singleton(0.3)
        event = Event({self.variable: interval})
        model, likelihood = self.model.conditional(event)
        self.assertEqual(len(list(model.probabilistic_circuit.nodes)), 4)
        self.assertIsInstance(model.probabilistic_circuit.root, DeterministicSumUnit)

    def test_conditional_with_none(self):
        event = Event({self.variable: 2})
        conditional, probability = self.model.conditional(event)
        self.assertEqual(conditional, None)


class DiscreteDistributionTestCase(unittest.TestCase):

    symbol = Symbolic("animal", ["cat", "dog", "fish"])
    integer = Integer("x", list(range(3)))

    symbolic_distribution: SymbolicDistribution
    integer_distribution: IntegerDistribution

    def test_creation(self):
        circuit = ProbabilisticCircuit()
        self.symbolic_distribution = SymbolicDistribution(self.symbol, [0.1, 0.2, 0.7])
        self.integer_distribution = IntegerDistribution(self.integer, [0.1, 0.2, 0.7])
        circuit.add_node(self.symbolic_distribution)
        circuit.add_node(self.integer_distribution)
        self.assertEqual(len(list(circuit.nodes)), 2)

