import unittest
from typing import Iterable, Tuple

import portion
from anytree import PreOrderIter
from random_events.events import EncodedEvent, Event
from random_events.variables import Symbolic, Integer, Continuous, Variable

from probabilistic_model.probabilistic_circuit.units import Unit, DeterministicSumUnit

from typing_extensions import Self


class DummyDistribution(Unit):

    def __init__(self, variable: Variable):
        super().__init__([variable])

    def _likelihood(self, event: Iterable) -> float:
        return 2


class DummySymbolicDistribution(Unit):

    def __init__(self):
        super().__init__((Symbolic('symbol', ('a', 'b', 'c')),))

    def _likelihood(self, event: Iterable) -> float:
        return 0.5

    def _probability(self, event: EncodedEvent) -> float:
        return 0.5

    def _mode(self) -> Tuple[Iterable[EncodedEvent], float]:
        mode = EncodedEvent({self.variables[0]: 0})
        return [mode], 0.5

    def _conditional(self, event: EncodedEvent) -> Tuple['DummySymbolicDistribution', float]:
        return DummySymbolicDistribution(), 0.5


class DummyRealDistribution(Unit):

    def __init__(self):
        super().__init__((Continuous("real"),))

    def _likelihood(self, event: Iterable) -> float:
        return 2.

    def _probability(self, event: EncodedEvent) -> float:
        return 2.

    def _mode(self) -> Tuple[Iterable[EncodedEvent], float]:
        mode = EncodedEvent({self.variables[0]: portion.open(1., 2.)})
        return [mode], 2.

    def _conditional(self, event: EncodedEvent) -> Tuple['DummyRealDistribution', float]:
        return DummyRealDistribution(), 2.


class DummyIntegerDistribution(Unit):

    def __init__(self):
        super().__init__((Integer('integer', (1, 2, 4)),))

    def _likelihood(self, event: Iterable) -> float:
        return 3.

    def _mode(self) -> Tuple[Iterable[EncodedEvent], float]:
        mode = EncodedEvent({self.variables[0]: 2})
        return [mode], 3

    def _conditional(self, event: EncodedEvent) -> Tuple['DummyIntegerDistribution', float]:
        return DummyIntegerDistribution(), 3.


class UnitCreationTestCase(unittest.TestCase):
    symbol = Symbolic('symbol', ('a', 'b', 'c'))
    integer = Integer('integer', (1, 2, 4))
    real = Continuous("real")
    variables = (integer, real, symbol)

    def test_node_creation(self):
        unit = Unit(self.variables)
        self.assertEqual(unit.variables, self.variables)
        self.assertEqual(unit.parent, None)
        self.assertEqual(unit.children, ())

    def test_node_creation_with_parent(self):
        node = Unit(self.variables)
        child = Unit(self.variables, node)
        self.assertEqual(child.parent, node)
        self.assertEqual(node.children, (child,))

    def test_unit_creation_by_summation(self):
        unit = DummyDistribution(self.symbol) * DummyDistribution(self.real) * DummyDistribution(self.integer)
        self.assertEqual(unit.variables, self.variables)
        self.assertEqual(unit.likelihood([1, 2, "a"]), 2 ** 3)
        leaves = list(PreOrderIter(unit, filter_=lambda node: node.is_leaf))
        self.assertEqual(len(leaves), 3)


class UnitInferenceTestCase(unittest.TestCase):

    def test_mode_real_distribution_unit(self):
        distribution = DummyRealDistribution()
        mode, likelihood = distribution.mode()
        self.assertEqual(likelihood, 2.)
        self.assertEqual(len(mode), 1)
        self.assertEqual(mode[0], Event({distribution.variables[0]: portion.open(1., 2.)}))

    def test_mode_integer_distribution_unit(self):
        distribution = DummyIntegerDistribution()
        mode, likelihood = distribution.mode()
        self.assertEqual(likelihood, 3.)
        self.assertEqual(len(mode), 1)
        self.assertEqual(mode[0], Event({distribution.variables[0]: 4}))

    def test_mode_symbolic_distribution_unit(self):
        distribution = DummySymbolicDistribution()
        mode, likelihood = distribution.mode()
        self.assertEqual(likelihood, 0.5)
        self.assertEqual(len(mode), 1)
        self.assertEqual(mode[0], Event({distribution.variables[0]: "a"}))

    def test_likelihood_of_product_unit(self):
        distribution = DummyRealDistribution() * DummySymbolicDistribution()
        self.assertEqual(distribution.likelihood([1., "a"]), 1.)
        distribution *= DummyIntegerDistribution()
        self.assertEqual(distribution.likelihood([1, 1., "a"]), 3.)

    def test_mode_of_product_unit(self):
        distribution = DummyRealDistribution() * DummySymbolicDistribution() * DummyIntegerDistribution()
        mode, likelihood = distribution.mode()
        self.assertEqual(likelihood, 3.)
        self.assertEqual(len(mode), 1)
        self.assertEqual(mode[0], Event({distribution.variables[0]: 4,
                                         distribution.variables[1]: portion.open(1., 2.),
                                         distribution.variables[2]: "a"}))

    def test_likelihood_of_continuous_sum_unit(self):
        distribution = DummyRealDistribution() + DummyRealDistribution()
        self.assertEqual(distribution.likelihood([1.]), 2.)

    def test_mode_of_continuous_deterministic_sum_unit(self):
        distribution = DummyRealDistribution() + DummyRealDistribution()
        distribution = DeterministicSumUnit.from_sum_unit(distribution)
        mode, likelihood = distribution.mode()
        self.assertEqual(likelihood, 1.)
        self.assertEqual(len(mode), 1)
        self.assertEqual(mode[0], Event({distribution.variables[0]: portion.open(1., 2.)}))

    def test_likelihood_of_integer_sum_unit(self):
        distribution = DummyIntegerDistribution() + DummyIntegerDistribution()
        self.assertEqual(distribution.likelihood([1]), 3.)

    def test_mode_of_integer_deterministic_sum_unit(self):
        distribution = DummyIntegerDistribution() + DummyIntegerDistribution()
        distribution = DeterministicSumUnit.from_sum_unit(distribution)
        mode, likelihood = distribution.mode()
        self.assertEqual(likelihood, 1.5)
        self.assertEqual(len(mode), 1)
        self.assertEqual(mode[0], Event({distribution.variables[0]: 4}))

    def test_likelihood_of_symbolic_sum_unit(self):
        distribution = DummySymbolicDistribution() + DummySymbolicDistribution()
        self.assertEqual(distribution.likelihood(["a"]), 0.5)

    def test_mode_of_symbolic_deterministic_sum_unit(self):
        distribution = DummySymbolicDistribution() + DummySymbolicDistribution()
        distribution = DeterministicSumUnit.from_sum_unit(distribution)
        mode, likelihood = distribution.mode()
        self.assertEqual(likelihood, 0.25)
        self.assertEqual(len(mode), 1)
        self.assertEqual(mode[0], Event({distribution.variables[0]: "a"}))

    def test_likelihood_of_mixed_sum_unit(self):
        distribution = ((DummySymbolicDistribution() * DummyRealDistribution() * DummyIntegerDistribution()) +
                        (DummySymbolicDistribution() * DummyRealDistribution() * DummyIntegerDistribution()))
        self.assertEqual(distribution.likelihood([1, 1., "a"]), 3.)

    def test_mode_of_mixed_deterministic_sum_unit(self):
        distribution = ((DummySymbolicDistribution() * DummyRealDistribution() * DummyIntegerDistribution()) +
                        (DummySymbolicDistribution() * DummyRealDistribution() * DummyIntegerDistribution()))
        distribution = DeterministicSumUnit.from_sum_unit(distribution)
        distribution.weights = [0.7, 0.3]
        mode, likelihood = distribution.mode()
        self.assertEqual(likelihood, 3 * 0.7)
        self.assertEqual(len(mode), 1)
        self.assertEqual(mode[0], Event({distribution.variables[0]: 4,
                                         distribution.variables[1]: portion.open(1., 2.),
                                         distribution.variables[2]: "a"}))

    def test_conditional_of_real_mixture_distribution_unit(self):
        distribution = DummyRealDistribution() + DummyRealDistribution()
        event = Event({distribution.variables[0]: portion.open(1., 2.)})
        conditional, probability = distribution.conditional(event)
        self.assertEqual(probability, 2.)
        self.assertEqual(len(conditional.variables), 1)
        self.assertEqual(conditional.weights, [.5, .5])

    def test_conditional_of_real_product_distribution_unit(self):
        distribution = DummyRealDistribution() * DummySymbolicDistribution()
        event = Event({distribution.variables[0]: portion.open(1., 2.)})
        conditional, probability = distribution.conditional(event)
        self.assertEqual(probability, 1)
        self.assertEqual(len(conditional.children), len(distribution.children))


if __name__ == '__main__':
    unittest.main()
