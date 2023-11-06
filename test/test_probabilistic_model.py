import unittest
from typing import Iterable

from random_events.events import Event, EncodedEvent
from random_events.variables import Symbolic, Discrete, Continuous, Variable

from probabilistic_model.probabilistic_model import (ProbabilisticModel)


class ProbabilisticModelTestDouble(ProbabilisticModel):
    """
    Test double implementation of a probabilistic model for test purposes.
    """

    def __init__(self):
        super().__init__([Symbolic("a", ("x", "y", "z")), Discrete("b", (1, 2, 4)), Continuous("c")])

    def _likelihood(self, event):
        return 1.0

    def _probability(self, event):
        return 1.0

    def _mode(self):
        result = [EncodedEvent({self.variables[0]: 0, self.variables[1]: 0, self.variables[2]: 0}),
                  EncodedEvent({self.variables[0]: 1, self.variables[1]: 1, self.variables[2]: 1})]
        return result, 1.0

    def marginal(self, variables: Iterable[Variable]) -> 'ProbabilisticModel':
        return ProbabilisticModelTestDouble()

    def _conditional(self, event: EncodedEvent) -> 'ProbabilisticModel':
        return ProbabilisticModelTestDouble()

    def sample(self, amount: int) -> Iterable:
        return [["a", 1, 5], ["b", 2, 5]]


class ProbabilisticModelWithoutImplementationTestCase(unittest.TestCase):
    model: ProbabilisticModel
    event: Event

    @classmethod
    def setUpClass(cls):
        cls.model = ProbabilisticModel([Symbolic("a", ("x", "y", "z")), Discrete("b", (1, 2, 4)), Continuous("c")])
        cls.event = Event({cls.model.variables[0]: "x", cls.model.variables[2]: 1.0})

    def test_likelihood(self):
        with self.assertRaises(NotImplementedError):
            self.model.likelihood(["x", 1, 5])

    def test_probability(self):
        with self.assertRaises(NotImplementedError):
            self.model.probability(self.event)

    def test_mode(self):
        with self.assertRaises(NotImplementedError):
            self.model.mode()

    def test_marginal(self):
        with self.assertRaises(NotImplementedError):
            self.model.marginal(self.model.variables)

    def test_conditional(self):
        with self.assertRaises(NotImplementedError):
            self.model.conditional(self.event)

    def test_sample(self):
        with self.assertRaises(NotImplementedError):
            self.model.sample(10)


class ProbabilisticModelTestDoubleTestCase(unittest.TestCase):
    model: ProbabilisticModel
    event: Event

    @classmethod
    def setUpClass(cls):
        cls.model = ProbabilisticModelTestDouble()
        cls.event = Event({cls.model.variables[0]: "x", cls.model.variables[2]: 1.0})

    def test_likelihood(self):
        self.assertEqual(self.model.likelihood(["x", 1, 5]), 1.0)

    def test_probability(self):
        self.assertEqual(self.model.probability(self.event), 1.0)

    def test_mode(self):
        mode, likelihood = self.model.mode()
        self.assertEqual(len(mode), 2)
        self.assertEqual(likelihood, 1.0)
        [self.assertIsInstance(event, Event) for event in mode]

    def test_marginal(self):
        self.assertEqual(self.model.variables, self.model.marginal(self.model.variables).variables)

    def test_conditional(self):
        self.assertEqual(self.model.variables, self.model.conditional(self.event).variables)

    def test_sample(self):
        self.assertEqual(self.model.sample(2), [["a", 1, 5], ["b", 2, 5]])


if __name__ == '__main__':
    unittest.main()
