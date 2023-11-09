import unittest
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution, SymbolicDistribution, \
    IntegerDistribution
from probabilistic_model.probabilistic_circuit.units import DeterministicSumUnit
from random_events.events import Event, VariableMap
from random_events.variables import Continuous, Symbolic, Integer
import portion


class UniformDistributionTestCase(unittest.TestCase):
    distribution: UniformDistribution = UniformDistribution(Continuous("x"), 0, 2)

    def test_creation_with_raises(self):
        with self.assertRaises(ValueError):
            UniformDistribution(Continuous("x"), 1, 0)
        with self.assertRaises(ValueError):
            UniformDistribution(Continuous("x"), 0, 0)

    def test_likelihood(self):
        self.assertEqual(self.distribution.likelihood([1]), 0.5)
        self.assertEqual(self.distribution.likelihood([0]), 0.5)
        self.assertEqual(self.distribution.likelihood([2]), 0.)
        self.assertEqual(self.distribution.likelihood([-1]), 0.)
        self.assertEqual(self.distribution.likelihood([3]), 0.)

    def test_probability_of_domain(self):
        self.assertEqual(self.distribution.probability(Event({self.distribution.variable: self.distribution.domain})),
                         1)

    def test_cdf(self):
        self.assertEqual(self.distribution.cdf(1), 0.5)
        self.assertEqual(self.distribution.cdf(0), 0)
        self.assertEqual(self.distribution.cdf(2), 1.)
        self.assertEqual(self.distribution.cdf(-1), 0.)
        self.assertEqual(self.distribution.cdf(3), 1)

    def test_probability(self):
        event = Event({self.distribution.variable: portion.closed(0, 1) | portion.closed(1.5, 2)})
        self.assertEqual(self.distribution.probability(event), 0.75)

    def test_mode(self):
        modes, likelihood = self.distribution.mode()
        self.assertEqual(modes, [Event({self.distribution.variable: self.distribution.domain})])
        self.assertEqual(likelihood, 0.5)

    def test_sample(self):
        samples = self.distribution.sample(100)
        self.assertEqual(len(samples), 100)
        for sample in samples:
            self.assertGreaterEqual(self.distribution.likelihood(sample), 0)

    def test_conditional_no_intersection(self):
        event = Event({self.distribution.variable: portion.closed(3, 4)})
        conditional, probability = self.distribution.conditional(event)
        self.assertIsNone(conditional)
        self.assertEqual(probability, 0)

    def test_conditional_simple_intersection(self):
        event = Event({self.distribution.variable: portion.closed(1, 2)})
        conditional, probability = self.distribution.conditional(event)
        self.assertIsInstance(conditional, UniformDistribution)
        self.assertEqual(probability, 0.5)
        self.assertEqual(conditional.lower, 1)
        self.assertEqual(conditional.upper, 2)

    def test_conditional_complex_intersection(self):
        event = Event({self.distribution.variable: portion.closed(1.5, 2) | portion.closed(0, 1)})
        conditional, probability = self.distribution.conditional(event)
        self.assertIsInstance(conditional, DeterministicSumUnit)
        self.assertEqual(probability, 0.75)
        self.assertEqual(len(conditional.children), 2)
        self.assertEqual(conditional.weights, [2 / 3, 1 / 3])
        self.assertEqual(conditional.children[0].domain, portion.closedopen(0, 1))
        self.assertEqual(conditional.children[1].domain, portion.closedopen(1.5, 2))

    def test_conditional_triple_complex_intersection(self):
        event = Event(
            {self.distribution.variable: portion.closed(1.5, 2) |
                                         portion.closed(0, 0.25) |
                                         portion.closed(0.75, 1)})

        conditional, probability = self.distribution.conditional(event)
        self.assertIsInstance(conditional, DeterministicSumUnit)
        self.assertEqual(probability, 0.5)
        self.assertEqual(len(conditional.children), 3)
        self.assertEqual(conditional.weights, [1 / 4, 1 / 4, 1 / 2])
        self.assertEqual(conditional.children[0].domain, portion.closedopen(0, 0.25))
        self.assertEqual(conditional.children[1].domain, portion.closedopen(0.75, 1))
        self.assertEqual(conditional.children[2].domain, portion.closedopen(1.5, 2))

    def test_conditional_mode(self):
        event = Event(
            {self.distribution.variable: portion.closedopen(1.5, 2) |
                                         portion.closedopen(0, 0.25) |
                                         portion.closedopen(0.75, 1)})

        conditional, probability = self.distribution.conditional(event)
        modes, likelihood = conditional.mode()
        self.assertEqual(len(modes), 1)
        self.assertEqual(modes[0][self.distribution.variable], event[conditional.variables[0]])
        self.assertEqual(likelihood, 1.)

    def test_moment(self):
        expectation = self.distribution.moment(VariableMap({self.distribution.variable: 1}),
                                               VariableMap({self.distribution.variable: 0}))
        self.assertEqual(expectation[self.distribution.variable], 1)
        variance = self.distribution.moment(VariableMap({self.distribution.variable: 2}), expectation)
        self.assertEqual(variance[self.distribution.variable], 1 / 3)


class SymbolicDistributionTestCase(unittest.TestCase):
    distribution: SymbolicDistribution = SymbolicDistribution(Symbolic("animal", {"cat", "dog", "chicken"}),
                                                              [0.3, 0.3, 0.4])

    def test_creating_with_invalid_weights(self):
        with self.assertRaises(ValueError):
            SymbolicDistribution(self.distribution.variable, [0, 1])

    def test_pdf(self):
        self.assertEqual(self.distribution.pdf("cat"), 0.3)

    def test_likelihood(self):
        self.assertEqual(self.distribution.likelihood(["cat"]), 0.3)

    def test_probability(self):
        event = Event({self.distribution.variable: ["cat", "dog"]})
        self.assertEqual(self.distribution.probability(event), 0.7)

    def test_mode(self):
        mode, likelihood = self.distribution.mode()
        self.assertEqual(likelihood, 0.4)
        self.assertEqual(mode, [Event({self.distribution.variable: "dog"})])

    def test_conditional(self):
        event = Event({self.distribution.variable: ["cat", "dog"]})
        conditional, probability = self.distribution.conditional(event)
        self.assertEqual(probability, 0.7)
        self.assertEqual(conditional, SymbolicDistribution(self.distribution.variable, [0.3 / 0.7, 0, 0.4 / 0.7]))

    def test_conditional_impossible(self):
        event = Event({self.distribution.variable: []})
        conditional, probability = self.distribution.conditional(event)
        self.assertIsNone(conditional)
        self.assertEqual(probability, 0)

    def test_sample(self):
        samples = self.distribution.sample(100)
        self.assertTrue(all([self.distribution.likelihood(sample) > 0 for sample in samples]))


class IntegerDistributionTestCase(unittest.TestCase):
    distribution: IntegerDistribution = IntegerDistribution(Integer("number", {1, 2, 4}),
                                                            [0.3, 0.3, 0.4])

    def test_pdf(self):
        self.assertEqual(self.distribution.pdf(1), 0.3)

    def test_cdf(self):
        self.assertEqual(self.distribution.cdf(1), 0.0)
        self.assertEqual(self.distribution.cdf(2), 0.3)
        self.assertEqual(self.distribution.cdf(4), 0.6)

    def test_moment(self):
        expectation = self.distribution.moment(VariableMap({self.distribution.variable: 1}),
                                               VariableMap({self.distribution.variable: 0}))
        self.assertEqual(expectation[self.distribution.variable], 2.5)
        variance = self.distribution.moment(VariableMap({self.distribution.variable: 2}), expectation)
        self.assertEqual(variance[self.distribution.variable], 1.65)


if __name__ == '__main__':
    unittest.main()
