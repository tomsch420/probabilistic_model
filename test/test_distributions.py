import unittest

from anytree import RenderTree

from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution, SymbolicDistribution, \
    IntegerDistribution, DiracDeltaDistribution
from probabilistic_model.probabilistic_circuit.units import DeterministicSumUnit, Unit
from random_events.events import Event, VariableMap
from random_events.variables import Continuous, Symbolic, Integer
import portion
import plotly.graph_objects as go


class UniformDistributionTestCase(unittest.TestCase):
    distribution: UniformDistribution = UniformDistribution(Continuous("x"), portion.closedopen(0, 2))

    def test_domain(self):
        self.assertEqual(self.distribution.domain, Event({self.distribution.variable: portion.closedopen(0, 2)}))

    def test_likelihood(self):
        self.assertEqual(self.distribution.likelihood([1]), 0.5)
        self.assertEqual(self.distribution.likelihood([0]), 0.5)
        self.assertEqual(self.distribution.likelihood([2]), 0.)
        self.assertEqual(self.distribution.likelihood([-1]), 0.)
        self.assertEqual(self.distribution.likelihood([3]), 0.)

    def test_probability_of_domain(self):
        self.assertEqual(self.distribution.probability(self.distribution.domain), 1)

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
        self.assertEqual(modes, [self.distribution.domain])
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
        self.assertEqual(conditional.children[0].interval, portion.closed(0, 1))
        self.assertEqual(conditional.children[1].interval, portion.closedopen(1.5, 2))

    def test_conditional_triple_complex_intersection(self):
        event = Event(
            {self.distribution.variable: portion.closed(1.5, 2) | portion.closed(0, 0.25) | portion.closed(0.75, 1)})

        conditional, probability = self.distribution.conditional(event)
        self.assertIsInstance(conditional, DeterministicSumUnit)
        self.assertEqual(probability, 0.5)
        self.assertEqual(len(conditional.children), 3)
        self.assertEqual(conditional.weights, [1 / 4, 1 / 4, 1 / 2])
        self.assertEqual(conditional.children[0].interval, portion.closed(0, 0.25))
        self.assertEqual(conditional.children[1].interval, portion.closed(0.75, 1))
        self.assertEqual(conditional.children[2].interval, portion.closedopen(1.5, 2))

    def test_conditional_mode(self):
        event = Event({
            self.distribution.variable: portion.closedopen(1.5, 2) | portion.closedopen(0, 0.25) | portion.closedopen(
                0.75, 1)})

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

    def test_inclusion_likelihood(self):
        distribution = UniformDistribution(Continuous("x"), portion.closed(0, 1))
        self.assertEqual(distribution.likelihood([1]), 1)
        self.assertEqual(distribution.likelihood([0]), 1)

    def test_exclusion_likelihood(self):
        distribution = UniformDistribution(Continuous("x"), portion.open(0, 1))
        self.assertEqual(distribution.likelihood([1]), 0)
        self.assertEqual(distribution.likelihood([0]), 0)

    def test_conditional_with_singleton(self):
        conditional, likelihood = self.distribution.conditional(Event({self.distribution.variable: 1}))
        self.assertIsInstance(conditional, DiracDeltaDistribution)
        self.assertEqual(likelihood, 0.5)

    def test_conditional_with_mixture_of_interval_and_singleton(self):
        event = Event({self.distribution.variable: portion.closed(1, 2) |
                                                   portion.closed(0, 0.25) |
                                                   portion.singleton(0.75)})
        conditional, likelihood = self.distribution.conditional(event)
        self.assertIsInstance(conditional, DeterministicSumUnit)
        self.assertEqual(likelihood, 1.125)
        self.assertEqual(len(conditional.children), 3)

    def test_serialization(self):
        serialization = self.distribution.to_json()
        self.assertEqual(serialization["type"],
                         "probabilistic_model.probabilistic_circuit.distributions.UniformDistribution")
        self.assertEqual(serialization["interval"], [(True, 0, 2, False)])
        deserialized = Unit.from_json(serialization)
        self.assertIsInstance(deserialized, UniformDistribution)


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

    def test_domain(self):
        domain = self.distribution.domain
        self.assertEqual(domain, Event({self.distribution.variable: self.distribution.variable.domain}))

    def test_domain_if_weights_are_zero(self):
        distribution = SymbolicDistribution(Symbolic("animal", {"cat", "dog", "chicken"}), [0, 0, 1])
        domain = distribution.domain
        self.assertEqual(domain, Event({distribution.variable: "dog"}))

    def test_fit(self):
        distribution = SymbolicDistribution(Symbolic("animal", {"cat", "dog", "chicken"}), [1/3] * 3)
        data = ["cat", "dog", "dog", "chicken", "chicken", "chicken"]
        distribution.fit(data)
        self.assertEqual(distribution.weights, [1/6, 3/6, 2/6])

    def test_plot(self):
        fig = go.Figure(data=self.distribution.plot())
        self.assertIsNotNone(fig)
        # fig.show()


class IntegerDistributionTestCase(unittest.TestCase):
    distribution: IntegerDistribution = IntegerDistribution(Integer("number", {1, 2, 4}), [0.3, 0.3, 0.4])

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

    def test_fit(self):
        distribution = IntegerDistribution(Integer("number", {1, 2, 4}), [1/3] * 3)
        data = [1, 2, 2, 4, 4, 4]
        distribution.fit(data)
        self.assertEqual(distribution.weights, [1/6, 2/6, 3/6])

    def test_plot(self):
        fig = go.Figure(self.distribution.plot())
        self.assertIsNotNone(fig)
        # fig.show()


class DiracDeltaTestCase(unittest.TestCase):
    variable = Continuous("x")
    distribution = DiracDeltaDistribution(variable, 0, 2)

    def test_pdf(self):
        self.assertEqual(self.distribution.pdf(1), 0)
        self.assertEqual(self.distribution.pdf(0), 2)
        self.assertEqual(self.distribution.pdf(2), 0)
        self.assertEqual(self.distribution.pdf(-1), 0)
        self.assertEqual(self.distribution.pdf(3), 0)

    def test_cdf(self):
        self.assertEqual(self.distribution.cdf(1), 1)
        self.assertEqual(self.distribution.cdf(0), 1)
        self.assertEqual(self.distribution.cdf(2), 1)
        self.assertEqual(self.distribution.cdf(-1), 0)
        self.assertEqual(self.distribution.cdf(3), 1)

    def test_probability(self):
        event = Event({self.distribution.variable: portion.closed(0, 1) | portion.closed(1.5, 2)})
        self.assertEqual(self.distribution.probability(event), 1)

    def test_conditional(self):
        event = Event({self.distribution.variable: portion.closed(-1, 2)})
        conditional, probability = self.distribution.conditional(event)
        self.assertEqual(conditional, self.distribution)
        self.assertEqual(probability, 1)

    def test_conditional_impossible(self):
        event = Event({self.distribution.variable: portion.closed(1, 2)})
        conditional, probability = self.distribution.conditional(event)
        self.assertIsNone(conditional)
        self.assertEqual(0, probability)

    def test_mode(self):
        mode, likelihood = self.distribution.mode()
        self.assertEqual(mode, [Event({self.distribution.variable: 0})])
        self.assertEqual(likelihood, 2)

    def test_sample(self):
        samples = self.distribution.sample(100)
        self.assertTrue(all([self.distribution.likelihood(sample) > 0 for sample in samples]))

    def test_expectation(self):
        self.assertEqual(self.distribution.expectation([self.variable])[self.variable], 0)

    def test_variance(self):
        self.assertEqual(self.distribution.variance([self.variable])[self.variable], 0)

    def test_higher_order_moment(self):
        center = self.distribution.expectation([self.variable])
        order = VariableMap({self.variable: 3})
        self.assertEqual(self.distribution.moment(order, center)[self.variable], 0)

    def test_equality_dirac_delta_and_other(self):
        self.assertNotEqual(self.distribution, UniformDistribution(self.variable, portion.closed(0, 2)))


if __name__ == '__main__':
    unittest.main()
