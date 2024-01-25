import math
import unittest

import plotly.graph_objects as go
import portion
from anytree import RenderTree
from random_events.events import Event, VariableMap
from random_events.variables import Continuous, Symbolic, Integer

from probabilistic_model.probabilistic_circuit.distribution import SymbolicDistribution, IntegerDistribution, \
    DiracDeltaDistribution, UnivariateContinuousSumUnit, UnivariateDiscreteSumUnit
from probabilistic_model.probabilistic_circuit.distributions.gaussian import GaussianDistribution, \
    TruncatedGaussianDistribution
from probabilistic_model.probabilistic_circuit.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.units import DeterministicSumUnit, Unit


class UniformDistributionTestCase(unittest.TestCase):
    distribution: UniformDistribution = UniformDistribution(Continuous("x"), portion.closedopen(0, 2))

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

    def test_conditional_with_mixture_of_interval_and_singleton(self):
        event = Event(
            {self.distribution.variable: portion.closed(1, 2) | portion.closed(0, 0.25) | portion.singleton(0.75)})
        conditional, likelihood = self.distribution.conditional(event)
        self.assertIsInstance(conditional, DeterministicSumUnit)
        self.assertEqual(likelihood, 1.125)
        self.assertEqual(len(conditional.children), 3)

    def test_serialization(self):
        serialization = self.distribution.to_json()
        self.assertEqual(serialization["type"],
                         "probabilistic_model.probabilistic_circuit.distributions.uniform.UniformDistribution")
        self.assertEqual(serialization["interval"], [(True, 0, 2, False)])
        deserialized = Unit.from_json(serialization)
        self.assertIsInstance(deserialized, UniformDistribution)


class UnivariateDiscreteSumUnitTestCase(unittest.TestCase):

    variable: Symbolic = Symbolic("animal", {"cat", "dog", "chicken"})
    distribution: UnivariateDiscreteSumUnit

    def setUp(self):
        distribution_1 = SymbolicDistribution(self.variable,
                                              [1 / 6, 3 / 6, 2 / 6])
        distribution_2 = SymbolicDistribution(self.variable,
                                              [3 / 6, 2 / 6, 1 / 6])
        self.distribution = UnivariateDiscreteSumUnit(self.variable, [0.3, 0.7])
        self.distribution.children = [distribution_1, distribution_2]

    def test_simplify(self):
        result = self.distribution.simplify()
        self.assertIsInstance(result, SymbolicDistribution)
        self.assertEqual(result.variable, self.variable)
        weights_by_hand = [(1. * 0.3 + 0.7 * 3.)/6.,
                           (3. * 0.3 + 0.7 * 2.)/6.,
                           (2. * 0.3 + 0.7 * 1)/6.]
        for w, w_ in zip(weights_by_hand, result.weights):
            self.assertAlmostEqual(w, w_)

    def test_probability(self):
        result = self.distribution.probability(Event({self.variable: ("cat", "dog")}))
        self.assertAlmostEqual(result, 1 - ((3. * 0.3 + 0.7 * 2.)/6.))


class UnivariateContinuousSumUnitTestCase(unittest.TestCase):
    model: UnivariateContinuousSumUnit

    def setUp(self):
        self.model = UnivariateContinuousSumUnit(Continuous("x"), [0.5, 0.5])
        self.model.children = [UniformDistribution(Continuous("x"), portion.closed(0, 1)),
                               UniformDistribution(Continuous("x"), portion.closed(2, 3))]

    def test_variable_getter(self):
        self.assertEqual(self.model.variable, Continuous("x"))

    def test_pdf(self):
        self.assertEqual(self.model.pdf(0.5), 0.5)
        self.assertEqual(self.model.pdf(2.5), 0.5)
        self.assertEqual(self.model.pdf(1.5), 0)
        self.assertEqual(self.model.pdf(3.5), 0)
        self.assertEqual(self.model.pdf(2), 0.5)

    def test_cdf(self):
        self.assertEqual(self.model.cdf(0.5), 0.25)
        self.assertEqual(self.model.cdf(2.5), 0.75)
        self.assertEqual(self.model.cdf(1.5), 0.5)
        self.assertEqual(self.model.cdf(3.5), 1)
        self.assertEqual(self.model.cdf(2), 0.5)

    def test_moment(self):
        expectation = self.model.expectation([self.model.variable])
        variance = self.model.variance([self.model.variable])
        self.assertEqual(expectation[self.model.variable], 1.5)
        self.assertEqual(variance[self.model.variable], 1.0833333333333333333)

    def test_plot(self):
        fig = go.Figure(self.model.plot())
        self.assertIsNotNone(fig)
        # fig.show()


class GaussianDistributionTestCase(unittest.TestCase):
    distribution: GaussianDistribution = GaussianDistribution(Continuous("x"), mean=2, variance=4)

    def test_conditional_complex_intersection(self):
        event = Event({self.distribution.variable: portion.closed(1.5, 2) | portion.closed(3, 4)})
        conditional, probability = self.distribution.conditional(event)
        self.assertIsInstance(conditional, DeterministicSumUnit)
        self.assertEqual(probability, self.distribution.probability(event))
        self.assertEqual(len(conditional.children), 2)

        weights_by_hand = [self.distribution.probability(Event({self.distribution.variable: portion.closed(1.5, 2)})),
                           self.distribution.probability(Event({self.distribution.variable: portion.closed(3, 4)}))]
        weights_by_hand = [weight / sum(weights_by_hand) for weight in weights_by_hand]

        self.assertEqual(conditional.weights, weights_by_hand)

        self.assertEqual(conditional.children[0].interval, portion.closed(1.5, 2))
        self.assertEqual(conditional.children[1].interval, portion.closed(3, 4))

    def test_conditional_triple_complex_intersection(self):
        event = Event(
            {self.distribution.variable: portion.closed(1.5, 2) | portion.closed(0, 0.25) | portion.closed(0.75, 1)})

        conditional, probability = self.distribution.conditional(event)
        self.assertIsInstance(conditional, DeterministicSumUnit)
        self.assertIsInstance(conditional.children[0], TruncatedGaussianDistribution)
        self.assertIsInstance(conditional.children[1], TruncatedGaussianDistribution)
        self.assertIsInstance(conditional.children[2], TruncatedGaussianDistribution)
        self.assertEqual(len(conditional.children), 3)

    # This unit test is not working, should work only for Truncated Gaussians
    def test_conditional_mode(self):
        event = Event(
            {self.distribution.variable: portion.closed(1.5, 2) | portion.closed(0, 0.25) | portion.closed(0.75, 1)})

        conditional, probability = self.distribution.conditional(event)
        modes, likelihood = conditional.mode()
        self.assertEqual(len(modes), 1)
        self.assertEqual(likelihood, conditional.mode()[1])
        self.assertEqual(modes[0][self.distribution.variable], portion.singleton(2))

    def test_pdf_striped(self):
        event = Event({self.distribution.variable: portion.closed(1.5, 2) | portion.closed(3, 4)})
        sub_event_1 = Event({self.distribution.variable: portion.closed(1.5, 2)})
        sub_event_2 = Event({self.distribution.variable: portion.closed(3, 4)})
        conditional, probability = self.distribution.conditional(event)
        sub_cond_1, prob_1 = self.distribution.conditional(sub_event_1)
        sub_cond_2, prob_2 = self.distribution.conditional(sub_event_2)
        self.assertEqual(probability, prob_1 + prob_2)
        self.assertEqual(self.distribution.pdf(1.7) / probability, conditional.weights[0] * sub_cond_1.pdf(1.7))

    def test_moments_conditoned_on_mixture(self):
        event = Event({self.distribution.variable: portion.closed(1.5, 2) | portion.closed(3, 4)})
        sub_event_1 = Event({self.distribution.variable: portion.closed(1.5, 2)})
        sub_event_2 = Event({self.distribution.variable: portion.closed(3, 4)})
        conditional, probability = self.distribution.conditional(event)
        expectation = conditional.expectation(conditional.variables)
        print(RenderTree(conditional))
        variance = conditional.variance(conditional.variables)
        sub_cond_1, prob_1 = self.distribution.conditional(sub_event_1)
        sub_cond_2, prob_2 = self.distribution.conditional(sub_event_2)
        sub_expectation_1 = sub_cond_1.expectation(sub_cond_1.variables)
        sub_expectation_2 = sub_cond_2.expectation(sub_cond_2.variables)

        self.assertAlmostEqual(expectation[conditional.variables[0]],
                               conditional.weights[0] * sub_expectation_1[sub_cond_1.variables[0]] +
                               conditional.weights[1] * sub_expectation_2[sub_cond_2.variables[0]], places=7)

        sub_variance_1 = sub_cond_1.moment(VariableMap({sub_cond_1.variable: 2}),
                                           sub_cond_1.expectation(sub_cond_1.variables))
        sub_variance_2 = sub_cond_2.moment(VariableMap({sub_cond_2.variable: 2}),
                                           sub_cond_2.expectation(sub_cond_2.variables))

        print("Mixture Expectation: ", expectation[conditional.variables[0]])

        self.assertAlmostEqual(variance[conditional.variables[0]],
                               conditional.weights[0] * sub_variance_1[sub_cond_1.variables[0]] + conditional.weights[
                                   1] * sub_variance_2[sub_cond_2.variables[0]] + conditional.weights[0] *
                               sub_expectation_1[sub_cond_1.variables[0]] ** 2 + conditional.weights[1] *
                               sub_expectation_2[sub_cond_2.variables[0]] ** 2 - (
                                       conditional.weights[0] * sub_expectation_1[sub_cond_1.variables[0]] + \
                                       conditional.weights[1] * sub_expectation_2[sub_cond_2.variables[0]]) ** 2,
                               places=7)

        print("Mixture Variance: ", variance[conditional.variables[0]])

    def test_serialization(self):
        serialization = self.distribution.to_json()
        self.assertEqual(serialization["type"],
                         "probabilistic_model.probabilistic_circuit.distributions.gaussian.GaussianDistribution")
        self.assertEqual(serialization["mean"], 2)
        self.assertEqual(serialization["variance"], 4)
        deserialized = Unit.from_json(serialization)
        self.assertIsInstance(deserialized, GaussianDistribution)


if __name__ == '__main__':
    unittest.main()
