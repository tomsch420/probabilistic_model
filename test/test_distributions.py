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

    def test_plot(self):
        fig = go.Figure(data=self.distribution.plot())
        self.assertIsNotNone(fig)  # fig.show()


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
        distribution = SymbolicDistribution(Symbolic("animal", {"cat", "dog", "chicken"}), [1 / 3] * 3)
        data = ["cat", "dog", "dog", "chicken", "chicken", "chicken"]
        distribution.fit(data)
        self.assertEqual(distribution.weights, [1 / 6, 3 / 6, 2 / 6])

    def test_plot(self):
        fig = go.Figure(data=self.distribution.plot())
        self.assertIsNotNone(fig)
        # fig.show()


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
        distribution = IntegerDistribution(Integer("number", {1, 2, 4}), [1 / 3] * 3)
        data = [1, 2, 2, 4, 4, 4]
        distribution.fit(data)
        self.assertEqual(distribution.weights, [1 / 6, 2 / 6, 3 / 6])

    def test_plot(self):
        fig = go.Figure(self.distribution.plot())
        self.assertIsNotNone(fig)  # fig.show()


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

    def test_domain(self):
        self.assertEqual(self.distribution.domain,
                         Event({self.distribution.variable: portion.closedopen(-portion.inf, portion.inf)}))

    def test_likelihood(self):
        self.assertEqual(self.distribution.likelihood([1]), self.distribution.pdf(1))
        self.assertEqual(self.distribution.likelihood([0]), self.distribution.pdf(0))
        self.assertEqual(self.distribution.likelihood([200]), self.distribution.pdf(200))
        self.assertEqual(self.distribution.likelihood([-portion.inf]), 0)
        self.assertEqual(self.distribution.likelihood([portion.inf]), self.distribution.pdf(portion.inf))

    def test_probability_of_domain(self):
        self.assertEqual(self.distribution.probability(self.distribution.domain), 1)

    def test_cdf(self):
        self.assertEqual(self.distribution.cdf(2), 0.5)
        self.assertEqual(self.distribution.cdf(1), self.distribution.cdf(1))
        self.assertEqual(self.distribution.cdf(-portion.inf), 0)
        self.assertEqual(self.distribution.cdf(portion.inf), 1)
        self.assertEqual(self.distribution.cdf(portion.inf), self.distribution.cdf(portion.inf))

    def test_probability_of_slices(self):
        event = Event({self.distribution.variable: portion.closed(0, 1)})
        self.assertEqual(self.distribution.probability(event), self.distribution.cdf(1) - self.distribution.cdf(0))

    def test_mode(self):
        modes, likelihood = self.distribution.mode()
        mode = modes[0]
        self.assertEqual(mode[self.distribution.variable].lower, self.distribution.mean)

    def test_sample(self):
        samples = self.distribution.sample(100)
        self.assertEqual(len(samples), 100)
        for sample in samples:
            self.assertGreaterEqual(self.distribution.likelihood(sample), 0)

    def test_conditional_simple_intersection(self):
        event = Event({self.distribution.variable: portion.closed(1, 2)})
        conditional, probability = self.distribution.conditional(event)
        self.assertIsInstance(conditional, TruncatedGaussianDistribution)
        self.assertEqual(probability, self.distribution.cdf(2) - self.distribution.cdf(1))
        self.assertEqual(conditional.lower, 1)
        self.assertEqual(conditional.upper, 2)

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

    def test_raw_moment(self):
        self.assertEqual(self.distribution.raw_moment(0), 1)
        self.assertEqual(self.distribution.raw_moment(1), self.distribution.mean)
        self.assertEqual(self.distribution.raw_moment(2), self.distribution.mean ** 2 + self.distribution.variance)

    def test_centered_moment(self):
        expectation = self.distribution.moment(VariableMap({self.distribution.variable: 1}),
                                               VariableMap({self.distribution.variable: 0}))
        self.assertEqual(expectation[self.distribution.variable], 2)
        variance = self.distribution.moment(VariableMap({self.distribution.variable: 2}), expectation)
        self.assertEqual(variance[self.distribution.variable], 4)

        third_order_moment = self.distribution.moment(VariableMap({self.distribution.variable: 3}), expectation)
        self.assertEqual(third_order_moment[self.distribution.variable], 0)

        fourth_order_moment = self.distribution.moment(VariableMap({self.distribution.variable: 4}), expectation)
        self.assertEqual(fourth_order_moment[self.distribution.variable], 48)

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

    def test_moment_with_different_center_than_expectation(self):
        center = VariableMap({self.distribution.variable: 2.5})

        expectation = self.distribution.moment(VariableMap({self.distribution.variable: 1}), center)
        self.assertEqual(expectation[self.distribution.variable], -0.5)
        variance = self.distribution.moment(VariableMap({self.distribution.variable: 2}), center)
        self.assertEqual(variance[self.distribution.variable], 4.25)

        third_order_moment = self.distribution.moment(VariableMap({self.distribution.variable: 3}), center)
        self.assertEqual(third_order_moment[self.distribution.variable], -6.125)

    def test_serialization(self):
        serialization = self.distribution.to_json()
        self.assertEqual(serialization["type"],
                         "probabilistic_model.probabilistic_circuit.distributions.gaussian.GaussianDistribution")
        self.assertEqual(serialization["mean"], 2)
        self.assertEqual(serialization["variance"], 4)
        deserialized = Unit.from_json(serialization)
        self.assertIsInstance(deserialized, GaussianDistribution)

    def test_plot(self):
        fig = go.Figure(data=self.distribution.plot())
        self.assertIsNotNone(fig)  # fig.show()


class TruncatedGaussianDistributionTestCase(unittest.TestCase):
    distribution: TruncatedGaussianDistribution

    def setUp(self):
        self.distribution = TruncatedGaussianDistribution(Continuous("real"), portion.closed(-2, 2), 2, 4.)

    def test_init(self):
        print(self.distribution)
        self.assertEqual(self.distribution.mean, 2.)

    def test_cdf(self):
        self.assertAlmostEqual(self.distribution.cdf(0), 0.285, places=3)
        self.assertEqual(self.distribution.cdf(3), 1)
        self.assertEqual(self.distribution.cdf(-3), 0)

    def test_raw_moment(self):
        gauss_distribution: GaussianDistribution = GaussianDistribution(Continuous("x"), mean=0, variance=1)
        beta = (self.distribution.upper - self.distribution.mean) / math.sqrt(self.distribution.variance)
        alpha = (self.distribution.lower - self.distribution.mean) / math.sqrt(self.distribution.variance)
        expectation = self.distribution.moment(VariableMap({self.distribution.variable: 1}),
                                               VariableMap({self.distribution.variable: 0}))
        self.assertAlmostEqual(expectation[self.distribution.variable], 0.5544205, places=7)
        second_moment = self.distribution.moment(VariableMap({self.distribution.variable: 2}),
                                                 VariableMap({self.distribution.variable: 0}))
        variance = self.distribution.variance * ((1 - (
                beta * gauss_distribution.pdf(beta) - alpha * gauss_distribution.pdf(
            alpha)) / self.distribution.normalizing_constant) - ((gauss_distribution.pdf(beta) - gauss_distribution.pdf(
            alpha)) / self.distribution.normalizing_constant) ** 2)
        self.assertAlmostEqual(second_moment[self.distribution.variable],
                               variance + expectation[self.distribution.variable] ** 2, places=7)

    def test_centered_moment(self):
        gauss_distribution: GaussianDistribution = GaussianDistribution(Continuous("x"), mean=0, variance=1)
        beta = (self.distribution.upper - self.distribution.mean) / math.sqrt(self.distribution.variance)
        alpha = (self.distribution.lower - self.distribution.mean) / math.sqrt(self.distribution.variance)
        center = VariableMap({self.distribution.variable: self.distribution.mean})
        expectation = self.distribution.moment(VariableMap({self.distribution.variable: 1}), center)
        offset_term = -math.sqrt(self.distribution.variance) * (gauss_distribution.pdf(beta) - gauss_distribution.pdf(
            alpha)) / self.distribution.normalizing_constant
        self.assertAlmostEqual(expectation[self.distribution.variable], 0 + offset_term, places=7)
        second_moment = self.distribution.moment(VariableMap({self.distribution.variable: 2}), center)
        variance = self.distribution.variance * ((1 - (
                beta * gauss_distribution.pdf(beta) - alpha * gauss_distribution.pdf(
            alpha)) / self.distribution.normalizing_constant) - ((gauss_distribution.pdf(beta) - gauss_distribution.pdf(
            alpha)) / self.distribution.normalizing_constant) ** 2)
        self.assertAlmostEqual(second_moment[self.distribution.variable], variance + offset_term ** 2, places=7)

    def test_moment_with_different_center_than_expectation(self):
        center = VariableMap({self.distribution.variable: 2.5})
        expectation = self.distribution.moment(VariableMap({self.distribution.variable: 1}),
                                               VariableMap({self.distribution.variable: 0}))
        first_moment = self.distribution.moment(VariableMap({self.distribution.variable: 1}), center)
        self.assertAlmostEqual(first_moment[self.distribution.variable],
                               expectation[self.distribution.variable] - center[self.distribution.variable], places=7)
        second_moment = self.distribution.moment(VariableMap({self.distribution.variable: 2}), center)
        second_raw_moment = self.distribution.moment(VariableMap({self.distribution.variable: 2}),
                                                     VariableMap({self.distribution.variable: 0}))
        self.assertAlmostEqual(second_moment[self.distribution.variable],
                               second_raw_moment[self.distribution.variable] + center[
                                   self.distribution.variable] ** 2 - 2 * center[self.distribution.variable] *
                               expectation[self.distribution.variable], places=7)

    def test_conditional_simple_intersection(self):
        event = Event({self.distribution.variable: portion.closed(1, 2)})
        conditional, probability = self.distribution.conditional(event)
        self.assertIsInstance(conditional, TruncatedGaussianDistribution)
        self.assertEqual(probability, self.distribution.cdf(2) - self.distribution.cdf(1))
        self.assertEqual(conditional.lower, 1)
        self.assertEqual(conditional.upper, 2)

    def test_sample(self):
        samples = self.distribution.sample(100)
        self.assertEqual(len(samples), 100)
        for sample in samples:
            sample = sample[0]
            self.assertTrue(sample in self.distribution.domain[self.distribution.variable])
            self.assertGreater(self.distribution.pdf(sample), 0)

    def test_plot(self):
        fig = go.Figure(data=self.distribution.plot())
        self.assertIsNotNone(fig)  # fig.show()


class TruncatedGaussianDistributionJapaneseManTestCase(unittest.TestCase):
    distribution = GaussianDistribution(Continuous("x"), mean=0, variance=1)
    example_2: TruncatedGaussianDistribution
    example_3: TruncatedGaussianDistribution

    @classmethod
    def setUpClass(cls):
        cls.example_2, _ = cls.distribution.conditional(
            Event({cls.distribution.variable: portion.closed(0.5, float("inf"))}))

        cls.example_3, _ = cls.distribution.conditional(Event({cls.distribution.variable: portion.closed(-1, 1)}))

    def test_raw_expectation_example_2(self):
        center = VariableMap({self.distribution.variable: 0})
        order = VariableMap({self.distribution.variable: 1})
        result = self.example_2.moment(order, center)
        expectation = result[self.distribution.variable]
        self.assertAlmostEqual(expectation, 1.14, delta=0.01)

    def test_raw_expectation_example_3(self):
        center = VariableMap({self.distribution.variable: 0})
        order = VariableMap({self.distribution.variable: 1})
        result = self.example_3.moment(order, center)
        expectation = result[self.distribution.variable]
        self.assertAlmostEqual(expectation, 0, delta=0.01)

    def test_raw_second_moment_example_2(self):
        center = VariableMap({self.distribution.variable: 0})
        order = VariableMap({self.distribution.variable: 2})
        result = self.example_2.moment(order, center)
        raw_moment = result[self.distribution.variable]
        self.assertAlmostEqual(raw_moment, 1.57, delta=0.01)

    def test_raw_second_moment_example_3(self):
        center = VariableMap({self.distribution.variable: 0})
        order = VariableMap({self.distribution.variable: 2})
        result = self.example_3.moment(order, center)
        raw_moment = result[self.distribution.variable]
        self.assertAlmostEqual(raw_moment, 0.291, delta=0.01)

    def test_raw_third_moment_example_2(self):
        center = VariableMap({self.distribution.variable: 0})
        order = VariableMap({self.distribution.variable: 3})
        result = self.example_2.moment(order, center)
        raw_moment = result[self.distribution.variable]
        self.assertAlmostEqual(raw_moment, 2.57, delta=0.01)

    def test_raw_third_moment_example_3(self):
        center = VariableMap({self.distribution.variable: 0})
        order = VariableMap({self.distribution.variable: 3})
        result = self.example_3.moment(order, center)
        raw_moment = result[self.distribution.variable]
        self.assertAlmostEqual(raw_moment, 0, delta=0.01)


if __name__ == '__main__':
    unittest.main()
