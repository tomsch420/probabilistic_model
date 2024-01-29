import math
import unittest

import plotly.graph_objects as go
import portion
from random_events.events import Event, VariableMap
from random_events.variables import Continuous

from probabilistic_model.distributions.gaussian import GaussianDistribution, TruncatedGaussianDistribution
from probabilistic_model.utils import SubclassJSONSerializer


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

    def test_moment_with_different_center_than_expectation(self):
        center = VariableMap({self.distribution.variable: 2.5})

        expectation = self.distribution.moment(VariableMap({self.distribution.variable: 1}), center)
        self.assertEqual(expectation[self.distribution.variable], -0.5)
        variance = self.distribution.moment(VariableMap({self.distribution.variable: 2}), center)
        self.assertEqual(variance[self.distribution.variable], 4.25)

        third_order_moment = self.distribution.moment(VariableMap({self.distribution.variable: 3}), center)
        self.assertEqual(third_order_moment[self.distribution.variable], -6.125)

    def test_plot(self):
        fig = go.Figure(data=self.distribution.plot())
        self.assertIsNotNone(fig)  # fig.show()

    def test_serialization(self):
        serialized = self.distribution.to_json()
        deserialized = SubclassJSONSerializer.from_json(serialized)
        self.assertEqual(self.distribution, deserialized)
        self.assertIsInstance(deserialized, GaussianDistribution)


class TruncatedGaussianDistributionTestCase(unittest.TestCase):
    distribution: TruncatedGaussianDistribution

    def setUp(self):
        self.distribution = TruncatedGaussianDistribution(Continuous("real"), portion.closed(-2, 2), 2, 4.)

    def test_init(self):
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

    def test_serialization(self):
        serialized = self.distribution.to_json()
        deserialized = SubclassJSONSerializer.from_json(serialized)
        self.assertEqual(self.distribution, deserialized)
        self.assertIsInstance(deserialized, GaussianDistribution)



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
