import unittest

import plotly.graph_objects as go
import portion
from random_events.events import Event, VariableMap
from random_events.variables import Continuous, Integer, Discrete

from probabilistic_model.distributions.distributions import (UnivariateDistribution, ContinuousDistribution,
                                                             DiscreteDistribution, IntegerDistribution,
                                                             DiracDeltaDistribution)
from probabilistic_model.utils import SubclassJSONSerializer


class UnivariateDistributionTestCase(unittest.TestCase):
    variable = Continuous("x")
    model: UnivariateDistribution

    def setUp(self):
        self.model = UnivariateDistribution(self.variable)

    def test_variable(self):
        self.assertEqual(self.model.variable, self.variable)


class ContinuousDistributionTestCase(unittest.TestCase):
    variable = Continuous("x")
    model: ContinuousDistribution

    def setUp(self):
        self.model = ContinuousDistribution(self.variable)

    def test_cdf(self):
        self.assertEqual(self.model.cdf(float("inf")), 1)
        self.assertEqual(self.model.cdf(-float("inf")), 0)


class DiscreteTestCase(unittest.TestCase):
    variable = Discrete("x", (1, 2, 3))
    model: DiscreteDistribution

    def setUp(self):
        self.model = DiscreteDistribution(self.variable, [4 / 20, 5 / 20, 11 / 20])

    def test_creating_with_invalid_weights(self):
        with self.assertRaises(ValueError):
            DiscreteDistribution(self.variable, [0, 1])

    def test_pdf(self):
        self.assertEqual(self.model.pdf(1), 1 / 5)

    def test_probability(self):
        event = Event({self.variable: (1, 3)})
        self.assertEqual(self.model.probability(event), 15 / 20)

    def test_mode(self):
        mode, likelihood = self.model.mode()
        self.assertEqual(likelihood, 11 / 20)
        self.assertEqual(mode, [Event({self.variable: 3})])

    def test_conditional(self):
        event = Event({self.variable: (1, 2)})
        conditional, likelihood = self.model.conditional(event)
        self.assertEqual(likelihood, 9 / 20)
        self.assertAlmostEqual(conditional.weights[0], 4 / 9)
        self.assertAlmostEqual(conditional.weights[1], 5 / 9)
        self.assertAlmostEqual(conditional.weights[2], 0.)

    def test_conditional_impossible(self):
        event = Event({self.variable: []})
        conditional, probability = self.model.conditional(event)
        self.assertIsNone(conditional)
        self.assertEqual(probability, 0)

    def test_sample(self):
        samples = self.model.sample(10)
        for sample in samples:
            self.assertGreater(self.model.likelihood(sample), 0)

    def test_copy(self):
        model = self.model.__copy__()
        self.assertEqual(model.weights, self.model.weights)
        model.weights = [1 / 3, 1 / 3, 1 / 3]
        self.assertNotEqual(model.weights, self.model.weights)

    def test_fit(self):
        data = [1, 2, 2, 2]
        self.model.fit(data)
        self.assertEqual(self.model.weights, [1 / 4, 3 / 4, 0])

    def test_domain(self):
        domain = self.model.domain
        self.assertEqual(domain, Event({self.variable: self.variable.domain}))

    def test_domain_if_weights_are_zero(self):
        distribution = DiscreteDistribution(self.variable, [0, 0, 1])
        domain = distribution.domain
        self.assertEqual(domain, Event({distribution.variable: 3}))

    def test_plot(self):
        fig = go.Figure(self.model.plot())  # fig.show()

    def test_serialization(self):
        serialized = self.model.to_json()
        deserialized = SubclassJSONSerializer.from_json(serialized)
        self.assertIsInstance(deserialized, DiscreteDistribution)
        self.assertEqual(deserialized, self.model)


class IntegerDistributionTestCase(unittest.TestCase):
    variable = Integer("x", (1, 2, 4))
    model: IntegerDistribution

    def setUp(self):
        self.model = IntegerDistribution(self.variable, [1 / 4, 1 / 4, 1 / 2])

    def test_cdf(self):
        self.assertEqual(self.model.cdf(4), 0.5)

    def test_moment(self):
        expectation = self.model.expectation(self.model.variables)
        self.assertEqual(expectation[self.variable], 2.75)

    def test_plot(self):
        fig = go.Figure(self.model.plot())
        # fig.show()

    def test_serialization(self):
        serialized = self.model.to_json()
        deserialized = SubclassJSONSerializer.from_json(serialized)
        self.assertIsInstance(deserialized, IntegerDistribution)
        self.assertEqual(deserialized, self.model)


class DiracDeltaDistributionTestCase(unittest.TestCase):
    variable = Continuous("x")
    model: DiracDeltaDistribution

    def setUp(self):
        self.model = DiracDeltaDistribution(self.variable, 0, 2)

    def test_pdf(self):
        self.assertEqual(self.model.pdf(1), 0)
        self.assertEqual(self.model.pdf(0), 2)
        self.assertEqual(self.model.pdf(2), 0)
        self.assertEqual(self.model.pdf(-1), 0)
        self.assertEqual(self.model.pdf(3), 0)

    def test_cdf(self):
        self.assertEqual(self.model.cdf(1), 1)
        self.assertEqual(self.model.cdf(0), 1)
        self.assertEqual(self.model.cdf(2), 1)
        self.assertEqual(self.model.cdf(-1), 0)
        self.assertEqual(self.model.cdf(3), 1)

    def test_probability(self):
        event = Event({self.model.variable: portion.closed(0, 1) | portion.closed(1.5, 2)})
        self.assertEqual(self.model.probability(event), 1)

    def test_probability_0(self):
        event = Event({self.variable: portion.openclosed(0, 1)})
        self.assertEqual(self.model.probability(event), 0.)

    def test_conditional(self):
        event = Event({self.model.variable: portion.closed(-1, 2)})
        conditional, probability = self.model.conditional(event)
        self.assertEqual(conditional, self.model)
        self.assertEqual(probability, 1)

    def test_conditional_impossible(self):
        event = Event({self.model.variable: portion.closed(1, 2)})
        conditional, probability = self.model.conditional(event)
        self.assertIsNone(conditional)
        self.assertEqual(0, probability)

    def test_mode(self):
        mode, likelihood = self.model.mode()
        self.assertEqual(mode, [Event({self.model.variable: 0})])
        self.assertEqual(likelihood, 2)

    def test_sample(self):
        samples = self.model.sample(100)
        self.assertTrue(all([self.model.likelihood(sample) > 0 for sample in samples]))

    def test_expectation(self):
        self.assertEqual(self.model.expectation([self.variable])[self.variable], 0)

    def test_variance(self):
        self.assertEqual(self.model.variance([self.variable])[self.variable], 0)

    def test_higher_order_moment(self):
        center = self.model.expectation([self.variable])
        order = VariableMap({self.variable: 3})
        self.assertEqual(self.model.moment(order, center)[self.variable], 0)

    def test_serialization(self):
        serialized = self.model.to_json()
        deserialized = SubclassJSONSerializer.from_json(serialized)
        self.assertIsInstance(deserialized, DiracDeltaDistribution)
        self.assertEqual(deserialized, self.model)

if __name__ == '__main__':
    unittest.main()
