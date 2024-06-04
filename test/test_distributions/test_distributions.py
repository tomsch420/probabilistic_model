import unittest

import plotly.graph_objects as go
import portion

from probabilistic_model.distributions.distributions import *
from probabilistic_model.utils import SubclassJSONSerializer


class IntegerDistributionTestCase(unittest.TestCase):
    x = Integer("x")
    model: IntegerDistribution

    def setUp(self):
        probabilities = defaultdict(float)
        probabilities[1] = 4/20
        probabilities[2] = 5/20
        probabilities[4] = 11/20
        self.model = IntegerDistribution(self.x, probabilities)

    def test_pdf(self):
        self.assertEqual(self.model.pdf(1), 1 / 5)
        self.assertEqual(self.model.pdf(3), 0)

    def test_probability(self):
        event = SimpleEvent({self.x: closed(1, 3)}).as_composite_set()
        self.assertEqual(self.model.probability(event), 9 / 20)

    def test_mode(self):
        mode, likelihood = self.model.mode()
        self.assertEqual(likelihood, 11 / 20)
        self.assertEqual(mode, SimpleEvent({self.x: singleton(4)}).as_composite_set())

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
        samples = self.model.sample(100)
        likelihoods = self.model.likelihoods(samples)
        self.assertTrue(all(likelihoods > 0))

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
        self.assertEqual(domain, ComplexEvent([Event({self.variable: self.variable.domain})]))

    def test_domain_if_weights_are_zero(self):
        distribution = DiscreteDistribution(self.variable, [0, 0, 1])
        domain = distribution.domain
        self.assertEqual(domain.events[0], Event({distribution.variable: 3}))

    def test_plot(self):
        fig = go.Figure(self.model.plot())  # fig.show()

    def test_serialization(self):
        serialized = self.model.to_json()
        deserialized = SubclassJSONSerializer.from_json(serialized)
        self.assertIsInstance(deserialized, DiscreteDistribution)
        self.assertEqual(deserialized, self.model)
#
#
# class IntegerDistributionTestCase(unittest.TestCase):
#     variable = Integer("x", (1, 2, 4))
#     model: IntegerDistribution
#
#     def setUp(self):
#         self.model = IntegerDistribution(self.variable, [1 / 4, 1 / 4, 1 / 2])
#
#     def test_cdf(self):
#         self.assertEqual(self.model.cdf(4), 0.5)
#
#     def test_moment(self):
#         expectation = self.model.expectation(self.model.variables)
#         self.assertEqual(expectation[self.variable], 2.75)
#
#     def test_plot(self):
#         fig = go.Figure(self.model.plot())
#         # fig.show()
#
#     def test_serialization(self):
#         serialized = self.model.to_json()
#         deserialized = SubclassJSONSerializer.from_json(serialized)
#         self.assertIsInstance(deserialized, IntegerDistribution)
#         self.assertEqual(deserialized, self.model)
#

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
        self.assertEqual(mode, ComplexEvent([Event({self.model.variable: 0})]))
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
