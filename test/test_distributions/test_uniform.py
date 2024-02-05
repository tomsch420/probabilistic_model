import unittest

import plotly.graph_objects as go
import portion
from random_events.events import Event, VariableMap
from random_events.variables import Continuous

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.utils import SubclassJSONSerializer


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

    def test_conditional_singleton_intersection(self):
        event = Event({self.distribution.variable: portion.singleton(1)})
        conditional, probability = self.distribution.conditional(event)
        self.assertEqual(conditional, DiracDeltaDistribution(self.distribution.variable, 1, 0.5))
        self.assertEqual(probability, 0.5)

    def test_conditional_simple_intersection(self):
        event = Event({self.distribution.variable: portion.closed(1, 2)})
        conditional, probability = self.distribution.conditional(event)
        self.assertIsInstance(conditional, UniformDistribution)
        self.assertEqual(probability, 0.5)
        self.assertEqual(conditional.lower, 1)
        self.assertEqual(conditional.upper, 2)

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

    def test_plot(self):
        fig = go.Figure(data=self.distribution.plot())
        self.assertIsNotNone(fig)
        # fig.show()

    def test_serialization(self):
        serialized = self.distribution.to_json()
        deserialized = SubclassJSONSerializer.from_json(serialized)
        self.assertEqual(self.distribution, deserialized)
        self.assertIsInstance(deserialized, UniformDistribution)

    def test_variable_setting(self):
        distribution = UniformDistribution(Continuous("x"), portion.closed(0, 1))
        self.assertEqual(distribution.variable, Continuous("x"))
        distribution.variables = [Continuous("y")]
        self.assertEqual(distribution.variable, Continuous("y"))
