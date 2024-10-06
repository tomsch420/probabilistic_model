import unittest

from matplotlib import pyplot as plt
from random_events.interval import closed
from random_events.variable import Continuous
from typing_extensions import Union

from probabilistic_model.probabilistic_circuit.nx.distributions import (UniformDistribution)
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import *


class ShowMixin:
    model: Union[ProbabilisticCircuit, ProbabilisticCircuitMixin]

    def show(self, model: Optional[Union[ProbabilisticCircuit, ProbabilisticCircuitMixin]] = None):

        if model is None:
            model = self.model

        if isinstance(model, ProbabilisticCircuitMixin):
            model = model.probabilistic_circuit

        pos = nx.planar_layout(model)
        nx.draw(model, pos=pos, with_labels=True)
        plt.show()


class NormalizationTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")

    def test_normalization(self):
        u1 = UniformDistribution(self.x, closed(0, 1).simple_sets[0])
        u2 = UniformDistribution(self.x, closed(3, 4).simple_sets[0])
        sum_unit = SumUnit()
        sum_unit.add_subcircuit(u1, 0.5)
        sum_unit.add_subcircuit(u2, 0.3)
        sum_unit.normalize()
        self.assertAlmostEqual(sum_unit.weights[0], 0.5 / 0.8)
        self.assertAlmostEqual(sum_unit.weights[1], 0.3 / 0.8)

    def test_plot(self):
        u1 = UniformDistribution(self.x, closed(0, 1).simple_sets[0])
        u2 = UniformDistribution(self.x, closed(3, 4).simple_sets[0])
        sum_unit = SumUnit()
        sum_unit.add_subcircuit(u1, 0.5)
        sum_unit.add_subcircuit(u2, 0.3)
        sum_unit.normalize()
        traces = sum_unit.plot()
        self.assertGreater(len(traces), 0)  # go.Figure(traces, sum_unit.plotly_layout()).show()


class SumUnitTestCase(unittest.TestCase, ShowMixin):
    x: Continuous = Continuous("x")
    model: SumUnit

    def setUp(self):
        u1 = UniformDistribution(self.x, closed(0, 1).simple_sets[0])
        u2 = UniformDistribution(self.x, closed(3, 4).simple_sets[0])

        self.model = SumUnit()
        self.model.add_subcircuit(u1, 0.6)
        self.model.add_subcircuit(u2, 0.4)

    def test_setup(self):
        self.assertEqual(len(self.model.probabilistic_circuit.nodes()), 3)
        self.assertEqual(len(self.model.probabilistic_circuit.edges()), 2)

    def test_variables(self):
        self.assertEqual(self.model.variables, SortedSet([self.x]))

    def test_latent_variable(self):
        self.assertEqual(len(self.model.latent_variable.domain.simple_sets), 2)

    def test_domain(self):
        domain = self.model.support
        domain_by_hand = SimpleEvent({self.x: closed(0, 1) | closed(3, 4)}).as_composite_set()
        self.assertEqual(domain, domain_by_hand)

    def test_weighted_subcircuits(self):
        weighted_subcircuits = self.model.weighted_subcircuits
        self.assertEqual(len(weighted_subcircuits), 2)
        self.assertEqual([weighted_subcircuit[0] for weighted_subcircuit in weighted_subcircuits], [0.6, 0.4])

    def test_likelihood(self):
        event = np.array([[0.5]])
        result = self.model.likelihood(event)
        self.assertEqual(result, 0.6)

    def test_probability(self):
        event = SimpleEvent({self.x: closed(0, 3.5)}).as_composite_set()
        result = self.model.probability(event)
        self.assertEqual(result, 0.8)

    def test_conditional(self):
        event = SimpleEvent({self.x: closed(0, 0.5)}).as_composite_set()
        result, probability = self.model.conditional(event)
        self.assertAlmostEqual(probability, 0.3)
        self.assertEqual(len(result.probabilistic_circuit.nodes()), 2)
        self.assertIsInstance(result, SumUnit)
        self.assertIsInstance(result.probabilistic_circuit.root, SumUnit)
        self.assertEqual(len(result.weighted_subcircuits), 1)
        self.assertEqual(result.weighted_subcircuits[0][0], 1)

    def test_conditional_impossible(self):
        event = SimpleEvent({self.x: closed(5, 6)}).as_composite_set()
        result, probability = self.model.conditional(event)
        self.assertEqual(probability, 0.)
        self.assertIsNone(result)

    def test_sample(self):
        samples = self.model.sample(100)
        likelihoods = self.model.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))

    def test_moment(self):
        expectation = self.model.expectation(self.model.variables)
        self.assertEqual(expectation[self.x], 0.5 * 0.6 + 0.4 * 3.5)

    def test_marginal(self):
        marginal = self.model.marginal([self.x])
        self.assertEqual(self.model, marginal)

    def test_mode(self):
        mode, likelihood = self.model.mode()
        self.assertEqual(likelihood, 0.6)
        self.assertEqual(mode, SimpleEvent({self.x: closed(0, 1)}).as_composite_set())

    def test_serialization(self):
        serialized = self.model.to_json()
        deserialized = SumUnit.from_json(serialized)
        self.assertEqual(self.model, deserialized)

    def test_copy(self):
        copy = self.model.__copy__()
        self.assertEqual(self.model, copy)
        self.assertNotEqual(id(copy), id(self.model))

    def test_conditional_inference(self):
        event = SimpleEvent({self.x: closed(0, 0.5)}).as_composite_set()
        result, probability = self.model.conditional(event)
        self.assertEqual(result.probability(event), 1)

    def test_deep_mount(self):
        s1 = SumUnit()
        s2 = SumUnit()
        s3 = SumUnit()
        u1 = UniformDistribution(self.x, closed(0, 1).simple_sets[0])
        s2.probabilistic_circuit.add_nodes_from([s2, s3, u1])
        s2.probabilistic_circuit.add_weighted_edges_from([(s2, s3, 1.), (s3, u1, 1.)])
        s1.mount(s2)
        self.assertEqual(len(s1.probabilistic_circuit.nodes()), 4)

    def test_sample_not_equal(self):
        samples = self.model.sample(10)
        uniques = np.unique(samples, axis=0)
        self.assertEqual(len(samples), len(uniques))

    def test_determinism(self):
        self.assertTrue(self.model.is_deterministic())
