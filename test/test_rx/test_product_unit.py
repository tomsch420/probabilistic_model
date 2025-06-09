import unittest

from random_events.interval import closed, open, closed_open
from random_events.variable import Integer, Continuous

from probabilistic_model.probabilistic_circuit.rustworkx.probabilistic_circuit import *
from probabilistic_model.distributions.uniform import UniformDistribution

import plotly.graph_objects as go


class ProductUnitTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")
    y: Continuous = Continuous("y")
    model: ProbabilisticCircuit

    def setUp(self):
        u1 = UnivariateContinuousLeaf(UniformDistribution(self.x, closed(0, 1).simple_sets[0]))
        u2 = UnivariateContinuousLeaf(UniformDistribution(self.y, closed(3, 4).simple_sets[0]))

        product_unit = ProductUnit()
        product_unit.add_subcircuit(u1)
        product_unit.add_subcircuit(u2)
        self.model = product_unit._probabilistic_circuit

    def test_setup(self):
        self.assertEqual(len(list(self.model.nodes)), 3)
        self.assertEqual(len(self.model.edges), 2)

    def test_variables(self):
        self.assertEqual(self.model.variables, SortedSet([self.x, self.y]))

    def test_leaves(self):
        self.assertEqual(len(self.model.leaves), 2)

    def test_likelihood(self):
        event = np.array([[0.5, 3.5]])
        result = self.model.likelihood(event)
        self.assertEqual(result, 1)

    def test_probability(self):
        event = SimpleEvent({self.x: closed(0, 0.5), self.y: closed(3, 3.5)}).as_composite_set()
        result = self.model.probability(event)
        self.assertEqual(result, 0.25)

    def test_mode(self):
        mode, likelihood = self.model.mode()
        self.assertEqual(likelihood, 1)
        self.assertEqual(mode,
                         SimpleEvent({self.x: closed(0, 1), self.y: closed(3, 4)}).as_composite_set())

    def test_sample(self):
        samples = self.model.sample(100)
        likelihood = self.model.likelihood(samples)
        self.assertTrue(all(likelihood > 0))

    def test_moment(self):
        expectation = self.model.expectation(self.model.variables)
        self.assertEqual(expectation[self.x], 0.5)
        self.assertEqual(expectation[self.y], 3.5)

    def test_conditional(self):
        event = SimpleEvent({self.x: closed(0, 0.5)}).as_composite_set()
        result, probability = self.model.conditional(event)
        self.assertEqual(probability, 0.5)
        self.assertEqual(len(list(result.nodes)), 3)
        self.assertIsInstance(result.root, ProductUnit)

    def test_conditional_with_0_evidence(self):
        event = SimpleEvent({self.x: closed(1.5, 2)}).as_composite_set()
        result, probability = self.model.conditional(event)
        self.assertEqual(probability, 0)
        self.assertEqual(result, None)

    def test_marginal_with_intersecting_variables(self):
        marginal = self.model.marginal([self.x])
        self.assertEqual(len(list(marginal.nodes)), 1)
        self.assertEqual(marginal.variables, SortedSet([self.x]))

    def test_marginal_without_intersecting_variables(self):
        marginal = self.model.marginal([])

        self.assertIsNone(marginal)

    def test_domain(self):
        domain = self.model.support
        domain_by_hand = SimpleEvent({self.x: closed(0, 1),
                                      self.y: closed(3, 4)}).as_composite_set()
        self.assertEqual(domain, domain_by_hand)

    def test_serialization(self):
        serialized = self.model.to_json()
        deserialized = ProductUnit.from_json(serialized)
        self.assertEqual(self.model, deserialized)

    def test_copy(self):
        copy = self.model.__deepcopy__()
        self.assertEqual(len(self.model.nodes), len(copy.nodes))
        self.assertEqual(len(self.model.edges), len(copy.edges))
        self.assertNotEqual(id(copy), id(self.model))

    def test_sample_not_equal(self):
        samples = self.model.sample(10)
        uniques = np.unique(samples, axis=0)
        self.assertEqual(len(samples), len(uniques))

    def test_determinism(self):
        self.assertTrue(self.model.is_deterministic())


if __name__ == '__main__':
    unittest.main()