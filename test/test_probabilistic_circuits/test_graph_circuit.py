import random
import unittest

import numpy as np
import portion
from matplotlib import pyplot as plt
from random_events.variables import Integer, Continuous
from typing_extensions import Union

from probabilistic_model.distributions.multinomial import MultinomialDistribution
from probabilistic_model.probabilistic_circuit.distributions.distributions import (ContinuousDistribution,
                                                                                   UniformDistribution,
                                                                                   GaussianDistribution)
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import *


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


class ProductUnitTestCase(unittest.TestCase, ShowMixin):
    x: Continuous = Continuous("x")
    y: Continuous = Continuous("y")
    model: DecomposableProductUnit

    def setUp(self):
        u1 = UniformDistribution(self.x, portion.closed(0, 1))
        u2 = UniformDistribution(self.y, portion.closed(3, 4))

        product_unit = DecomposableProductUnit()
        product_unit.probabilistic_circuit.add_nodes_from([product_unit, u1, u2])
        product_unit.probabilistic_circuit.add_edges_from([(product_unit, u1), (product_unit, u2)])
        self.model = product_unit

    def test_setup(self):
        self.assertEqual(len(self.model.probabilistic_circuit.nodes()), 3)
        self.assertEqual(len(self.model.probabilistic_circuit.edges()), 2)

    def test_variables(self):
        self.assertEqual(self.model.variables, (self.x, self.y))

    def test_leaves(self):
        self.assertEqual(len(self.model.leaves), 2)

    def test_likelihood(self):
        event = [0.5, 3.5]
        result = self.model.likelihood(event)
        self.assertEqual(result, 1)

    def test_probability(self):
        event = Event({self.x: portion.closed(0, 0.5), self.y: portion.closed(3, 3.5)})
        result = self.model.probability(event)
        self.assertEqual(result, 0.25)

    def test_mode(self):
        mode, likelihood = self.model.mode()
        self.assertEqual(likelihood, 1)
        self.assertEqual(mode, [Event({self.x: portion.closed(0, 1), self.y: portion.closed(3, 4)})])

    def test_sample(self):
        samples = self.model.sample(100)
        for sample in samples:
            self.assertGreater(self.model.likelihood(sample), 0)

    def test_moment(self):
        expectation = self.model.expectation(self.model.variables)
        self.assertEqual(expectation[self.x], 0.5)
        self.assertEqual(expectation[self.y], 3.5)

    def test_conditional(self):
        event = Event({self.x: portion.closed(0, 0.5)})
        result, probability = self.model.conditional(event)
        self.assertEqual(probability, 0.5)
        self.assertEqual(len(result.probabilistic_circuit.nodes()), 3)
        self.assertIsInstance(result, DecomposableProductUnit)
        self.assertIsInstance(result.probabilistic_circuit.root, DecomposableProductUnit)

    def test_conditional_with_0_evidence(self):
        event = Event({self.x: portion.closed(1.5, 2)})
        result, probability = self.model.conditional(event)
        self.assertEqual(probability, 0)
        self.assertEqual(result, None)

    def test_marginal_with_intersecting_variables(self):
        marginal = self.model.marginal([self.x])
        self.assertEqual(len(marginal.probabilistic_circuit.nodes()), 2)
        self.assertEqual(marginal.probabilistic_circuit.variables, (self.x,))

    def test_marginal_without_intersecting_variables(self):
        marginal = self.model.marginal([])
        self.assertIsNone(marginal)

    def test_domain(self):
        domain = self.model.domain
        self.assertEqual(domain[self.x], portion.closed(0, 1))
        self.assertEqual(domain[self.y], portion.closed(3, 4))

    def test_serialization(self):
        serialized = self.model.to_json()
        deserialized = DecomposableProductUnit.from_json(serialized)
        self.assertEqual(self.model, deserialized)

    def test_copy(self):
        copy = self.model.__copy__()
        self.assertEqual(self.model, copy)
        self.assertNotEqual(id(copy), id(self.model))

    def test_sample_not_equal(self):
        samples = self.model.sample(10)
        for sample in samples:
            same_samples = [s for s in samples if s == sample]
            self.assertEqual(len(same_samples), 1)


class SumUnitTestCase(unittest.TestCase, ShowMixin):
    x: Continuous = Continuous("x")
    model: DeterministicSumUnit

    def setUp(self):
        u1 = UniformDistribution(self.x, portion.closed(0, 1))
        u2 = UniformDistribution(self.x, portion.closed(3, 4))

        sum_unit = DeterministicSumUnit()
        e1 = (sum_unit, u1, 0.6)
        e2 = (sum_unit, u2, 0.4)

        sum_unit.probabilistic_circuit.add_weighted_edges_from([e1, e2])
        self.model = sum_unit

    def test_setup(self):
        self.assertEqual(len(self.model.probabilistic_circuit.nodes()), 3)
        self.assertEqual(len(self.model.probabilistic_circuit.edges()), 2)

    def test_variables(self):
        self.assertEqual(self.model.variables, (self.x,))

    def test_latent_variable(self):
        self.assertEqual(self.model.latent_variable.domain, (0, 1))

    def test_domain(self):
        domain = self.model.domain
        self.assertEqual(domain[self.x], portion.closed(0, 1) | portion.closed(3, 4))

    def test_weighted_subcircuits(self):
        weighted_subcircuits = self.model.weighted_subcircuits
        self.assertEqual(len(weighted_subcircuits), 2)
        self.assertEqual([weighted_subcircuit[0] for weighted_subcircuit in weighted_subcircuits], [0.6, 0.4])

    def test_likelihood(self):
        event = [0.5]
        result = self.model.likelihood(event)
        self.assertEqual(result, 0.6)

    def test_probability(self):
        event = Event({self.x: portion.closed(0, 3.5)})
        result = self.model.probability(event)
        self.assertEqual(result, 0.8)

    def test_conditional(self):
        event = Event({self.x: portion.closed(0, 0.5)})
        result, probability = self.model.conditional(event)
        self.assertEqual(probability, 0.3)
        self.assertEqual(len(result.probabilistic_circuit.nodes()), 2)
        self.assertIsInstance(result, DeterministicSumUnit)
        self.assertIsInstance(result.probabilistic_circuit.root, DeterministicSumUnit)
        self.assertEqual(len(result.weighted_subcircuits), 1)
        self.assertEqual(result.weighted_subcircuits[0][0], 1)

    def test_conditional_impossible(self):
        event = Event({self.x: portion.closed(5, 6)})
        result, probability = self.model.conditional(event)
        self.assertEqual(probability, 0.)
        self.assertIsNone(result)

    def test_sample(self):
        samples = self.model.sample(100)
        for sample in samples:
            self.assertGreater(self.model.likelihood(sample), 0)

    def test_moment(self):
        expectation = self.model.expectation(self.model.variables)
        self.assertEqual(expectation[self.x], 0.5 * 0.6 + 0.4 * 3.5)

    def test_marginal(self):
        marginal = self.model.marginal([self.x])
        self.assertEqual(self.model, marginal)

    def test_mode(self):
        mode, likelihood = self.model.mode()
        self.assertEqual(likelihood, 0.6)
        self.assertEqual(mode, [Event({self.x: portion.closed(0, 1)})])

    def test_serialization(self):
        serialized = self.model.to_json()
        deserialized = SmoothSumUnit.from_json(serialized)
        self.assertEqual(self.model, deserialized)

    def test_copy(self):
        copy = self.model.__copy__()
        self.assertEqual(self.model, copy)
        self.assertNotEqual(id(copy), id(self.model))

    def test_conditional_inference(self):
        event = Event({self.x: portion.closed(0, 0.5)})
        result, probability = self.model.conditional(event)
        self.assertEqual(result.probability(event), 1)

    def test_deep_mount(self):
        s1 = SmoothSumUnit()
        s2 = SmoothSumUnit()
        s3 = SmoothSumUnit()
        u1 = UniformDistribution(self.x, portion.closed(0, 1))
        s2.probabilistic_circuit.add_nodes_from([s2, s3, u1])
        s2.probabilistic_circuit.add_weighted_edges_from([(s2, s3, 1.), (s3, u1, 1.)])
        s1.mount(s2)
        self.assertEqual(len(s1.probabilistic_circuit.nodes()), 4)

    def test_sample_not_equal(self):
        samples = self.model.sample(10)
        for sample in samples:
            same_samples = [s for s in samples if s == sample]
            self.assertEqual(len(same_samples), 1)


class MinimalGraphCircuitTestCase(unittest.TestCase, ShowMixin):
    integer = Integer("integer", (1, 2, 4))
    symbol = Symbolic("symbol", ("a", "b", "c"))
    real = Continuous("x")
    real2 = Continuous("y")
    real3 = Continuous("z")

    model: ProbabilisticCircuit

    def setUp(self):
        model = ProbabilisticCircuit()

        u1 = UniformDistribution(self.real, portion.closed(0, 1))
        u2 = UniformDistribution(self.real, portion.closed(3, 5))
        model.add_node(u1)
        model.add_node(u2)

        sum_unit_1 = DeterministicSumUnit()

        model.add_node(sum_unit_1)

        model.add_edge(sum_unit_1, u1, weight=0.5)
        model.add_edge(sum_unit_1, u2, weight=0.5)

        u3 = UniformDistribution(self.real2, portion.closed(2, 2.25))
        u4 = UniformDistribution(self.real2, portion.closed(2, 5))
        sum_unit_2 = SmoothSumUnit()
        model.add_nodes_from([u3, u4, sum_unit_2])

        e3 = (sum_unit_2, u3, 0.7)
        e4 = (sum_unit_2, u4, 0.3)
        model.add_weighted_edges_from([e3, e4])

        product_1 = DecomposableProductUnit()
        model.add_node(product_1)

        e5 = (product_1, sum_unit_1)
        e6 = (product_1, sum_unit_2)
        model.add_edges_from([e5, e6])

        self.model = model

    def test_setup(self):
        self.assertEqual(len(self.model.nodes()), 7)
        self.assertEqual(len(self.model.edges()), 6)

    def test_variables(self):
        self.assertEqual(self.model.variables, (self.real, self.real2))

    def test_is_valid(self):
        self.assertTrue(self.model.is_valid())

    def test_root(self):
        self.assertIsInstance(self.model.root, DecomposableProductUnit)

    def test_variables_of_component(self):
        self.assertEqual(self.model.root.variables, (self.real, self.real2))
        self.assertEqual(list(self.model.nodes)[2].variables, (self.real,))
        self.assertEqual(list(self.model.nodes)[5].variables, (self.real2,))
        self.assertEqual(list(self.model.nodes)[6].variables, (self.real, self.real2))

    def test_likelihood(self):
        event = [1., 2.]
        result = self.model.likelihood(event)
        self.assertEqual(result, 0.5 * ((4 * 0.7) + (1 / 3 * 0.3)))

    def test_probability_everywhere(self):
        event = Event({self.real: portion.closed(0, 5), self.real2: portion.closed(2, 5)})
        result = self.model.probability(event)
        self.assertEqual(result, 1.)

    def test_probability_nowhere(self):
        event = Event({self.real: portion.closed(0, 0.5), self.real2: portion.closed(0, 1)})
        result = self.model.probability(event)
        self.assertEqual(result, 0.)

    def test_probability_somewhere(self):
        event = Event({self.real2: portion.closed(0, 3)})
        result = self.model.probability(event)
        self.assertAlmostEqual(result, 0.8)

    def test_caching_reset(self):
        event = Event({self.real: portion.closed(0, 5), self.real2: portion.closed(2, 5)})
        _ = self.model.probability(event)

        for node in self.model.nodes():
            self.assertIsNone(node.result_of_current_query)
            self.assertFalse(node.cache_result)

    def test_caching(self):
        event = Event({self.real: portion.closed(0, 5), self.real2: portion.closed(2, 5)})
        self.model.root.cache_result = True
        _ = self.model.root.probability(event)

        for node in self.model.nodes():
            if not isinstance(node, ContinuousDistribution):
                self.assertIsNotNone(node.result_of_current_query)

    def test_mode(self):
        marginal = self.model.marginal([self.real])
        mode, likelihood = marginal.mode()
        self.assertEqual(likelihood, 0.5)
        self.assertEqual(mode, [Event({self.real: portion.closed(0, 1)})])

    def test_mode_raising(self):
        with self.assertRaises(NotImplementedError):
            _ = self.model.mode()

    def test_conditional(self):
        event = Event({self.real: portion.closed(0, 1)})
        result, probability = self.model.conditional(event)
        self.assertEqual(len(result.nodes()), 6)

    def test_sample(self):
        samples = self.model.sample(100)
        for sample in samples:
            self.assertGreater(self.model.likelihood(sample), 0)

    def test_serialization(self):
        serialized = self.model.to_json()
        deserialized = ProbabilisticCircuit.from_json(serialized)
        self.assertEqual(self.model, deserialized)

    def test_update_variables(self):
        self.model.update_variables(VariableMap({self.real: self.real3}))
        self.assertEqual(self.model.variables, (self.real2, self.real3))

    def test_sample_not_equal(self):
        samples = self.model.sample(10)
        for sample in samples:
            same_samples = [s for s in samples if s == sample]
            self.assertEqual(len(same_samples), 1)


class FactorizationTestCase(unittest.TestCase, ShowMixin):
    x: Continuous = Continuous("x")
    y: Continuous = Continuous("y")
    z: Continuous = Continuous("z")
    sum_unit_1: SmoothSumUnit
    sum_unit_2: SmoothSumUnit
    interaction_model: MultinomialDistribution

    def setUp(self):
        u1 = UniformDistribution(self.x, portion.closed(0, 1))
        u2 = UniformDistribution(self.x, portion.closed(3, 4))
        sum_unit_1 = DeterministicSumUnit()
        sum_unit_1.add_subcircuit(u1, 0.5)
        sum_unit_1.add_subcircuit(u2, 0.5)
        self.sum_unit_1 = sum_unit_1

        u3 = UniformDistribution(self.y, portion.closed(0, 1))
        u4 = UniformDistribution(self.y, portion.closed(5, 6))
        sum_unit_2 = DeterministicSumUnit()
        sum_unit_2.add_subcircuit(u3, 0.5)
        sum_unit_2.add_subcircuit(u4, 0.5)
        self.sum_unit_2 = sum_unit_2

        interaction_probabilities = np.array([[0, 0.5], [0.3, 0.2]])

        if self.sum_unit_1.latent_variable > self.sum_unit_2.latent_variable:
            interaction_probabilities = interaction_probabilities.T

        self.interaction_model = MultinomialDistribution([sum_unit_1.latent_variable, sum_unit_2.latent_variable],
                                                         interaction_probabilities)

    def test_setup(self):
        self.assertEqual(self.interaction_model.marginal([self.sum_unit_1.latent_variable]).probabilities.tolist(),
                         [0.5, 0.5])
        self.assertEqual(self.interaction_model.marginal([self.sum_unit_2.latent_variable]).probabilities.tolist(),
                         [0.3, 0.7])
        self.assertEqual(len(self.sum_unit_1.probabilistic_circuit.nodes()), 3)
        self.assertEqual(len(self.sum_unit_2.probabilistic_circuit.nodes()), 3)

    def test_mount_with_interaction(self):
        self.sum_unit_1.mount_with_interaction_terms(self.sum_unit_2, self.interaction_model)
        self.assertIsInstance(self.sum_unit_1.probabilistic_circuit.root, DeterministicSumUnit)

        for subcircuit in self.sum_unit_1.subcircuits:
            self.assertIsInstance(subcircuit, DecomposableProductUnit)

        self.assertEqual(len(self.sum_unit_1.probabilistic_circuit.nodes()), 9)
        self.assertEqual(len(self.sum_unit_1.probabilistic_circuit.edges()), 9)
        self.assertEqual(1, self.sum_unit_1.probability(Event()))


class MountedInferenceTestCase(unittest.TestCase, ShowMixin):
    x: Continuous = Continuous("x")
    y: Continuous = Continuous("y")

    probabilities = np.array([[0, 1],
                              [1, 0]])
    model: DeterministicSumUnit

    def setUp(self):
        random.seed(69)
        model = DeterministicSumUnit()
        model.add_subcircuit(UniformDistribution(self.x, portion.closed(-1.5, -0.5)), 0.5)
        model.add_subcircuit(UniformDistribution(self.x, portion.closed(0.5, 1.5)), 0.5)
        next_model = model.__copy__()
        for leaf in next_model.leaves:
            leaf._variables = (self.y,)

        transition_model = MultinomialDistribution([model.latent_variable, next_model.latent_variable],
                                                   self.probabilities)
        next_model.mount_with_interaction_terms(model, transition_model)
        self.model = next_model

    def test_setup(self):
        self.assertEqual(self.model.variables, (self.x, self.y))
        self.assertTrue(self.model.probabilistic_circuit.is_decomposable())

    def test_sample_from_uniform(self):
        for leaf in self.model.leaves:
            samples = leaf.sample(2)
            self.assertNotEqual(samples[0], samples[1])

    def test_sample(self):
        samples: List = self.model.sample(2)
        self.assertEqual(len(samples), 2)

        self.assertNotEqual(samples[0], samples[1])

    def test_samples_in_sequence(self):
        samples = self.model.probabilistic_circuit.sample(1) + self.model.probabilistic_circuit.sample(1)
        self.assertEqual(len(samples), 2)
        self.assertNotEqual(samples[0], samples[1])

    def test_plot_non_deterministic(self):
        gaussian_1 = GaussianDistribution(Continuous("x"), 0, 1)
        gaussian_2 = GaussianDistribution(Continuous("x"), 5, 0.5)
        mixture = SmoothSumUnit()
        mixture.add_subcircuit(gaussian_1, 0.5)
        mixture.add_subcircuit(gaussian_2, 0.5)
        traces = mixture.plot()
        self.assertGreater(len(traces), 0)
        # go.Figure(mixture.plot(), mixture.plotly_layout()).show()

    def test_simplify(self):
        simplified = self.model.simplify()
        self.assertEqual(len(simplified.probabilistic_circuit.nodes()), 7)
        self.assertEqual(len(simplified.probabilistic_circuit.edges()), 6)

    def test_plot_2d(self):
        traces = self.model.plot_2d()
        assert len(traces) > 0
        # go.Figure(traces, self.model.plotly_layout()).show()


class ComplexMountedInferenceTestCase(unittest.TestCase, ShowMixin):
    x: Continuous = Continuous("x")
    y: Continuous = Continuous("y")

    probabilities = np.array([[0.9, 0.1],
                              [0.3, 0.7]])
    model: DeterministicSumUnit

    def setUp(self):
        random.seed(69)
        model = DeterministicSumUnit()
        model.add_subcircuit(UniformDistribution(self.x, portion.closed(-1.5, -0.5)), 0.5)
        model.add_subcircuit(UniformDistribution(self.x, portion.closed(0.5, 1.5)), 0.5)
        next_model = model.__copy__()
        for leaf in next_model.leaves:
            leaf._variables = (self.y,)

        transition_model = MultinomialDistribution([model.latent_variable, next_model.latent_variable],
                                                   self.probabilities)
        next_model.mount_with_interaction_terms(model, transition_model)
        self.model = next_model

    def test_simplify(self):
        simplified = self.model.probabilistic_circuit.simplify().root
        self.assertEqual(len(simplified.probabilistic_circuit.nodes()), len(self.model.probabilistic_circuit.nodes))
        self.assertEqual(len(simplified.probabilistic_circuit.edges()), len(self.model.probabilistic_circuit.edges))

    def test_sample_not_equal(self):
        samples = self.model.sample(10)
        for sample in samples:
            same_samples = [s for s in samples if s == sample]
            self.assertEqual(len(same_samples), 1)

    def test_serialization(self):
        model = self.model.probabilistic_circuit
        serialized_model = model.to_json()
        deserialized_model = ProbabilisticCircuit.from_json(serialized_model)
        self.assertIsInstance(deserialized_model, ProbabilisticCircuit)
        self.assertEqual(len(model.nodes), len(deserialized_model.nodes))
        self.assertEqual(len(model.edges), len(deserialized_model.edges))
        event = Event({self.x: portion.closed(-1, 1), self.y: portion.closed(-1, 1)})
        self.assertEqual(model.probability(event), deserialized_model.probability(event))


class NormalizationTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")

    def test_normalization(self):
        u1 = UniformDistribution(self.x, portion.closed(0, 1))
        u2 = UniformDistribution(self.x, portion.closed(3, 4))
        sum_unit = DeterministicSumUnit()
        sum_unit.add_subcircuit(u1, 0.5)
        sum_unit.add_subcircuit(u2, 0.3)
        sum_unit.normalize()
        self.assertAlmostEqual(sum_unit.weights[0], 0.5/0.8)
        self.assertAlmostEqual(sum_unit.weights[1], 0.3/0.8)


class MultivariateGaussianTestCase(unittest.TestCase):

    x: Continuous = Continuous("x")
    y: Continuous = Continuous("y")
    model: ProbabilisticCircuit

    def setUp(self):
        product = DecomposableProductUnit()
        n1 = GaussianDistribution(self.x, 0, 1)
        n2 = GaussianDistribution(self.y, 0.5, 2)
        product.add_subcircuit(n1)
        product.add_subcircuit(n2)
        self.model = product.probabilistic_circuit

    def test_plot_2d(self):
        traces = self.model.plot()
        assert len(traces) > 0
        # go.Figure(traces, self.model.plotly_layout()).show()


if __name__ == '__main__':
    unittest.main()
