import json
import os.path
import unittest
from typing import Type

import numpy as np
import portion
from matplotlib import pyplot as plt
from random_events.variables import Integer, Continuous
from typing_extensions import Union
from probabilistic_model.distributions.multinomial import MultinomialDistribution
from probabilistic_model.learning.jpt.jpt import JPT, DecomposableProductUnit as JPTLeaf
from probabilistic_model.probabilistic_circuit.convolution.convolution import (UniformDistributionConvolution,
                                                                               GaussianDistributionConvolution,
                                                                               TruncatedGaussianDistributionConvolution,
                                                                               DiracDeltaDistributionConvolution)
from probabilistic_model.probabilistic_circuit.distributions.distributions import (ContinuousDistribution,
                                                                                   UniformDistribution,
                                                                                   GaussianDistribution,
                                                                                   DiracDeltaDistribution,
                                                                                   TruncatedGaussianDistribution)
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import *
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit


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
        self.assertEqual(mode.events,
                         [Event({self.x: portion.closed(0, 1), self.y: portion.closed(3, 4)})])

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
        domain = self.model.domain.events[0]
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

    def test_determinism(self):
        self.assertTrue(self.model.is_deterministic())


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
        self.assertEqual(domain.events[0][self.x], portion.closed(0, 1) | portion.closed(3, 4))

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
        self.assertEqual(mode.events[0], Event({self.x: portion.closed(0, 1)}))

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

    def test_determinism(self):
        self.assertTrue(self.model.is_deterministic())


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
        self.assertEqual(mode.events, [Event({self.real: portion.closed(0, 1)})])

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

    def test_determinism(self):
        self.assertFalse(self.model.is_deterministic())

    def test_avm(self):
        x = Continuous("x")
        y = Continuous("y")
        c1 = DecomposableProductUnit()
        c1.add_subcircuit(UniformDistribution(x, portion.closed(2, 3)))
        c1.add_subcircuit(UniformDistribution(y, portion.closed(0, 1)))

        c2 = DecomposableProductUnit()
        c2.add_subcircuit(UniformDistribution(x, portion.closed(6, 7)))
        c2.add_subcircuit(UniformDistribution(y, portion.closed(0, 1)))
        result = c1.area_validation_metric(c2)
        self.assertAlmostEqual(result, 1)
    def test_avm_is_equal(self):

        other_model = ProbabilisticCircuit()

        u1 = UniformDistribution(self.real, portion.closed(0, 1))
        u2 = UniformDistribution(self.real, portion.closed(3, 5))
        other_model.add_node(u1)
        other_model.add_node(u2)

        sum_unit_1 = DeterministicSumUnit()

        other_model.add_node(sum_unit_1)

        other_model.add_edge(sum_unit_1, u1, weight=0.5)
        other_model.add_edge(sum_unit_1, u2, weight=0.5)

        u3 = UniformDistribution(self.real2, portion.closed(2, 2.25))
        u4 = UniformDistribution(self.real2, portion.closed(2, 5))
        sum_unit_2 = SmoothSumUnit()
        other_model.add_nodes_from([u3, u4, sum_unit_2])

        e3 = (sum_unit_2, u3, 0.7)
        e4 = (sum_unit_2, u4, 0.3)
        other_model.add_weighted_edges_from([e3, e4])

        product_1 = DecomposableProductUnit()
        other_model.add_node(product_1)

        e5 = (product_1, sum_unit_1)
        e6 = (product_1, sum_unit_2)
        other_model.add_edges_from([e5, e6])
        self.assertEqual(self.model.root.area_validation_metric(other_model.root), 0)

    def test_avm_not_equal_weights(self):

        other_model = ProbabilisticCircuit()

        u1 = UniformDistribution(self.real, portion.closed(0, 1))
        u2 = UniformDistribution(self.real, portion.closed(3, 5))
        other_model.add_node(u1)
        other_model.add_node(u2)

        sum_unit_1 = DeterministicSumUnit()

        other_model.add_node(sum_unit_1)

        other_model.add_edge(sum_unit_1, u1, weight=0.02)
        other_model.add_edge(sum_unit_1, u2, weight=0.98)

        u3 = UniformDistribution(self.real2, portion.closed(2, 2.25))
        u4 = UniformDistribution(self.real2, portion.closed(2, 5))
        sum_unit_2 = SmoothSumUnit()
        other_model.add_nodes_from([u3, u4, sum_unit_2])

        e3 = (sum_unit_2, u3, 0.6)
        e4 = (sum_unit_2, u4, 0.4)
        other_model.add_weighted_edges_from([e3, e4])

        product_1 = DecomposableProductUnit()
        other_model.add_node(product_1)

        e5 = (product_1, sum_unit_1)
        e6 = (product_1, sum_unit_2)
        other_model.add_edges_from([e5, e6])

        self.assertEqual(self.model.root.area_validation_metric(other_model.root), 0)


    def test_avm_not_equal_leafs(self):

        other_model = ProbabilisticCircuit()

        u1 = UniformDistribution(self.real, portion.closed(0, 1))
        u2 = UniformDistribution(self.real, portion.closed(3, 7))
        other_model.add_node(u1)
        other_model.add_node(u2)

        sum_unit_1 = DeterministicSumUnit()

        other_model.add_node(sum_unit_1)

        other_model.add_edge(sum_unit_1, u1, weight=0.5)
        other_model.add_edge(sum_unit_1, u2, weight=0.5)

        u3 = UniformDistribution(self.real2, portion.closed(2, 3))
        u4 = UniformDistribution(self.real2, portion.closed(4, 8))
        sum_unit_2 = SmoothSumUnit()
        other_model.add_nodes_from([u3, u4, sum_unit_2])

        e3 = (sum_unit_2, u3, 0.7)
        e4 = (sum_unit_2, u4, 0.3)
        other_model.add_weighted_edges_from([e3, e4])

        product_1 = DecomposableProductUnit()
        other_model.add_node(product_1)

        e5 = (product_1, sum_unit_1)
        e6 = (product_1, sum_unit_2)
        other_model.add_edges_from([e5, e6])
        print(self.model.is_structured_decomposable(), self.model.decomposes_as(other_model))

        self.assertAlmostEqual(self.model.root.area_validation_metric(other_model.root), 0.56, places=2)










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
        #  go.Figure(traces, self.model.plotly_layout()).show()


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


class MultivariateGaussianTestCase(unittest.TestCase, ShowMixin):

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

    def test_truncation(self):
        event = Event({self.x: portion.open(-0.1, 0.1), self.y: portion.open(-0.1, 0.1)})
        outer_event = event.complement()

        # first truncation
        conditional, probability = self.model.conditional(outer_event)

        self.assertEqual(outer_event, conditional.domain)

        # go.Figure(conditional.plot(), conditional.plotly_layout()).show()
        samples = list(conditional.sample(500))

        for sample in samples:
            self.assertTrue(conditional.likelihood(sample) > 0)
            self.assertFalse(sample[0] in event[self.x] and sample[1] in event[self.y])

        # second truncation
        limiting_event = Event({self.x: portion.open(-2, 2), self.y: portion.open(-2, 2)})

        conditional, probability = conditional.conditional(limiting_event)
        #  self.show(conditional)
        #  go.Figure(conditional.domain.plot()).show()
        self.assertEqual(len(conditional.sample(500)), 500)

        # go.Figure(conditional.plot(), conditional.plotly_layout()).show()

        domain = outer_event & limiting_event
        self.assertEqual(domain, conditional.domain)

    def test_open_closed_set_bug(self):
        tg1 = TruncatedGaussianDistribution(self.y, portion.open(-0.1, 0.1), 0, 1)
        event = Event({self.x: portion.open(-2, 2), self.y: portion.open(-2, 2)})
        r, _ = tg1.conditional(event)

        self.assertIsNotNone(r)


class ComplexInferenceTestCase(unittest.TestCase):

    x: Continuous = Continuous("x")
    y: Continuous = Continuous("y")
    model: ProbabilisticCircuit

    e1: Event = Event({x: portion.closed(0, 1), y: portion.closedopen(0, 1)})
    e2: Event = Event({x: portion.closed(1.5, 2), y: portion.closed(1.5, 2)})

    event: ComplexEvent = e1 | e2

    def setUp(self):
        root = DecomposableProductUnit()
        px = UniformDistribution(self.x, portion.closed(0, 2))
        py = UniformDistribution(self.y, portion.closed(0, 3))
        root.add_subcircuit(px)
        root.add_subcircuit(py)
        self.model = root.probabilistic_circuit

    def test_complex_probability(self):
        p = self.model.probability(self.event)
        self.assertEqual(self.model.probability(self.e1) + self.model.probability(self.e2), p)

    def test_complex_conditional(self):
        conditional, probability = self.model.conditional(self.event)
        self.assertAlmostEqual(conditional.probability(self.event), 1.)


class ConvolutionTestCase(unittest.TestCase, ShowMixin):

    def setUp(self):
        self.variable = Continuous("x")
        self.interval = portion.closed(-1, 1)
        self.mean = 0
        self.scale = 1
        self.location = 3
        self.density_cap = 1

    def test_uniform_with_dirac_delta_convolution(self):
        uniform_distribution = UniformDistribution(self.variable, self.interval)
        dirac_delta_distribution = DiracDeltaDistribution(self.variable, self.location, self.density_cap)
        convolution = UniformDistributionConvolution(uniform_distribution)
        result = convolution.convolve_with_dirac_delta(dirac_delta_distribution)
        self.assertEqual(result.interval, portion.closed(self.interval.lower + self.location,
                                                         self.interval.upper + self.location))

    def test_dirac_delta_with_dirac_delta_convolution(self):
        dirac_delta_distribution1 = DiracDeltaDistribution(self.variable, self.location, self.density_cap)
        dirac_delta_distribution2 = DiracDeltaDistribution(self.variable, self.location, self.density_cap)
        convolution = DiracDeltaDistributionConvolution(dirac_delta_distribution1)
        result = convolution.convolve_with_dirac_delta(dirac_delta_distribution2)
        self.assertEqual(result.location, self.location * 2)

    def test_gaussian_with_dirac_delta_convolution(self):
        gaussian_distribution = GaussianDistribution(self.variable, self.mean, self.scale)
        dirac_delta_distribution = DiracDeltaDistribution(self.variable, self.location, self.density_cap)
        convolution = GaussianDistributionConvolution(gaussian_distribution)
        result = convolution.convolve_with_dirac_delta(dirac_delta_distribution)
        self.assertEqual(result.mean, self.mean + self.location)

    def test_gaussian_with_gaussian_convolution(self):
        gaussian_distribution1 = GaussianDistribution(self.variable, self.mean, self.scale)
        gaussian_distribution2 = GaussianDistribution(self.variable, self.mean, self.scale)
        convolution = GaussianDistributionConvolution(gaussian_distribution1)
        result = convolution.convolve_with_gaussian(gaussian_distribution2)
        self.assertEqual(result.mean, self.mean * 2)
        self.assertEqual(result.scale, self.scale * 2)

    def test_truncated_gaussian_with_dirac_delta_convolution(self):
        truncated_gaussian_distribution = TruncatedGaussianDistribution(self.variable, self.interval, self.mean,
                                                                        self.scale)
        dirac_delta_distribution = DiracDeltaDistribution(self.variable, self.location, self.density_cap)
        convolution = TruncatedGaussianDistributionConvolution(truncated_gaussian_distribution)
        result = convolution.convolve_with_dirac_delta(dirac_delta_distribution)
        self.assertEqual(result.interval, portion.closed(self.interval.lower + self.location,
                                                         self.interval.upper + self.location))
        self.assertEqual(result.mean, self.mean + self.location)

class StructuredDecomposabilityTestCase(unittest.TestCase):

    model = ProbabilisticCircuit()
    x = Continuous("x")
    y = Continuous("y")
    z = Continuous("z")

    sum_unit_1 = DeterministicSumUnit()
    model.add_node(sum_unit_1)
    product_1, product_2, product_3 = DecomposableProductUnit(), DecomposableProductUnit(), DecomposableProductUnit()
    product_4, product_5, product_6 = DecomposableProductUnit(), DecomposableProductUnit(), DecomposableProductUnit()

    model.add_node(product_1)
    model.add_node(product_2)
    model.add_edge(sum_unit_1, product_1, weight=0.5)
    model.add_edge(sum_unit_1, product_2, weight=0.5)

    sum_unit_2 = DeterministicSumUnit()
    sum_unit_3 = DeterministicSumUnit()
    product_1.add_subcircuit(sum_unit_2)
    product_1.add_subcircuit(UniformDistribution(z, portion.closed(2, 3)))
    product_2.add_subcircuit(sum_unit_3)
    product_2.add_subcircuit(UniformDistribution(z, portion.closed(4, 5)))

    sum_unit_2.add_subcircuit(product_3, weight=0.5)
    sum_unit_2.add_subcircuit(product_4, weight=0.5)
    sum_unit_3.add_subcircuit(product_5, weight=0.5)
    sum_unit_3.add_subcircuit(product_6, weight=0.5)

    range1 = portion.closed(0, 2)
    range2 = portion.closed(4, 6)
    for unit in [product_3, product_4, product_5, product_6]:
        unit.add_subcircuit(UniformDistribution(x, range1))
        unit.add_subcircuit(UniformDistribution(y, range2))
    def test_is_structured_decomposable(self):
        assert self.model.is_structured_decomposable()

    def test_structured_decomposable_as_ture(self):
        model_other = ProbabilisticCircuit()
        x = Continuous("x")
        y = Continuous("y")
        z = Continuous("z")

        sum_unit_1 = DeterministicSumUnit()
        model_other.add_node(sum_unit_1)
        product_1, product_2, product_3 = DecomposableProductUnit(), DecomposableProductUnit(), DecomposableProductUnit()
        product_4, product_5, product_6 = DecomposableProductUnit(), DecomposableProductUnit(), DecomposableProductUnit()

        model_other.add_node(product_1)
        model_other.add_node(product_2)
        model_other.add_edge(sum_unit_1, product_1, weight=0.5)
        model_other.add_edge(sum_unit_1, product_2, weight=0.5)

        sum_unit_2 = DeterministicSumUnit()
        sum_unit_3 = DeterministicSumUnit()
        product_1.add_subcircuit(sum_unit_2)
        product_1.add_subcircuit(UniformDistribution(z, portion.closed(7, 19)))
        product_2.add_subcircuit(sum_unit_3)
        product_2.add_subcircuit(UniformDistribution(z, portion.closed(0, 5)))

        sum_unit_2.add_subcircuit(product_3, weight=0.3)
        sum_unit_2.add_subcircuit(product_4, weight=0.7)
        sum_unit_3.add_subcircuit(product_5, weight=0.5)
        sum_unit_3.add_subcircuit(product_6, weight=0.5)

        range1 = portion.closed(0, 5)
        range2 = portion.closed(4, 6)
        for unit in [product_3, product_4, product_5, product_6]:
            unit.add_subcircuit(UniformDistribution(x, range1))
            unit.add_subcircuit(UniformDistribution(y, range2))

        assert self.model.decomposes_as(model_other)

    def test_structured_decomposable_as_false(self):
        model_other = ProbabilisticCircuit()
        x = Continuous("z")
        y = Continuous("y")
        z = Continuous("x")

        sum_unit_1 = DeterministicSumUnit()
        model_other.add_node(sum_unit_1)
        product_1, product_2, product_3 = DecomposableProductUnit(), DecomposableProductUnit(), DecomposableProductUnit()
        product_4, product_5, product_6 = DecomposableProductUnit(), DecomposableProductUnit(), DecomposableProductUnit()

        model_other.add_node(product_1)
        model_other.add_node(product_2)
        model_other.add_edge(sum_unit_1, product_1, weight=0.5)
        model_other.add_edge(sum_unit_1, product_2, weight=0.5)

        sum_unit_2 = DeterministicSumUnit()
        sum_unit_3 = DeterministicSumUnit()
        product_1.add_subcircuit(sum_unit_2)
        product_1.add_subcircuit(UniformDistribution(z, portion.closed(7, 19)))
        product_2.add_subcircuit(sum_unit_3)
        product_2.add_subcircuit(UniformDistribution(z, portion.closed(0, 5)))

        sum_unit_2.add_subcircuit(product_3, weight=0.3)
        sum_unit_2.add_subcircuit(product_4, weight=0.7)
        sum_unit_3.add_subcircuit(product_5, weight=0.5)
        sum_unit_3.add_subcircuit(product_6, weight=0.5)

        range1 = portion.closed(0, 5)
        range2 = portion.closed(4, 6)
        for unit in [product_3, product_4, product_5, product_6]:
            unit.add_subcircuit(UniformDistribution(x, range1))
            unit.add_subcircuit(UniformDistribution(y, range2))

        assert not self.model.decomposes_as(model_other)

class AreaValidationMetricTestCase(unittest.TestCase):

    x = Continuous("x")
    y = Continuous("y")
    standard_circuit = JPTLeaf()
    standard_circuit.add_subcircuit(UniformDistribution(x, portion.closed(0, 1)))
    standard_circuit.add_subcircuit(UniformDistribution(y, portion.closed(0, 1)))
    standard_circuit = standard_circuit.probabilistic_circuit

    event_1 = Event({x: portion.closed(0, .25), y: portion.closed(0, .25)})
    event_2 = Event({x: portion.closed(0.75, 1), y: portion.closed(0.75, 1)})


    circuit_1, _ = standard_circuit.conditional(event_1.complement())
    circuit_2, _ = standard_circuit.conditional(event_2.complement())
    circuit_3, _ = circuit_2.conditional(event_1)
    circuit_4, _ = circuit_1.conditional(event_2)



    def test_jpt_avm(self):
        result = JPT.area_validation_metric(self.circuit_1.root, self.circuit_2.root)

        p_event_by_hand = self.event_2
        q_event_by_hand = self.event_1
        self.assertEqual(self.circuit_2.probability(p_event_by_hand), 0)
        self.assertEqual(self.circuit_1.probability(q_event_by_hand), 0)
        result_by_hand = self.circuit_1.probability(p_event_by_hand) + self.circuit_2.probability(q_event_by_hand)
        self.assertAlmostEqual(result, result_by_hand/2, 4)

    def test_jpt_avm_same_input(self):
        result = JPT.area_validation_metric(self.circuit_1.root, self.circuit_1.root)
        self.assertEqual(result, 0)

    def test_jpt_avm_disjunct_input(self):
        result = JPT.area_validation_metric(self.circuit_3.root, self.circuit_4.root)

        self.assertEqual(result, 1)

    def test_avm_mc(self):
        import probabilistic_model.Monte_Carlo_Estimator as mc
        result = mc.monte_carlo_estimation_area_validation_metric(sample_amount=1000, first_model=self.circuit_1, senc_model=self.circuit_2)

        self.assertEqual(result, 0.13333333333333336/2)

if __name__ == '__main__':
    unittest.main()
