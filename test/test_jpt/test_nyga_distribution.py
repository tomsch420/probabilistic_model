import unittest
from typing import List

import numpy as np
import plotly.graph_objects as go
import portion
from random_events.variables import Continuous

from probabilistic_model.learning.nyga_distribution import NygaDistribution, InductionStep
from probabilistic_model.probabilistic_circuit.distributions import DiracDeltaDistribution
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit, SmoothSumUnit
from probabilistic_model.utils import SubclassJSONSerializer


class InductionStepTestCase(unittest.TestCase):
    variable: Continuous = Continuous("x")
    sorted_data: List[float] = [1, 2, 3, 4, 7, 9]
    weights: List[float] = [1 / 6] * 6
    induction_step: InductionStep

    def setUp(self) -> None:
        nyga_distribution = NygaDistribution(self.variable, min_samples_per_quantile=1, min_likelihood_improvement=0.01)
        self.induction_step = InductionStep(self.sorted_data, 6, self.weights, 0, len(self.sorted_data),
                                            nyga_distribution)

    def test_variable(self):
        self.assertEqual(self.induction_step.variable, self.variable)

    def test_left_connecting_point_edge_case(self):
        self.assertEqual(self.induction_step.left_connecting_point(), 1)

    def test_right_connecting_point_edge_case(self):
        self.assertEqual(self.induction_step.right_connecting_point(), 9)

    def test_left_connecting_point(self):
        self.assertEqual(self.induction_step.left_connecting_point_from_index(3), 3.5)

    def test_right_connecting_point(self):
        self.assertEqual(self.induction_step.right_connecting_point_from_index(5), 8.)

    def test_create_uniform_distribution_edge_case(self):
        distribution = self.induction_step.create_uniform_distribution()
        self.assertEqual(distribution, UniformDistribution(self.variable, portion.closed(1, 9)))

    def test_create_uniform_distribution(self):
        distribution = self.induction_step.create_uniform_distribution_from_indices(3, 5)
        self.assertEqual(distribution, UniformDistribution(self.variable, portion.closedopen(3.5, 8.0)))

    def test_sum_weights(self):
        self.assertAlmostEqual(self.induction_step.sum_weights(), 1)

    def test_sum_weights_from_indices(self):
        self.assertAlmostEqual(self.induction_step.sum_weights_from_indices(3, 5), 1 / 3)

    def test_compute_best_split(self):
        maximum, index = self.induction_step.compute_best_split()
        self.assertEqual(index, 1)

    def test_compute_best_split_without_result(self):
        self.induction_step.nyga_distribution.min_samples_per_quantile = 4
        maximum, index = self.induction_step.compute_best_split()
        self.assertEqual(index, None)
        self.assertEqual(maximum, 0)

    def test_compute_best_split_with_induced_indices(self):
        self.induction_step.begin_index = 3
        maximum, index = self.induction_step.compute_best_split()
        self.assertEqual(index, 5)

    def test_construct_left_induction_step(self):
        induction_step = self.induction_step.construct_left_induction_step(1)
        self.assertEqual(induction_step.begin_index, 0)
        self.assertEqual(induction_step.end_index, 1)
        self.assertEqual(induction_step.data, self.induction_step.data)
        self.assertEqual(induction_step.weights, self.induction_step.weights)

    def test_construct_right_induction_step(self):
        induction_step = self.induction_step.construct_right_induction_step(1)
        self.assertEqual(induction_step.begin_index, 1)
        self.assertEqual(induction_step.end_index, 6)
        self.assertEqual(induction_step.data, self.induction_step.data)
        self.assertEqual(induction_step.weights, self.induction_step.weights)

    def test_fit(self):
        np.random.seed(69)
        data = np.random.normal(0, 1, 100).tolist()
        distribution = self.induction_step.nyga_distribution
        distribution.fit(data)
        self.assertAlmostEqual(sum([weight for weight, _ in distribution.weighted_subcircuits]), 1.)

    def test_domain(self):
        np.random.seed(69)
        data = np.random.normal(0, 1, 100).tolist()
        distribution = self.induction_step.nyga_distribution
        distribution.fit(data)
        domain = distribution.domain
        self.assertEqual(domain[self.variable], portion.closed(min(data), max(data)))

    def test_plot(self):
        np.random.seed(69)
        data = np.random.normal(0, 1, 100).tolist()
        distribution = self.induction_step.nyga_distribution
        distribution.fit(data)
        fig = go.Figure(distribution.plot())
        self.assertIsNotNone(fig)  # fig.show()

    def test_fit_from_singular_data(self):
        data = [1., 1.]
        distribution = self.induction_step.nyga_distribution
        distribution.fit(data)
        self.assertEqual(len(distribution.probabilistic_circuit.nodes), 2)
        self.assertEqual(distribution.weighted_subcircuits[0][0], 1.)
        self.assertIsInstance(distribution.subcircuits[0], DiracDeltaDistribution)

    def test_serialization(self):
        np.random.seed(69)
        data = np.random.normal(0, 1, 100).tolist()
        distribution = NygaDistribution(self.variable, min_likelihood_improvement=0.01)
        distribution.fit(data)
        serialized = distribution.to_json()
        deserialized = SubclassJSONSerializer.from_json(serialized)
        self.assertIsInstance(deserialized, NygaDistribution)
        self.assertEqual(distribution, deserialized)

    def test_equality_and_copy(self):
        np.random.seed(69)
        data = np.random.normal(0, 1, 100).tolist()
        distribution = self.induction_step.nyga_distribution
        distribution.fit(data)
        distribution_ = distribution.__copy__()
        self.assertEqual(distribution, distribution_)

    def test_from_mixture_of_uniform_distributions(self):
        u1 = UniformDistribution(self.variable, portion.closed(0, 5))
        u2 = UniformDistribution(self.variable, portion.closed(2, 3))
        sum_unit = SmoothSumUnit()
        e1 = (sum_unit, u1, 0.5)
        e2 = (sum_unit, u2, 0.5)
        sum_unit.probabilistic_circuit.add_weighted_edges_from([e1, e2])
        distribution = NygaDistribution.from_uniform_mixture(sum_unit)

        solution_by_hand = NygaDistribution(self.variable)
        solution_by_hand.probabilistic_circuit = ProbabilisticCircuit()
        leaf_1 = UniformDistribution(self.variable, portion.closedopen(0, 2))
        leaf_2 = UniformDistribution(self.variable, portion.closedopen(2, 3))
        leaf_3 = UniformDistribution(self.variable, portion.closed(3, 5))

        e1 = (solution_by_hand, leaf_1, 0.2)
        e2 = (solution_by_hand, leaf_2, 0.6)
        e3 = (solution_by_hand, leaf_3, 0.2)

        solution_by_hand.probabilistic_circuit.add_weighted_edges_from([e1, e2, e3])
        self.assertEqual(len(distribution.leaves), 3)
        self.assertEqual(distribution.probabilistic_circuit, solution_by_hand.probabilistic_circuit)

    def test_deep_mount(self):
        np.random.seed(69)
        n1 = NygaDistribution(self.variable)
        data = np.random.normal(0, 1, 100).tolist()
        n2 = NygaDistribution(self.variable, min_likelihood_improvement=0.1)
        n2.fit(data)
        n1.mount(n2)
        self.assertEqual(len(n2.probabilistic_circuit.nodes), 3)


class NygaDistributionTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")
    model: NygaDistribution

    def setUp(self) -> None:
        self.model = NygaDistribution(self.x)
        self.model.add_subcircuit(UniformDistribution(self.x, portion.closed(-1.5, -0.5)), 0.5)
        self.model.add_subcircuit(UniformDistribution(self.x, portion.closed(0.5, 1.5)), 0.5)

    def test_plot(self):
        fig = go.Figure(self.model.plot())
        self.assertIsNotNone(fig)
        # fig.show()


class FittedNygaDistributionTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")
    model: NygaDistribution

    def setUp(self) -> None:
        self.model = NygaDistribution(self.x, min_likelihood_improvement=0.05)
        data = np.random.normal(0, 1, 100).tolist()
        self.model.fit(data)

    def test_plot(self):
        fig = go.Figure(self.model.plot())
        self.assertIsNotNone(fig)
        # fig.show()


if __name__ == '__main__':
    unittest.main()
