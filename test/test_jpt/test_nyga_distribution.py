import unittest

import numpy as np
import plotly.graph_objects as go
from numpy import testing
from random_events.interval import closed, closed_open
from random_events.product_algebra import Event
from random_events.variable import Continuous

from probabilistic_model.learning.nyga_distribution import NygaDistribution, InductionStep
from probabilistic_model.probabilistic_circuit.distributions import DiracDeltaDistribution
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit, SumUnit
from probabilistic_model.utils import SubclassJSONSerializer


class InductionStepTestCase(unittest.TestCase):
    variable: Continuous = Continuous("x")
    sorted_data: np.array = np.array([1, 2, 3, 4, 7, 9])
    weights: np.array = np.ones((len(sorted_data),))
    induction_step: InductionStep

    def setUp(self) -> None:
        nyga_distribution = NygaDistribution(self.variable, min_samples_per_quantile=1, min_likelihood_improvement=0.01)
        cumulative_log_weights = np.cumsum(np.log(self.weights))
        cumulative_log_weights = np.append(0, cumulative_log_weights)
        cumulative_weights = np.cumsum(self.weights)
        cumulative_weights = np.append(0, cumulative_weights, )
        self.induction_step = InductionStep(self.sorted_data, cumulative_weights, cumulative_log_weights, 0,
                                            len(self.sorted_data), nyga_distribution)

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
        self.assertEqual(distribution, UniformDistribution(self.variable, closed(1, 9).simple_sets[0]))

    def test_create_uniform_distribution(self):
        distribution = self.induction_step.create_uniform_distribution_from_indices(3, 5)
        self.assertEqual(distribution, UniformDistribution(self.variable, closed_open(3.5, 8.0).simple_sets[0]))

    def test_sum_weights(self):
        self.assertAlmostEqual(self.induction_step.sum_weights(), 6)

    def test_sum_weights_from_indices(self):
        self.assertAlmostEqual(self.induction_step.sum_weights_from_indices(3, 5), 2)

    def test_likelihood_of_split(self):
        """
        Test that the calculation of the likelihood of a split is correct and as it is in the notebook.
        """

        likelihood_without_split = self.induction_step.log_likelihood_without_split()
        self.assertAlmostEqual(likelihood_without_split, -12.48, delta=0.01)

        # k = 1
        likelihood_of_split_left = self.induction_step.log_likelihood_of_split_side(1, 1)
        self.assertAlmostEqual(likelihood_of_split_left, -1.1, delta=0.01)
        likelihood_of_split_right = self.induction_step.log_likelihood_of_split_side(1, 9)
        self.assertAlmostEqual(likelihood_of_split_right, -10.99, delta=0.01)

        # k = 2
        likelihood_of_split_left = self.induction_step.log_likelihood_of_split_side(2, 1)
        self.assertAlmostEqual(likelihood_of_split_left, -3.01, delta=0.01)
        likelihood_of_split_right = self.induction_step.log_likelihood_of_split_side(2, 9)
        self.assertAlmostEqual(likelihood_of_split_right, -9.11, delta=0.01)

        # k = 3
        likelihood_of_split_left = self.induction_step.log_likelihood_of_split_side(3, 1)
        self.assertAlmostEqual(likelihood_of_split_left, -4.83, delta=0.01)
        likelihood_of_split_right = self.induction_step.log_likelihood_of_split_side(3, 9)
        self.assertAlmostEqual(likelihood_of_split_right, -7.19, delta=0.01)

        # k = 4
        likelihood_of_split_left = self.induction_step.log_likelihood_of_split_side(4, 1)
        self.assertAlmostEqual(likelihood_of_split_left, -7.64, delta=0.01)
        likelihood_of_split_right = self.induction_step.log_likelihood_of_split_side(4, 9)
        self.assertAlmostEqual(likelihood_of_split_right, -4.7, delta=0.01)

        # k = 5
        likelihood_of_split_left = self.induction_step.log_likelihood_of_split_side(5, 1)
        self.assertAlmostEqual(likelihood_of_split_left, -10.64, delta=0.01)
        likelihood_of_split_right = self.induction_step.log_likelihood_of_split_side(5, 9)
        self.assertAlmostEqual(likelihood_of_split_right, -1.79, delta=0.01)

    def test_compute_best_split(self):
        maximum, index = self.induction_step.compute_best_split()
        self.assertEqual(index, 3)

    def test_compute_best_split_without_result(self):
        self.induction_step.nyga_distribution.min_samples_per_quantile = 4
        maximum, index = self.induction_step.compute_best_split()
        self.assertEqual(index, None)
        self.assertEqual(maximum, -float("inf"))

    def test_compute_best_split_with_induced_indices(self):
        self.induction_step.begin_index = 3
        maximum, index = self.induction_step.compute_best_split()
        self.assertEqual(index, 5)

    def test_construct_left_induction_step(self):
        induction_step = self.induction_step.construct_left_induction_step(1)
        self.assertEqual(induction_step.begin_index, 0)
        self.assertEqual(induction_step.end_index, 1)
        testing.assert_equal(induction_step.data, self.induction_step.data)
        testing.assert_equal(induction_step.cumulative_log_weights, self.induction_step.cumulative_log_weights)

    def test_construct_right_induction_step(self):
        induction_step = self.induction_step.construct_right_induction_step(1)
        self.assertEqual(induction_step.begin_index, 1)
        self.assertEqual(induction_step.end_index, 6)
        testing.assert_equal(induction_step.data, self.induction_step.data)
        testing.assert_equal(induction_step.cumulative_log_weights, self.induction_step.cumulative_log_weights)

    def test_fit(self):
        np.random.seed(69)
        self.induction_step.nyga_distribution.min_samples_per_quantile = 20
        self.induction_step.nyga_distribution.min_likelihood_improvement = 0
        data = np.random.normal(0, 1, 500).tolist()
        distribution = self.induction_step.nyga_distribution
        distribution.fit(data)
        self.assertLessEqual(len(distribution.subcircuits),
                             int(len(data) / self.induction_step.nyga_distribution.min_samples_per_quantile))
        self.assertAlmostEqual(sum(distribution.weights), 1.)

    def test_domain(self):
        np.random.seed(69)
        data = np.random.normal(0, 1, 100).tolist()
        distribution = self.induction_step.nyga_distribution
        distribution.fit(data)
        domain = distribution.support()
        self.assertEqual(len(domain.simple_sets), 1)
        self.assertEqual(domain.simple_sets[0][self.variable], closed(min(data), max(data)))

    def test_plot(self):
        np.random.seed(69)
        data = np.random.normal(0, 1, 100)
        distribution = self.induction_step.nyga_distribution
        distribution.fit(data)
        fig = go.Figure(distribution.plot())
        self.assertIsNotNone(fig)
        # fig.show()

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
        u1 = UniformDistribution(self.variable, closed(0, 5).simple_sets[0])
        u2 = UniformDistribution(self.variable, closed(2, 3).simple_sets[0])
        sum_unit = SumUnit()
        e1 = (sum_unit, u1, 0.5)
        e2 = (sum_unit, u2, 0.5)
        sum_unit.probabilistic_circuit.add_weighted_edges_from([e1, e2])
        distribution = NygaDistribution.from_uniform_mixture(sum_unit)

        solution_by_hand = NygaDistribution(self.variable)
        solution_by_hand.probabilistic_circuit = ProbabilisticCircuit()
        leaf_1 = UniformDistribution(self.variable, closed_open(0, 2).simple_sets[0])
        leaf_2 = UniformDistribution(self.variable, closed_open(2, 3).simple_sets[0])
        leaf_3 = UniformDistribution(self.variable, closed(3, 5).simple_sets[0])

        e1 = (solution_by_hand, leaf_1, 0.2)
        e2 = (solution_by_hand, leaf_2, 0.6)
        e3 = (solution_by_hand, leaf_3, 0.2)

        solution_by_hand.probabilistic_circuit.add_weighted_edges_from([e1, e2, e3])
        self.assertEqual(len(distribution.leaves), 3)
        self.assertEqual(distribution.probabilistic_circuit, solution_by_hand.probabilistic_circuit)

    def test_deep_mount(self):
        np.random.seed(69)
        n1 = NygaDistribution(self.variable)
        data = np.random.normal(0, 1, 100)
        n2 = NygaDistribution(self.variable, min_likelihood_improvement=1.1, min_samples_per_quantile=40)
        n2.fit(data)
        n1.mount(n2)
        self.assertEqual(len(n1.probabilistic_circuit.nodes), 4)


class NygaDistributionTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")
    model: NygaDistribution

    def setUp(self) -> None:
        self.model = NygaDistribution(self.x)
        self.model.add_subcircuit(UniformDistribution(self.x, closed(-1.5, -0.5).simple_sets[0]), 0.5)
        self.model.add_subcircuit(UniformDistribution(self.x, closed(0.5, 1.5).simple_sets[0]), 0.5)

    def test_plot(self):
        fig = go.Figure(self.model.plot())
        self.assertIsNotNone(fig)
        # fig.show()


class FittedNygaDistributionTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")
    model: NygaDistribution

    def setUp(self) -> None:
        self.model = NygaDistribution(self.x, min_likelihood_improvement=0.001, min_samples_per_quantile=300)
        data = np.random.normal(0, 1, 1000).tolist()
        self.model.fit(data)

    def test_plot(self):
        self.model.support()
        fig = go.Figure(self.model.plot())
        self.assertIsNotNone(fig)


if __name__ == '__main__':
    unittest.main()
