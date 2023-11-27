import unittest
from typing import List

import numpy as np
import portion
from anytree import RenderTree
from random_events.variables import Continuous

from probabilistic_model.learning.nyga_distribution import NygaDistribution, InductionStep
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution


class InductionStepTestCase(unittest.TestCase):
    variable: Continuous = Continuous("x")
    sorted_data: List[float] = [1, 2, 3, 4, 7, 9]
    weights: List[float] = [1 / 6] * 6
    induction_step: InductionStep

    def setUp(self) -> None:
        self.induction_step = InductionStep(self.sorted_data, self.weights, 0, len(self.sorted_data),
                                            NygaDistribution(self.variable, min_samples_per_quantile=1,
                                                             min_likelihood_improvement=0.01))

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
        self.assertEqual(distribution, UniformDistribution(self.variable, portion.closedopen(3.5, 8)))

    def test_sum_weights(self):
        self.assertAlmostEqual(self.induction_step.sum_weights(), 1)

    def test_sum_weights_from_indices(self):
        self.assertAlmostEqual(self.induction_step.sum_weights_from_indices(3, 5), 1 / 3)

    def test_compute_best_split(self):
        maximum, index = self.induction_step.compute_best_split()
        self.assertEqual(index, 1)
        self.assertAlmostEqual(maximum, 0.148158, delta=0.001)

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
        distribution = NygaDistribution(self.variable, min_likelihood_improvement=0.01)
        distribution.fit(data)
        self.assertAlmostEqual(sum([leaf.get_weight_if_possible() for leaf in distribution.leaves]), 1.)

    def test_plot(self):
        np.random.seed(69)
        data = np.random.normal(0, 1, 100).tolist()
        distribution = NygaDistribution(self.variable, min_likelihood_improvement=-1)
        distribution.fit(data)
        distribution.plot()  # .show()

if __name__ == '__main__':
    unittest.main()