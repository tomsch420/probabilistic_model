import unittest
from typing import List

import portion
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
                                            NygaDistribution(self.variable, 1.1), 1 / 9, 0)

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

    def test_create_deterministic_uniform_mixture_from_split_index(self):
        distribution = self.induction_step.create_deterministic_uniform_mixture_from_split_index(3)
        self.assertEqual(distribution.children[0], UniformDistribution(self.variable, portion.closedopen(1, 3.5)))
        self.assertEqual(distribution.children[1], UniformDistribution(self.variable, portion.closed(3.5, 9)))
        self.assertAlmostEqual(distribution.weights[0], 1 / 2)
        self.assertAlmostEqual(distribution.weights[1], 1 / 2)

    def test_compute_best_split(self):
        maximum, index = self.induction_step.compute_best_split()
        self.assertEqual(index, 1)
        self.assertAlmostEqual(maximum, 0.148158, delta=0.001)

    def test_construct_left_induction_step(self):
        induction_step = self.induction_step.construct_left_induction_step(1)
        self.assertEqual(induction_step.begin_index, 0)
        self.assertEqual(induction_step.end_index, 1)
        self.assertEqual(induction_step.data, self.induction_step.data)
        self.assertEqual(induction_step.weights, self.induction_step.weights)
        self.assertEqual(induction_step.current_node.parent, self.induction_step.current_node)

    def test_construct_right_induction_step(self):
        induction_step = self.induction_step.construct_right_induction_step(1)
        self.assertEqual(induction_step.begin_index, 1)
        self.assertEqual(induction_step.end_index, 6)
        self.assertEqual(induction_step.data, self.induction_step.data)
        self.assertEqual(induction_step.weights, self.induction_step.weights)
        self.assertEqual(induction_step.current_node.parent, self.induction_step.current_node)

    def test_construct_induction_step_left_and_right(self):
        lef_induction_step = self.induction_step.construct_left_induction_step(1)
        right_induction_step = self.induction_step.construct_right_induction_step(1)
        self.assertEqual(self.induction_step.current_node.children[0], lef_induction_step.current_node)
        self.assertEqual(self.induction_step.current_node.children[1], right_induction_step.current_node)
        self.assertAlmostEqual(self.induction_step.current_node.weights[0], 1 / 6)
        self.assertAlmostEqual(self.induction_step.current_node.weights[1], 5 / 6)


if __name__ == '__main__':
    unittest.main()
