import unittest
from typing import List

import numpy as np
import portion
from anytree import RenderTree
from random_events.variables import Continuous

from probabilistic_model.learning.nyga_distribution import NygaDistribution, InductionParameters
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution
from probabilistic_model.probabilistic_circuit.units import DecomposableProductUnit, DeterministicSumUnit


class InductionParametersTestCase(unittest.TestCase):
    variable: Continuous = Continuous("x")
    sorted_data: List[float] = [1, 2, 3, 4, 7, 9]
    induction_parameters: InductionParameters

    def setUp(self) -> None:
        self.induction_parameters = InductionParameters(0, len(self.sorted_data), 0,
                                                        DeterministicSumUnit([self.variable], []), 0)

    def test_left_connecting_point(self):
        left_connecting_point = self.induction_parameters.left_connecting_point(self.sorted_data)
        self.assertEqual(left_connecting_point, 1)
        self.induction_parameters.begin_index = 3
        self.assertEqual(self.induction_parameters.left_connecting_point(self.sorted_data), 3.5)

    def test_right_connecting_point(self):
        right_connecting_point = self.induction_parameters.right_connecting_point(self.sorted_data)
        self.assertEqual(right_connecting_point, 9)
        self.induction_parameters.end_index = 5
        self.assertEqual(self.induction_parameters.right_connecting_point(self.sorted_data), 8)

    def test_total_number_of_samples(self):
        self.assertEqual(self.induction_parameters.total_number_of_samples(), 6)
        self.induction_parameters.end_index = 5
        self.assertEqual(self.induction_parameters.total_number_of_samples(), 5)

    def test_get_from_sorted_data(self):
        self.assertEqual(self.induction_parameters.get_from_sorted_data(self.sorted_data), self.sorted_data)
        self.induction_parameters.begin_index = 2
        self.induction_parameters.end_index = 5
        self.assertEqual(self.induction_parameters.get_from_sorted_data(self.sorted_data), [3, 4, 7])

    def test_construct_left_induction_parameters(self):
        left_induction_parameters = self.induction_parameters.construct_left_induction_parameters(3, 10)
        self.assertEqual(left_induction_parameters.begin_index, 0)
        self.assertEqual(left_induction_parameters.end_index, 3)
        self.assertEqual(left_induction_parameters.previous_average_likelihood, 10)
        self.assertEqual(self.induction_parameters.current_node.weights[0], 1/2)
        self.assertEqual(left_induction_parameters.current_node.parent.weights[0], 1/2)
        self.assertEqual(left_induction_parameters.get_from_sorted_data(self.sorted_data), [1, 2, 3])

    def test_construct_right_induction_parameters(self):
        right_induction_parameters = self.induction_parameters.construct_right_induction_parameters(3, 10)
        self.assertEqual(right_induction_parameters.begin_index, 3)
        self.assertEqual(right_induction_parameters.end_index, 6)
        self.assertEqual(right_induction_parameters.previous_average_likelihood, 10)
        self.assertEqual(self.induction_parameters.current_node.weights[0], 1/2)
        self.assertEqual(right_induction_parameters.current_node.parent.weights[0], 1/2)
        self.assertEqual(right_induction_parameters.get_from_sorted_data(self.sorted_data), [4, 7, 9])

    def test_construct_left_and_right(self):
        left_induction_parameters = self.induction_parameters.construct_left_induction_parameters(2, 10)
        right_induction_parameters = self.induction_parameters.construct_right_induction_parameters(2, 10)
        self.assertEqual(self.induction_parameters.current_node.children, (left_induction_parameters.current_node,
                         right_induction_parameters.current_node))
        self.assertEqual(self.induction_parameters.current_node.weights, [2/6, 4/6])


class NygaDistributionTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")
    data: np.ndarray = np.concatenate((np.random.normal(0, 1, 100), np.random.normal(5, 2, 100)))

    def test_no_duplicate_parent(self):
        y = Continuous("y")
        product = DecomposableProductUnit([self.x, y])

        distribution_x = NygaDistribution(self.x, parent=product)
        distribution_y = NygaDistribution(y, parent=product)

        self.assertEqual(len(product.children), 2)

    def test_fit_with_known_result(self):
        distribution = NygaDistribution(self.x)
        data = [1, 4, 2]
        result = distribution.fit(data)
        self.assertTrue(result.is_deterministic())
        self.assertTrue(all([result.likelihood([value]) > 0 for value in data]))
        self.assertEqual(len(result.leaves), 3)

    def test_create_deterministic_uniform_mixture_from_datasets_without_connecting_points(self):
        left_dataset = [1, 2, 3]
        right_dataset = [4, 7, 9]
        distribution = NygaDistribution(self.x)
        dsu = distribution._create_deterministic_uniform_mixture_from_datasets(left_dataset, right_dataset, 1, 9)
        self.assertEqual(dsu.children[0], UniformDistribution(self.x, portion.closedopen(1, 3.5)))
        self.assertEqual(dsu.children[1], UniformDistribution(self.x, portion.closed(3.5, 9)))
        self.assertEqual(dsu.weights, [3 / 6, 3 / 6])

    def test_create_deterministic_uniform_mixture_from_datasets_with_connecting_points(self):
        left_dataset = [1, 2, 3]
        right_dataset = [4, 7, 9]
        distribution = NygaDistribution(self.x)
        dsu = distribution._create_deterministic_uniform_mixture_from_datasets(left_dataset, right_dataset, 0, 10)
        self.assertEqual(dsu.children[0], UniformDistribution(self.x, portion.closedopen(0, 3.5)))
        self.assertEqual(dsu.children[1], UniformDistribution(self.x, portion.closed(3.5, 10)))
        self.assertEqual(dsu.weights, [3 / 6, 3 / 6])

    def test_compute_best_split(self):
        dataset = [1, 2, 3, 4, 7, 9]
        distribution = NygaDistribution(self.x)
        maximum_likelihood, best_sum_node, split_index = distribution.compute_most_likely_split(dataset, 1, 9)
        self.assertEqual(split_index, 1)
        self.assertEqual(len(best_sum_node.children), 2)
        self.assertEqual(best_sum_node.children[0], UniformDistribution(self.x, portion.closedopen(1, 1.5)))

    def test_nyga_distribution(self):
        dataset = [1, 2, 3, 4, 7, 9]
        distribution = NygaDistribution(self.x, min_likelihood_improvement=1.01)
        result = distribution.fit(dataset)
        print(result)


if __name__ == '__main__':
    unittest.main()
