import unittest

import numpy as np
import portion
from random_events.variables import Continuous

from probabilistic_model.learning.nyga_distribution import NygaDistribution
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution
from probabilistic_model.probabilistic_circuit.units import DecomposableProductUnit


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
        distribution.fit([1, 4, 2])

    def test_create_deterministic_uniform_mixture_from_datasets(self):
        left_dataset = [1, 2, 3]
        right_dataset = [4, 7, 9]
        distribution = NygaDistribution(self.x)
        dsu = distribution._create_deterministic_uniform_mixture_from_datasets(left_dataset, right_dataset)
        self.assertEqual(dsu.children[0], UniformDistribution(self.x, portion.closedopen(1, 4)))
        self.assertEqual(dsu.children[1], UniformDistribution(self.x, portion.closed(4, 9)))

    def test_compute_best_split(self):
        dataset = [1, 2, 3, 4, 7, 9]
        distribution = NygaDistribution(self.x)
        maximum_likelihood, best_sum_node = distribution.compute_most_likely_split(dataset)
        print(best_sum_node)
        print(maximum_likelihood)

if __name__ == '__main__':
    unittest.main()
