import collections
from typing import Optional, List

import portion
from random_events.variables import Continuous
from typing_extensions import Self

from ..probabilistic_circuit.distributions import ContinuousDistribution, UniformDistribution
from ..probabilistic_circuit.units import DeterministicSumUnit, Unit

epsilon = pow(10, -10)


class NygaDistribution(ContinuousDistribution, DeterministicSumUnit):
    """
    A Nyga distribution is a deterministic mixture of uniform distributions.
    """

    min_likelihood_improvement: float = 1.1
    """
    The relative, minimal likelihood improvement needed to create a new quantile.
    """

    min_samples_per_quantile: int = 1
    """
    The minimal number of samples per quantile.
    """

    def __init__(self, variable: Continuous, min_likelihood_improvement: Optional[float] = None, parent: 'Unit' = None):
        super().__init__(variable, None)
        DeterministicSumUnit.__init__(self, self.variables, [], parent)

        if min_likelihood_improvement is not None:
            self.min_likelihood_improvement = min_likelihood_improvement

    def fit(self, data: List[float]) -> Self:
        """
        Fit the model to the data.
        :param data:
        :return:
        """
        # sort the data
        sorted_data = list(sorted(data))

        minimal_distribution = UniformDistribution(self.variable, portion.closed(sorted_data[0], sorted_data[-1]))
        minimal_average_likelihood = (sum([minimal_distribution.likelihood([value]) for value in sorted_data])
                                      / len(sorted_data))

        compute_most_likely_split_parameters = collections.deque()
        compute_most_likely_split_parameters.append((sorted_data, 0, len(data)))

        while len(compute_most_likely_split_parameters) > 0:

            parameters = compute_most_likely_split_parameters.pop()

            # calculate the best possible split
            new_maximum_likelihood, best_sum_node, split_index = self.compute_most_likely_split(sorted_data)

            # if no further splits could be made
            if split_index is None:
                break

            # if the improvement is not large enough
            if new_maximum_likelihood <= minimal_average_likelihood * self.min_likelihood_improvement:
                break

        return self

    def compute_most_likely_split(self, data: List[float]):
        maximum_likelihood = 0
        best_sum_node = None
        index = None
        for index in range(self.min_samples_per_quantile, len(data) - self.min_samples_per_quantile):

            distribution = self._create_deterministic_uniform_mixture_from_datasets(data[:index], data[index:])
            average_likelihood = sum([distribution.likelihood([value]) for value in data]) / len(data)
            if average_likelihood > maximum_likelihood:
                maximum_likelihood = average_likelihood
                best_sum_node = distribution

        return maximum_likelihood, best_sum_node, index

    def _create_deterministic_uniform_mixture_from_datasets(self, left_data: List[float],
                                                            right_data: List[float]) -> DeterministicSumUnit:
        """
        Create a deterministic uniform mixture from two datasets.
        The left dataset is included in the left uniform distribution up to the middle point between the last
        point in the left dataset and the first point in the right dataset.
        The right dataset is included in the right uniform distribution from the middle point.
        The weights of the mixture correspond to the relative size of the datasets.

        :param left_data: The data for the left distribution.
        :param right_data: The data for the right distribution.
        :return: A deterministic uniform mixture of the two datasets.
        """

        connecting_point = (left_data[-1] + right_data[0]) / 2

        # creat uniform distribution from the left including to the right excluding
        left_uniform_distribution = UniformDistribution(self.variable,
                                                        portion.closedopen(left_data[0], connecting_point))
        right_uniform_distribution = UniformDistribution(self.variable,
                                                         portion.closed(connecting_point, right_data[-1]))

        datapoints_total = len(left_data) + len(right_data)

        result = DeterministicSumUnit(self.variables,
                                      [len(left_data) / datapoints_total, len(right_data) / datapoints_total])
        result.children = [left_uniform_distribution, right_uniform_distribution]
        return result
