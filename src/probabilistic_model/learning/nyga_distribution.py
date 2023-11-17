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
        ...

    def compute_most_likely_split(self, data: List[float]):
        maximum_likelihood = 0
        best_sum_node = None

        for index in range(1, len(data) - 1):
            distribution = self._create_deterministic_uniform_mixture_from_datasets(data[:index], data[index:])
            average_likelihood = sum([distribution.likelihood([value]) for value in data]) / len(data)
            if average_likelihood > maximum_likelihood:
                maximum_likelihood = average_likelihood
                best_sum_node = distribution

        return maximum_likelihood, best_sum_node

    def _create_deterministic_uniform_mixture_from_datasets(self, left_data: List[float],
                                                            right_data: List[float]) -> DeterministicSumUnit:

        # creat uniform distribution from the left including to the right excluding
        left_uniform_distribution = UniformDistribution(self.variable, portion.closedopen(left_data[0], right_data[0]))
        right_uniform_distribution = UniformDistribution(self.variable, portion.closed(right_data[0], right_data[-1]))

        datapoints_total = len(left_data) + len(right_data)

        result = DeterministicSumUnit(self.variables,
                                      [len(left_data) / datapoints_total, len(right_data) / datapoints_total])
        result.children = [left_uniform_distribution, right_uniform_distribution]
        return result
