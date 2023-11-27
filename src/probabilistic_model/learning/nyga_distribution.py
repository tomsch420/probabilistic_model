import collections
import dataclasses
from typing import Optional, List, Deque, Union, Tuple

import plotly.graph_objects as go
import portion
from random_events.variables import Continuous
from typing_extensions import Self

from ..probabilistic_circuit.distributions import ContinuousDistribution, UniformDistribution
from ..probabilistic_circuit.units import DeterministicSumUnit, Unit


@dataclasses.dataclass
class InductionStep:
    data: List[float]
    """
    The entire sorted data.
    """

    weights: List[float]
    """
    The weights of the samples in dataset.
    """

    begin_index: int
    """
    Included index of the first sample.
    """

    end_index: int
    """
    Excluded index of the last sample.
    """

    nyga_distribution: 'NygaDistribution'
    """
    The Nyga Distribution to mount the quantile distributions into and read the parameters from.
    """

    @property
    def variable(self):
        """
        The variable of the distribution.
        """
        return self.nyga_distribution.variable

    @property
    def min_samples_per_quantile(self):
        """
        The minimal number of samples per quantile.
        """
        return self.nyga_distribution.min_samples_per_quantile

    @property
    def min_likelihood_improvement(self):
        """
        The relative, minimal likelihood improvement needed to create a new quantile.
        """
        return self.nyga_distribution.min_likelihood_improvement

    def left_connecting_point(self) -> float:
        """
        Calculate the left connecting point.
        """
        return self.left_connecting_point_from_index(self.begin_index)

    def left_connecting_point_from_index(self, index) -> float:
        """
        Calculate the left connecting point given some beginning index.

        :param index: The index of the left datapoint.
        """
        if index > 0:
            left_connecting_point = (self.data[index - 1] + self.data[index]) / 2
        else:
            left_connecting_point = self.data[index]
        return left_connecting_point

    def right_connecting_point(self) -> float:
        """
        Calculate the right connecting point.
        """
        return self.right_connecting_point_from_index(self.end_index)

    def right_connecting_point_from_index(self, index) -> float:
        """
        Calculate the right connecting point given some ending index.

        :param index: The index of the right datapoint.
        """
        if index < len(self.data):
            right_connecting_point = (self.data[index] + self.data[index - 1]) / 2
        else:
            right_connecting_point = self.data[index - 1]
        return right_connecting_point

    def create_uniform_distribution(self) -> UniformDistribution:
        """
        Create a uniform distribution from this induction step.
        """
        return self.create_uniform_distribution_from_indices(self.begin_index, self.end_index)

    def create_uniform_distribution_from_indices(self, begin_index: int, end_index: int) -> UniformDistribution:
        """
        Create a uniform distribution from the datapoint at `begin_index` to the datapoint at `end_index`.

        :param begin_index: The index of the first datapoint.
        :param end_index: The index of the last datapoint.
        """
        if end_index == len(self.data):
            interval = portion.closed(self.left_connecting_point_from_index(begin_index),
                                      self.right_connecting_point_from_index(end_index))
        else:
            interval = portion.closedopen(self.left_connecting_point_from_index(begin_index),
                                          self.right_connecting_point_from_index(end_index))
        return UniformDistribution(self.variable, interval)

    def sum_weights_from_indices(self, begin_index: int, end_index: int) -> float:
        """
        Sum the weights from `begin_index` to `end_index`.
        """
        return sum(self.weights[begin_index:end_index])

    def sum_weights(self):
        """
        Sum the weights of this induction step.
        """
        return self.sum_weights_from_indices(self.begin_index, self.end_index)

    def compute_best_split(self) -> Tuple[float, Optional[int]]:

        # initialize the maximum likelihood and the best split index
        maximum_likelihood = 0
        best_split_index = None

        # calculate the connecting points
        right_connecting_point = self.right_connecting_point()
        left_connecting_point = self.left_connecting_point()

        # for every possible splitting index
        for split_index in range(self.begin_index + self.min_samples_per_quantile,
                                 self.end_index - self.min_samples_per_quantile + 1):

            # calculate the split value
            split_value = (self.data[split_index - 1] + self.data[split_index]) / 2

            # calculate left likelihood
            average_likelihood_left = (self.sum_weights_from_indices(self.begin_index, split_index) /
                                       (split_value - left_connecting_point) * (split_index - self.begin_index))
            # calculate right likelihood
            average_likelihood_right = (self.sum_weights_from_indices(split_index, self.end_index) /
                                        (right_connecting_point - split_value) * (self.end_index - split_index))

            # calculate average likelihood
            average_likelihood = ((average_likelihood_left + average_likelihood_right) /
                                  (self.end_index - self.begin_index))

            # update the maximum likelihood and the best split index
            if average_likelihood > maximum_likelihood:
                maximum_likelihood = average_likelihood
                best_split_index = split_index

        return maximum_likelihood, best_split_index

    def construct_left_induction_step(self, split_index: int) -> Self:
        """
        Construct the left induction step.

        :param split_index: The index of the split.
        """
        return InductionStep(self.data, self.weights, self.begin_index, split_index, self.nyga_distribution)

    def construct_right_induction_step(self, split_index: int) -> Self:
        """
        Construct the right induction step.

        :param split_index: The index of the split.
        """
        return InductionStep(self.data, self.weights, split_index, self.end_index, self.nyga_distribution)

    def induce(self) -> List[Self]:
        """
        Perform one induction step.

        :return: The (possibly empty) list of new induction steps.
        """
        # calculate the likelihood without splitting
        average_likelihood_without_split = 1 / (self.right_connecting_point() - self.left_connecting_point())

        # calculate the best likelihood with splitting
        maximum_likelihood, best_split_index = self.compute_best_split()

        # if the improvement is good enough
        if maximum_likelihood > (1 + self.min_likelihood_improvement) * average_likelihood_without_split:

            # create the left and right induction steps
            left_induction_step = self.construct_left_induction_step(best_split_index)
            right_induction_step = self.construct_right_induction_step(best_split_index)
            return [left_induction_step, right_induction_step]

        # if the improvement is not good enough
        else:

            # mount a uniform distribution
            distribution = self.create_uniform_distribution()
            self.nyga_distribution.weights += [self.sum_weights()]
            distribution.parent = self.nyga_distribution
            return []


class NygaDistribution(DeterministicSumUnit, ContinuousDistribution):
    """
    A Nyga distribution is a deterministic mixture of uniform distributions.
    """

    min_likelihood_improvement: float
    """
    The relative, minimal likelihood improvement needed to create a new quantile.
    """

    min_samples_per_quantile: int
    """
    The minimal number of samples per quantile.
    """

    def __init__(self, variable: Continuous, min_samples_per_quantile: Optional[int] = 1,
                 min_likelihood_improvement: Optional[float] = 1.1, parent: 'Unit' = None):
        DeterministicSumUnit.__init__(self, [variable], [], parent)
        ContinuousDistribution.__init__(self, variable, parent)
        self.min_samples_per_quantile = min_samples_per_quantile
        self.min_likelihood_improvement = min_likelihood_improvement

    def _pdf(self, value: Union[float, int]) -> float:
        return sum([child._pdf(value) * weight for child, weight in zip(self.children, self.weights)])

    def _cdf(self, value: Union[float, int]) -> float:
        return sum([child._cdf(value) * weight for child, weight in zip(self.children, self.weights)])

    def fit(self, data: List[float]) -> Self:
        """
        Fit the distribution to the data.

        :param data: The data to fit the distribution to.

        :return: The fitted distribution.
        """

        # sort the data and calculate the weights
        sorted_data = sorted(set(data))
        weights = [data.count(value) / len(data) for value in sorted_data]

        # construct the initial induction step
        initial_induction_step = InductionStep(data=sorted_data, weights=weights, begin_index=0,
                                               end_index=len(sorted_data), nyga_distribution=self)
        # initialize the queue
        induction_steps: Deque[InductionStep] = collections.deque([initial_induction_step])

        # induce the distribution
        while len(induction_steps) > 0:
            induction_step = induction_steps.popleft()
            new_induction_steps = induction_step.induce()
            induction_steps.extend(new_induction_steps)

        return self

    def pdf_trace(self) -> go.Scatter:
        """
        Create a plotly trace for the probability density function.
        """
        domain_size = self.domain[self.variable].upper - self.domain[self.variable].lower
        x = [self.domain[self.variable].lower - domain_size * 0.05, self.domain[self.variable].lower, None]
        y = [0, 0, None]
        for uniform in self.leaves:
            lower_value = uniform.interval.lower
            upper_value = uniform.interval.upper
            pdf_value = uniform.pdf_value() * uniform.get_weight_if_possible()
            x += [lower_value, upper_value, None]
            y += [pdf_value, pdf_value, None]

        x.extend([self.domain[self.variable].upper, self.domain[self.variable].upper + domain_size * 0.05])
        y.extend([0, 0])
        return go.Scatter(x=x, y=y, mode='lines', name="Probability Density Function")

    def cdf_trace(self) -> go.Scatter:
        """
        Create a plotly trace for the cumulative distribution function.
        """
        domain_size = self.domain[self.variable].upper - self.domain[self.variable].lower
        x = [self.domain[self.variable].lower - domain_size * 0.05, self.domain[self.variable].lower, None]
        y = [0, 0, None]
        for uniform in self.leaves:
            lower_value = uniform.interval.lower
            upper_value = uniform.interval.upper
            x += [lower_value, upper_value, None]
            y += [self.cdf(lower_value), self.cdf(upper_value), None]

        x.extend([self.domain[self.variable].upper, self.domain[self.variable].upper + domain_size * 0.05])
        y.extend([1, 1])
        return go.Scatter(x=x, y=y, mode='lines', name="Cumulative Distribution Function")

    def plot(self) -> go.Figure:
        """
        Plot the distribution with PDF, CDF, Expectation and Mode.
        """
        figure = go.Figure(data=[self.pdf_trace(), self.cdf_trace()])

        mode, maximum_likelihood = self.mode()
        mode = mode[0][self.variable]

        expectation = self.expectation([self.variable])[self.variable]
        figure.add_trace(go.Scatter(x=[mode.lower, mode.upper, mode.upper, mode.lower],
                                    y=[0, maximum_likelihood * 1.05, maximum_likelihood * 1.05, 0],
                                    mode='lines+markers',name="Mode", fill="toself"))
        figure.add_trace(go.Scatter(x=[expectation, expectation], y=[0, maximum_likelihood * 1.05],
                                    mode='lines+markers', name="Expectation"))

        figure.update_layout(title=f'Nyga Distribution of {self.variable.name}',
                             xaxis_title=self.variable.name)
        return figure