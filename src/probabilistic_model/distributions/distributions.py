import random

import portion
from random_events.events import EncodedEvent, Event
from random_events.variables import Variable, Continuous, Discrete, Symbolic, Integer
from typing_extensions import Union, Iterable, Any, Self, Dict, List, Tuple
import plotly.graph_objects as go


from ..probabilistic_model import ProbabilisticModel


class UnivariateDistribution(ProbabilisticModel):
    """
    Abstract Base class for Univariate distributions.
    """

    def __init__(self, variable: Variable):
        super().__init__([variable])

    @property
    def variable(self) -> Variable:
        """
        The variable of this distribution.
        """
        return self.variables[0]

    def _pdf(self, value: Union[float, int]) -> float:
        """
        Evaluate the probability density function at the encoded `value`.
        :param value: The encoded value to evaluate the pdf on.
        :return: The density
        """
        raise NotImplementedError

    def _likelihood(self, event: Iterable) -> float:
        return self._pdf(list(event)[0])

    def pdf(self, value: Any) -> float:
        """
        Evaluate the probability density function at `value`.
        :param value: The value to evaluate the pdf on.
        :return: The density
        """
        return self._pdf(self.variable.encode(value))

    def likelihood(self, event: Iterable) -> float:
        return self.pdf(list(event)[0])

    def plot(self) -> List:
        """
        Generate a list of traces that can be used to plot the distribution in plotly figures.
        """
        raise NotImplementedError


class ContinuousDistribution(UnivariateDistribution):
    """
    Abstract base class for continuous distributions.
    """

    variables: Tuple[Continuous]

    @property
    def variable(self) -> Continuous:
        return self.variables[0]

    def _cdf(self, value: float) -> float:
        """
        Evaluate the cumulative distribution function at the encoded `value`.
        :param value: The encoded value to evaluate the cdf on.
        :return: The cumulative probability.
        """
        raise NotImplementedError

    def cdf(self, value: Any):
        """
        Evaluate the cumulative distribution function at `value`.
        :param value: The value to evaluate the cdf on.
        :return: The cumulative probability.
        """
        if value <= -float("inf"):
            return 0
        if value >= float("inf"):
            return 1
        return self._cdf(self.variable.encode(value))

    def _probability(self, event: EncodedEvent) -> float:
        interval: portion.Interval = event[self.variable]
        probability = 0.

        for interval_ in interval:
            probability += self.cdf(interval_.upper) - self.cdf(interval_.lower)

        return probability


class DiscreteDistribution(UnivariateDistribution):
    """
    Abstract base class for univariate discrete distributions.
    """
    variables: Tuple[Discrete]

    weights: List[float]
    """
    The probability of each value in the domain of this distributions variable.
    """

    def __init__(self, variable: Discrete, weights: Iterable[float]):
        super().__init__(variable)
        self.weights = list(weights)

        if len(self.weights) != len(self.variable.domain):
            raise ValueError("The number of weights has to be equal to the number of values of the variable.")

    @property
    def domain(self) -> Event:
        return Event(
            {self.variable: [value for value, weight in zip(self.variable.domain, self.weights) if weight > 0]})

    @property
    def variable(self) -> Discrete:
        return self.variables[0]

    def _pdf(self, value: int) -> float:
        """
        Calculate the probability of a value.
        :param value: The index of the value to calculate the probability of.
        :return: The probability of the value.
        """
        return self.weights[value]

    def _probability(self, event: EncodedEvent) -> float:
        return sum(self._pdf(value) for value in event[self.variable])

    def _mode(self) -> Tuple[List[EncodedEvent], float]:
        maximum_weight = max(self.weights)
        mode = EncodedEvent(
            {self.variable: [index for index, weight in enumerate(self.weights) if weight == maximum_weight]})

        return [mode], maximum_weight

    def _conditional(self, event: EncodedEvent) -> Tuple[Self, float]:
        unnormalized_weights = [weight if index in event[self.variable] else 0. for index, weight in
                                enumerate(self.weights)]
        probability = sum(unnormalized_weights)

        if probability == 0:
            return None, 0

        normalized_weights = [weight / probability for weight in unnormalized_weights]
        return self.__class__(self.variable, normalized_weights), probability

    def sample(self, amount: int) -> Iterable:
        return [random.choices(self.variable.domain, self.weights) for _ in range(amount)]

    def __copy__(self):
        return self.__class__(self.variable, self.weights)

    def __eq__(self, other):
        return (isinstance(other, DiscreteDistribution) and self.weights == other.weights and
                super().__eq__(other))

    def _fit(self, data: List[int]) -> Self:
        """
        Fit the distribution to a list of encoded values

        :param data: The encoded values
        :return: The fitted distribution
        """
        weights = []
        for value in range(len(self.variable.domain)):
            weights.append(data.count(value) / len(data))
        self.weights = weights
        return self

    def fit(self, data: Iterable[Any]) -> Self:
        """
        Fit the distribution to a list of raw values.

        :param data: The not processed data.
        :return: The fitted distribution
        """
        return self._fit(list(self.variable.encode_many(data)))

    def plot(self) -> List[go.Bar]:
        """
        Plot the distribution.
        """

        mode, likelihood = self.mode()
        mode = mode[0][self.variable]

        traces = list()
        traces.append(go.Bar(x=[value for value in self.variable.domain if value not in mode], y=self.weights,
                             name="Probability"))
        traces.append(go.Bar(x=mode, y=[likelihood] * len(mode), name="Mode"))
        return traces


class SymbolicDistribution(DiscreteDistribution):
    """
    Class for symbolic (categorical) distributions.
    """

    variables: Tuple[Symbolic]

    @property
    def variable(self) -> Symbolic:
        return self.variables[0]

    @property
    def representation(self):
        return f"Nominal{self.variable.domain}"


class IntegerDistribution(DiscreteDistribution, ContinuousDistribution):
    """
    Abstract base class for integer distributions. Integer distributions also implement the methods of continuous
    distributions.
    """
    variables: Tuple[Integer]

    @property
    def variable(self) -> Integer:
        return self.variables[0]

    @property
    def representation(self):
        return f"Ordinal{self.variable.domain}"

    def _cdf(self, value: int) -> float:
        """
        Calculate the cumulative distribution function at `value`.
        :param value: The value to evaluate the cdf on.
        :return: The cumulative probability.
        """
        return sum(self._pdf(value) for value in range(value))

    def plot(self) -> List[Union[go.Bar, go.Scatter]]:
        traces = super().plot()
        _, likelihood = self.mode()
        expectation = self.expectation([self.variable])[self.variable]
        traces.append(go.Scatter(x=[expectation, expectation], y=[0, likelihood * 1.05], mode="lines+markers",
                                 name="Expectation"))
        return traces