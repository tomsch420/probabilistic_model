import copy
import random
from typing import Iterable, Tuple, Union, List, Optional, Any
from typing_extensions import Self

from probabilistic_model.probabilistic_circuit.units import Unit, DeterministicSumUnit
from random_events.events import EncodedEvent, VariableMap
from random_events.variables import Variable, Continuous, Symbolic, Integer, Discrete
import portion


class UnivariateDistribution(Unit):

    def __init__(self, variable: Variable, parent: 'Unit' = None):
        super().__init__([variable], parent)

    @property
    def variable(self) -> Variable:
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

    def _parameter_copy(self):
        return copy.copy(self)


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
        return self._cdf(self.variable.encode(value))


class UnivariateDiscreteDistribution(UnivariateDistribution):
    """
    Abstract base class for univariate discrete distributions.
    """
    variables: Tuple[Discrete]

    weights: List[float]
    """
    The probability of each value.
    """

    def __init__(self, variable: Discrete, weights: Iterable[float], parent=None):
        super().__init__(variable, parent)
        self.weights = list(weights)

        if len(self.weights) != len(self.variable.domain):
            raise ValueError("The number of weights has to be equal to the number of values of the variable.")

    def __repr__(self):
        return f"Categorical()"

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
        mode = EncodedEvent({self.variable: [index for index, weight in enumerate(self.weights)
                                             if weight == maximum_weight]})

        return [mode], maximum_weight

    def _conditional(self, event: EncodedEvent) -> Tuple[Self, float]:
        unnormalized_weights = [weight if index in event[self.variable] else 0. for index, weight in
                                enumerate(self.weights)]
        probability = sum(unnormalized_weights)

        if probability == 0:
            return None, 0

        normalized_weights = [weight / probability for weight in unnormalized_weights]
        return self.__class__(self.variable, normalized_weights), probability

    def __eq__(self, other):
        if not isinstance(other, SymbolicDistribution):
            return False
        return self.variable == other.variable and self.weights == other.weights

    def sample(self, amount: int) -> Iterable:
        return [random.choices(self.variable.domain, self.weights) for _ in range(amount)]

    def __copy__(self):
        return self.__class__(self.variable, self.weights)


class SymbolicDistribution(UnivariateDiscreteDistribution):
    """
    Class for symbolic (categorical) distributions.
    """

    variables: Tuple[Symbolic]

    @property
    def variable(self) -> Symbolic:
        return self.variables[0]

    def moment(self, order: VariableMap[Union[Integer, Continuous], int],
               center: VariableMap[Union[Integer, Continuous], float]) \
            -> VariableMap[Union[Integer, Continuous], float]:
        return VariableMap()


class IntegerDistribution(UnivariateDiscreteDistribution, ContinuousDistribution):
    """
    Abstract base class for integer distributions. Integer distributions also implement the methods of continuous
    distributions.
    """
    variables: Tuple[Integer]

    @property
    def variable(self) -> Integer:
        return self.variables[0]

    def _cdf(self, value: int) -> float:
        """
        Calculate the cumulative distribution function at `value`.
        :param value: The value to evaluate the cdf on.
        :return: The cumulative probability.
        """
        return sum(self._pdf(value) for value in range(value))

    def moment(self, order: VariableMap[Union[Integer, Continuous], int],
               center: VariableMap[Union[Integer, Continuous], float]) \
            -> VariableMap[Union[Integer, Continuous], float]:
        order = order[self.variable]
        center = center[self.variable]
        result = sum([self.pdf(value) * (value - center) ** order for value in self.variable.domain])
        return VariableMap({self.variable: result})


class UniformDistribution(ContinuousDistribution):
    """
    Class for uniform distributions over the half open interval [lower, upper).
    """

    lower: float
    """
    The included lower bound of the interval.
    """

    upper: float
    """
    The excluded upper bound of the interval.
    """

    def __init__(self, variable: Continuous, lower: float, upper: float, parent=None):
        super().__init__(variable, parent)
        if lower >= upper:
            raise ValueError("upper has to be greater than lower. lower: {}; upper: {}")
        self.lower = lower
        self.upper = upper

    @property
    def domain(self) -> portion.Interval:
        return portion.closedopen(self.lower, self.upper)

    def pdf_value(self) -> float:
        """
        Calculate the density of the uniform distribution.
        """
        return 1 / (self.upper - self.lower)

    def _pdf(self, value: float) -> float:
        if value in self.domain:
            return self.pdf_value()
        else:
            return 0

    def _cdf(self, value: float) -> float:
        if value < self.lower:
            return 0
        elif value > self.upper:
            return 1
        else:
            return (value - self.lower) / (self.upper - self.lower)

    def _probability(self, event: EncodedEvent) -> float:
        interval: portion.Interval = event[self.variable]
        probability = 0.

        for interval_ in interval:
            probability += self.cdf(interval_.upper) - self.cdf(interval_.lower)

        return probability

    def _mode(self):
        return [EncodedEvent({self.variable: self.domain})], self.pdf_value()

    def sample(self, amount: int) -> List[List[float]]:
        return [[random.uniform(self.lower, self.upper)] for _ in range(amount)]

    def _conditional(self, event: EncodedEvent) -> Tuple[Optional[Union[DeterministicSumUnit, Self]], float]:
        interval = event[self.variable]
        resulting_distributions = []
        resulting_probabilities = []

        for interval_ in interval:

            # calculate the probability of the interval
            probability = self._probability(EncodedEvent({self.variable: interval_}))

            # if the probability is 0, skip this interval
            if probability == 0:
                continue

            resulting_probabilities.append(probability)
            intersection = self.domain & interval_
            resulting_distributions.append(UniformDistribution(self.variable, intersection.lower, intersection.upper))

        # if there is only one interval, don't create a deterministic sum
        if len(resulting_distributions) == 1:
            return resulting_distributions[0], resulting_probabilities[0]

        # if there are multiple intersections almost surely, create a deterministic sum
        elif len(resulting_distributions) > 1:
            deterministic_sum = DeterministicSumUnit(self.variables, resulting_probabilities)
            deterministic_sum.children = resulting_distributions
            return deterministic_sum.normalize(), sum(resulting_probabilities)

        else:
            return None, 0

    def moment(self, order: VariableMap[Union[Integer, Continuous], int],
               center: VariableMap[Union[Integer, Continuous], float])\
            -> VariableMap[Union[Integer, Continuous], float]:

        order = order[self.variable]
        center = center[self.variable]

        def evaluate_integral_at(x) -> float:
            """
            Helper method to calculate

            .. math::

                    \int_{-\infty}^{\infty} (x - center)^{order} pdf(x) dx = \fract{p(x-center)^(1+order)}{1+order}

            """
            return (self.pdf_value() * (x - center) ** (order + 1)) / (order + 1)
        result = evaluate_integral_at(self.upper) - evaluate_integral_at(self.lower)

        return VariableMap({self.variable: result})

    def __eq__(self, other):
        if not isinstance(other, UniformDistribution):
            return False
        return self.variable == other.variable and self.lower == other.lower and self.upper == other.upper

    def __repr__(self):
        return f"U{self.lower, self.upper}"

    def __copy__(self):
        return self.__class__(self.variable, self.lower, self.upper)
