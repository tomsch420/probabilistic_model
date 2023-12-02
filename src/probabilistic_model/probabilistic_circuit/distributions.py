import copy
import random
from typing import Iterable, Tuple, Union, List, Optional, Any, Dict

import plotly.graph_objects as go
import portion
import random_events.variables
from random_events.events import EncodedEvent, VariableMap, Event
from random_events.variables import Variable, Continuous, Symbolic, Integer, Discrete
from typing_extensions import Self

from probabilistic_model.probabilistic_circuit.units import Unit, DeterministicSumUnit
from probabilistic_model.probabilistic_model import OrderType, CenterType, MomentType


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

    def is_decomposable(self) -> bool:
        return True

    def is_smooth(self) -> bool:
        return True

    def is_deterministic(self) -> bool:
        return True

    def maximize_expressiveness(self) -> Self:
        return copy.copy(self)

    def simplify(self) -> Self:
        return copy.copy(self)

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "variable": self.variable.to_json()}

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

    def conditional_from_singleton(self, singleton: portion.Interval) -> Tuple[
        Optional['DiracDeltaDistribution'], float]:
        """
        Create a dirac impulse from a singleton interval. The density is capped at the likelihood of the given value.

        :param singleton: The singleton interval from an encoded event.
        :return: A dirac impulse and the likelihood.
        """
        if singleton.lower != singleton.upper:
            raise ValueError("This method can only be used with singletons.")

        likelihood = self.pdf(singleton.lower)

        if likelihood == 0:
            return None, 0

        else:
            return DiracDeltaDistribution(self.variable, singleton.lower, density_cap=likelihood), likelihood

    def conditional_from_interval(self, interval: portion.Interval) -> Tuple[Optional[Self], float]:
        """
        Create a conditional distribution from an interval that is not singleton.

        This is the method that should be overloaded by subclasses.
        The _conditional method will call this method if the interval is not singleton.

        :param interval: The interval from an encoded event.
        :return: A conditional distribution and the probability.
        """
        raise NotImplementedError

    def _conditional(self, event: EncodedEvent) -> Tuple[Optional[Self], float]:

        resulting_distributions = []
        resulting_probabilities = []

        for interval in event[self.variable]:

            # handle the singleton case
            if interval.lower == interval.upper:
                distribution, probability = self.conditional_from_singleton(interval)

            # handle the non-singleton case
            else:
                distribution, probability = self.conditional_from_interval(interval)

            if probability > 0:
                resulting_distributions.append(distribution)
                resulting_probabilities.append(probability)

        if len(resulting_distributions) == 0:
            return None, 0

        # if there is only one interval, don't create a deterministic sum
        if len(resulting_distributions) == 1:
            return resulting_distributions[0], resulting_probabilities[0]

        # if there are multiple intersections almost surely, create a deterministic sum
        elif len(resulting_distributions) > 1:
            deterministic_sum = DeterministicSumUnit(self.variables, resulting_probabilities)
            deterministic_sum.children = resulting_distributions
            return deterministic_sum.normalize(), sum(resulting_probabilities)


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

    # @property
    # def representation(self):
    #     return f"Categorical()"

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
        return (isinstance(other, UnivariateDiscreteDistribution) and self.weights == other.weights and super().__eq__(
            other))

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "weights": self.weights}

    @classmethod
    def from_json_with_variables_and_children(cls, data: Dict[str, Any], variables: List[Variable],
                                              children: List['Unit']) -> Self:
        variable = random_events.variables.Variable.from_json(data["variable"])
        return cls(variable, data["weights"])

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


class SymbolicDistribution(UnivariateDiscreteDistribution):
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

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
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

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        order = order[self.variable]
        center = center[self.variable]
        result = sum([self.pdf(value) * (value - center) ** order for value in self.variable.domain])
        return VariableMap({self.variable: result})

    def plot(self) -> List[Union[go.Bar, go.Scatter]]:
        traces = UnivariateDiscreteDistribution.plot(self)
        _, likelihood = self.mode()
        expectation = self.expectation([self.variable])[self.variable]
        traces.append(go.Scatter(x=[expectation, expectation], y=[0, likelihood * 1.05], mode="lines+markers",
                                 name="Expectation"))
        return traces


class UniformDistribution(ContinuousDistribution):
    """
    Class for uniform distributions over the half-open interval [lower, upper).
    """

    interval: portion.Interval
    """
    The interval that the Uniform distribution is defined over.
    """

    def __init__(self, variable: Continuous, interval: portion.Interval, parent=None):
        super().__init__(variable, parent)
        self.interval = interval

    @property
    def domain(self) -> Event:
        return Event({self.variable: self.interval})

    @property
    def lower(self) -> float:
        return self.interval.lower

    @property
    def upper(self) -> float:
        return self.interval.upper

    def pdf_value(self) -> float:
        """
        Calculate the density of the uniform distribution.
        """
        return 1 / (self.upper - self.lower)

    def _pdf(self, value: float) -> float:
        if portion.singleton(value) in self.interval:
            return self.pdf_value()
        else:
            return 0

    def _cdf(self, value: float) -> float:

        # check edge cases
        if value <= -portion.inf:
            return 0.
        if value >= portion.inf:
            return 1.

        # convert to singleton
        singleton = portion.singleton(value)

        if singleton < self.interval:
            return 0
        elif singleton > self.interval:
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
        return [self.domain.encode()], self.pdf_value()

    def sample(self, amount: int) -> List[List[float]]:
        return [[random.uniform(self.lower, self.upper)] for _ in range(amount)]

    def conditional_from_interval(self, interval: portion.Interval) -> Tuple[
        Optional[Union[DeterministicSumUnit, Self]], float]:

        # calculate the probability of the interval
        probability = self._probability(EncodedEvent({self.variable: interval}))

        # if the probability is 0, return None
        if probability == 0:
            return None, 0

        # else, form the intersection of the interval and the domain
        intersection = self.interval & interval
        resulting_distribution = UniformDistribution(self.variable, intersection)
        return resulting_distribution, probability

    def moment(self, order: OrderType, center: CenterType) -> MomentType:

        order = order[self.variable]
        center = center[self.variable]

        def evaluate_integral_at(x) -> float:
            r"""
            Helper method to calculate

            .. math::

                    \int_{-\infty}^{\infty} (x - center)^{order} pdf(x) dx = \frac{p(x-center)^(1+order)}{1+order}

            """
            return (self.pdf_value() * (x - center) ** (order + 1)) / (order + 1)

        result = evaluate_integral_at(self.upper) - evaluate_integral_at(self.lower)

        return VariableMap({self.variable: result})

    def __eq__(self, other):
        return isinstance(other, UniformDistribution) and self.interval == other.interval and super().__eq__(other)

    @property
    def representation(self):
        return f"U{self.interval}"

    def __copy__(self):
        return self.__class__(self.variable, self.interval)

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "interval": portion.to_data(self.interval)}

    @classmethod
    def from_json_with_variables_and_children(cls, data: Dict[str, Any], variables: List[Variable],
                                              children: List['Unit']) -> Self:
        variable = random_events.variables.Variable.from_json(data["variable"])
        return cls(variable, portion.from_data(data["interval"]))


class DiracDeltaDistribution(ContinuousDistribution):
    """
    Class for Dirac delta distributions.
    """

    location: float
    """
    The location of the Dirac delta distribution.
    """

    density_cap: float
    """
    The density cap of the Dirac delta distribution.
    This value will be used to replace infinity in likelihood.
    """

    def __init__(self, variable: Continuous, location: float, density_cap: float = float("inf"), parent=None):
        super().__init__(variable, parent)
        self.location = location
        self.density_cap = density_cap

    @property
    def domain(self) -> Event:
        return Event({self.variable: portion.singleton(self.location)})

    def _pdf(self, value: float) -> float:
        if value == self.location:
            return self.density_cap
        else:
            return 0

    def _cdf(self, value: float) -> float:
        if value < self.location:
            return 0
        else:
            return 1

    def _probability(self, event: EncodedEvent) -> float:
        if self.location in event[self.variable]:
            return 1
        else:
            return 0

    def _mode(self):
        return [self.domain.encode()], self.density_cap

    def sample(self, amount: int) -> List[List[float]]:
        return [[self.location] for _ in range(amount)]

    def _conditional(self, event: EncodedEvent) -> Tuple[Optional[Union[DeterministicSumUnit, Self]], float]:
        if self.location in event[self.variable]:
            return self.__copy__(), 1
        else:
            return None, 0

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        order = order[self.variable]
        center = center[self.variable]

        if order == 1:
            return VariableMap({self.variable: self.location - center})
        elif order == 2:
            return VariableMap({self.variable: (self.location - center) ** 2})
        else:
            return VariableMap({self.variable: 0})

    def __eq__(self, other):
        return (isinstance(other,
                           self.__class__) and self.location == other.location and self.density_cap == other.density_cap and super().__eq__(
            other))

    @property
    def representation(self):
        return f"DiracDelta({self.location}, {self.density_cap})"

    def __copy__(self):
        return self.__class__(self.variable, self.location, self.density_cap)

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "location": self.location, "density_cap": self.density_cap}

    @classmethod
    def from_json_with_variables_and_children(cls, data: Dict[str, Any], variables: List[Variable],
                                              children: List['Unit']) -> Self:
        variable = random_events.variables.Variable.from_json(data["variable"])
        return cls(variable, data["location"], data["density_cap"])
