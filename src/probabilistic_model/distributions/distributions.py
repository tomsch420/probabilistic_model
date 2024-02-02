import abc
import random
from typing import Optional

import portion
import random_events.utils
from random_events.events import EncodedEvent, Event, VariableMap
from random_events.variables import Variable, Continuous, Discrete, Symbolic, Integer
from typing_extensions import Union, Iterable, Any, Self, Dict, List, Tuple
import plotly.graph_objects as go


from ..probabilistic_model import ProbabilisticModel, OrderType, MomentType, CenterType
from ..utils import SubclassJSONSerializer


class UnivariateDistribution(ProbabilisticModel, SubclassJSONSerializer):
    """
    Abstract Base class for Univariate distributions.
    """

    def __init__(self, variable: Variable):
        super().__init__([variable])

    @property
    def domain(self) -> Event:
        """
        The domain of this distribution.
        :return: The domain (support) of this distribution as event.
        """
        raise NotImplementedError

    @property
    def representation(self) -> str:
        """
        The symbol used to represent this distribution.
        """
        return self.__class__.__name__

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

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        if self.variable in variables:
            return self
        else:
            return None

    def __eq__(self, other: Self):
        return self.variables == other.variables

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "variable": self.variable.to_json()
        }

    def plotly_layout(self) -> Dict[str, Any]:
        """
        :return: The layout argument for plotly figures as dict
        """
        return {
            "title": f"{self.__class__.__name__}",
            "xaxis": {"title": self.variable.name}
        }


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

    def conditional_from_singleton(self, singleton: portion.Interval) -> Tuple[Optional['DiracDeltaDistribution'], float]:
        """
        Create a conditional distribution from a singleton interval.

        :return: A DiracDelta distribution at the point described by the singleton. The density cap is set
        to the pdf value of this distribution at the point.
        """
        if singleton.lower != singleton.upper:
            raise ValueError("This method can only be used with singletons.")

        likelihood = self.pdf(singleton.lower)

        if likelihood == 0:
            return None, 0

        else:
            return DiracDeltaDistribution(self.variable, singleton.lower, density_cap=likelihood), likelihood

    def conditional_from_simple_interval(self, interval: portion.Interval) -> Tuple[Optional[Self], float]:
        """
        Create a conditional distribution from an interval that has length one and is not singleton.

        This is the method that should be overloaded by subclasses.
        The _conditional method will call this method if the interval is not singleton and has length one.

        :param interval: The interval to condition on
        :return: A conditional distribution and the probability.
        """
        raise NotImplementedError()

    def conditional_from_complex_interval(self, interval: portion.Interval) -> Tuple[Optional[Self], float]:
        raise NotImplementedError()

    def _conditional(self, event: EncodedEvent) -> \
            Tuple[Optional[Union['ContinuousDistribution', 'DiracDeltaDistribution', ProbabilisticModel]], float]:

        # form intersection of event and domain
        intersection: portion.Interval = event[self.variable].intersection(self.domain[self.variable])

        # if intersection is empty
        if intersection.empty:
            return None, 0

        # if intersection is singleton
        elif intersection.lower == intersection.upper:
            return self.conditional_from_singleton(intersection)

        # if intersection is simple interval
        elif len(intersection) == 1:
            return self.conditional_from_simple_interval(intersection)

        # if intersection is complex interval
        return self.conditional_from_complex_interval(intersection)

    def plot(self) -> List:
        """
        Generate a list of traces that can be used to plot the distribution in plotly figures.
        """

        traces = []
        samples = [sample[0] for sample in self.sample(1000)]
        samples.sort()

        minimal_value = self.domain[self.variable].lower
        if minimal_value <= -float("inf"):
            minimal_value = samples[0]

        maximal_value = self.domain[self.variable].upper
        if maximal_value >= float("inf"):
            maximal_value = samples[-1]

        sample_range = maximal_value - minimal_value
        minimal_value -= 0.05 * sample_range
        maximal_value += 0.05 * sample_range

        samples_with_padding = [minimal_value, samples[0]] + samples + [samples[-1], maximal_value]

        pdf_values = [0, 0] + [self.pdf(sample) for sample in samples] + [0, 0]
        traces.append(go.Scatter(x=samples_with_padding, y=pdf_values, mode="lines", name="PDF"))

        cdf_values = [0, 0] + [self.cdf(sample) for sample in samples] + [1, 1]

        traces.append(go.Scatter(x=samples_with_padding, y=cdf_values, mode="lines", name="CDF"))
        mean = self.expectation([self.variable])[self.variable]

        try:
            modes, maximum_likelihood = self.mode()
        except NotImplementedError:
            modes = []
            maximum_likelihood = max(pdf_values)

        mean_trace = go.Scatter(x=[mean, mean], y=[0, maximum_likelihood * 1.05], mode="lines+markers",
                                name="Expectation")
        traces.append(mean_trace)

        xs = []
        ys = []
        for mode in modes:
            mode = mode[self.variable]
            xs.extend([mode.lower, mode.lower, mode.upper, mode.upper, None])
            ys.extend([0, maximum_likelihood * 1.05, maximum_likelihood * 1.05, 0, None])
        mode_trace = go.Scatter(x=xs, y=ys, mode="lines+markers", name="Mode", fill="toself")
        traces.append(mode_trace)

        return traces


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

    def __hash__(self):
        return hash((self.variable, tuple(self.weights)))

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

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "weights": self.weights
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> 'SubclassJSONSerializer':
        variable = Variable.from_json(data["variable"])
        weights = data["weights"]
        return cls(variable, weights)


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

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        order = order[self.variable]
        center = center[self.variable]
        result = sum([self.pdf(value) * (value - center) ** order for value in self.variable.domain])
        return VariableMap({self.variable: result})

    def plot(self) -> List[Union[go.Bar, go.Scatter]]:
        traces = super().plot()
        _, likelihood = self.mode()
        expectation = self.expectation([self.variable])[self.variable]
        traces.append(go.Scatter(x=[expectation, expectation], y=[0, likelihood * 1.05], mode="lines+markers",
                                 name="Expectation"))
        return traces


class DiracDeltaDistribution(ContinuousDistribution):
    """
    Class for Dirac delta distributions.
    The Dirac measure is used whenever evidence is given as a singleton instance.
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

    def __init__(self, variable: Continuous, location: float, density_cap: float = float("inf")):
        super().__init__(variable)
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

    def _conditional(self, event: EncodedEvent) -> Tuple[Optional[Self], float]:
        if self.location in event[self.variable]:
            return self.__copy__(), 1
        else:
            return None, 0

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        order = order[self.variable]
        center = center[self.variable]

        if order == 0:
            moment = 1.
        elif order == 1:
            moment = self.location - center
        else:
            moment = 0.

        return VariableMap({self.variable: moment})

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                super().__eq__(other) and
                self.location == other.location and
                self.density_cap == other.density_cap)

    def __hash__(self):
        return hash((self.variable, self.location, self.density_cap))

    @property
    def representation(self):
        return f"δ({self.location}, {self.density_cap})"

    def __copy__(self):
        return self.__class__(self.variable, self.location, self.density_cap)

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "location": self.location,
            "density_cap": self.density_cap}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> 'SubclassJSONSerializer':
        variable = Variable.from_json(data["variable"])
        location = data["location"]
        density_cap = data["density_cap"]
        return cls(variable, location, density_cap)

    def __repr__(self):
        return f"δ({self.variable.name}, {self.location}, {self.density_cap})"
