from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

import numpy
import numpy as np
import portion
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import *
from random_events.interval import *
from typing_extensions import Union, Iterable, Any, Self, Dict, List, Tuple, DefaultDict
import plotly.graph_objects as go


from ..probabilistic_model import ProbabilisticModel, OrderType, MomentType, CenterType, FullEvidenceType
from ..utils import SubclassJSONSerializer


class UnivariateDistribution(ProbabilisticModel, SubclassJSONSerializer, ABC):
    """
    Abstract Base class for Univariate distributions.
    """

    @property
    @abstractmethod
    def variable(self) -> Variable:
        """
        :return: The variable of the distribution.
        """
        raise NotImplementedError

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return (self.variable, )

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        if self.variable in variables:
            return self
        else:
            return None

    def __eq__(self, other: Self):
        return self.variables == other.variables and isinstance(other, self.__class__)

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "variable": self.variable.to_json()
        }

    @abstractmethod
    def plot(self) -> List:
        """
        Generate a list of traces that can be used to plot the distribution in plotly figures.
        """
        raise NotImplementedError

    def plotly_layout(self) -> Dict[str, Any]:
        """
        :return: The layout argument for plotly figures as dict
        """
        return {
            "title": f"{self.representation}",
            "xaxis": {"title": self.variable.name}
        }


class ContinuousDistribution(UnivariateDistribution):
    """
    Abstract base class for continuous distributions.
    """

    @property
    @abstractmethod
    def variable(self) -> Continuous:
        """
        :return: The variable of the distribution.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def simple_support(self) -> SimpleInterval:
        """
        :return: The support of the distribution as simple Interval.
        """
        raise NotImplementedError

    def support(self) -> Event:
        return SimpleEvent({self.variable: self.simple_support}).as_composite_set()

    @abstractmethod
    def log_pdf(self, value: Union[float, int]) -> float:
        """
        Evaluate the logarithmic probability density function at `value`.
        :param value: x
        :return: p(x)
        """
        raise NotImplementedError

    def log_pdfs(self, values: np.array) -> np.array:
        """
        Evaluate the logarithmic probability density function at `values`.
        :param values: The array of values
        :return: The array of densities.
        """

    @abstractmethod
    def cdf(self, value: Union[float, int]) -> float:
        """
        Evaluate the cumulative distribution function at `value`.
        :param value: The value
        :return: The probability.
        """
        raise NotImplementedError

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        interval: Interval = event[self.variable]
        return sum(self.cdf(simple_interval.upper) - self.cdf(simple_interval.lower) for simple_interval
                   in interval.simple_sets)


class DiscreteDistribution(UnivariateDistribution):
    """
    Abstract base class for univariate discrete distributions.
    """

    variable: Union[Symbolic, Integer]

    probabilities: DefaultDict[Union[int, SetElement], float]
    """
    A dict that maps from elements of the variables domain to probabilities.
    """

    def __init__(self, variable: Union[Symbolic, Integer],
                 probabilities: Optional[DefaultDict[Union[int, SetElement], float]]):
        self.variable = variable

        if probabilities is None:
            probabilities = defaultdict(float)
        self.probabilities = probabilities

    def __copy__(self):
        return self.__class__(self.variable, self.probabilities)

    def __eq__(self, other):
        return (isinstance(other, DiscreteDistribution) and self.probabilities == other.probabilities and
                super().__eq__(other))

    def __hash__(self):
        return hash((self.variable, tuple(self.probabilities.items())))

    def log_likelihood(self, event: FullEvidenceType) -> float:
        return np.log(self.pmf(event))

    def pmf(self, value: Union[int, SetElement]) -> float:
        """
        Calculate the probability mass function at `value`.
        :param value: The value
        :return: The probability.
        """
        return self.probabilities[value]

    def fit(self, data: np.array) -> Self:
        """
        Fit the distribution to the data.

        The probabilities are set equal to the frequencies in the data.

        :param data: The data.
        :return: The fitted distribution
        """
        unique, counts = np.unique(data, return_counts=True)
        probabilities = defaultdict(float)
        for value, count in zip(unique, counts):
            probabilities[value] = count / len(data)
        self.probabilities = probabilities
        return self

    @abstractmethod
    def probabilities_for_plotting(self) -> Dict[Union[int, str], float]:
        """
        :return: The probabilities as dict that can be plotted.
        """
        raise NotImplementedError

    def plot(self) -> List[go.Bar]:
        """
        Plot the distribution.
        """
        probabilities = self.probabilities_for_plotting()
        traces = [go.Bar(x=probabilities.keys(), y=probabilities.values(), name="Probability")]

        max_likelihood = max(probabilities.values())
        mode = [key for key, value in probabilities.items() if value == max_likelihood]
        traces.append(go.Bar(x=mode, y=[max_likelihood] * len(mode), name="Mode"))
        return traces

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "probabilities": list(self.probabilities.items())
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variable = Variable.from_json(data["variable"])
        probabilities = defaultdict(float)
        for key, value in data["probabilities"]:
            probabilities[key] = value
        return cls(variable, probabilities)


class SymbolicDistribution(DiscreteDistribution):
    """
    Class for symbolic (categorical) distributions.
    """

    variable: Symbolic

    def probabilities_for_plotting(self) -> Dict[Union[int, str], float]:
        return {element.name: self.pmf(element) for element in self.variable.domain.simple_sets}

    def support(self) -> Event:
        return (SimpleEvent({self.variable: Set(key for key, value in self.probabilities.items() if value > 0)}).
                as_composite_set())

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        return sum(self.pmf(key) for key in event[self.variable].simple_sets)

    def mode(self) -> Tuple[Event, float]:
        max_likelihood = max(self.probabilities.values())
        mode = {key for key, value in self.probabilities.items() if value == max_likelihood}
        return SimpleEvent({self.variable: mode}).as_composite_set(), max_likelihood

    def sample(self, amount: int) -> np.array:
        sample_space = np.array([key.value for key in self.probabilities.keys()])
        sample_probabilities = np.array([value for value in self.probabilities.values()])
        return np.random.choice(sample_space, size=(amount, 1), replace=True, p=sample_probabilities)

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
    def domain(self) -> ComplexEvent:
        return ComplexEvent([Event({self.variable: portion.singleton(self.location)})])

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

    def _mode(self) -> Tuple[ComplexEvent, float]:
        return self.domain.encode(), self.density_cap

    def sample(self, amount: int) -> List[List[float]]:
        return [[self.location] for _ in range(amount)]

    def _conditional(self, event: ComplexEvent) -> Tuple[Optional[Self], float]:
        for event in event.events:
            if self.location in event[self.variable]:
                return self.__copy__(), 1
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

    def plot(self) -> List:
        lower_border = self.location - 1
        upper_border = self.location + 1
        pdf_trace = go.Scatter(x=[lower_border, self.location, self.location, self.location, upper_border],
                               y=[0, 0, self.density_cap, 0, 0], mode="lines", name="PDF")
        cdf_trace = go.Scatter(x=[lower_border, self.location, self.location, upper_border],
                               y=[0, 0, 1, 1], mode="lines", name="CDF")
        expectation_trace = go.Scatter(x=[self.location, self.location], y=[0, self.density_cap * 1.05],
                                       mode="lines+markers", name="Expectation")
        mode_trace = go.Scatter(x=[self.location, self.location], y=[0, self.density_cap * 1.05],
                                mode="lines+markers", name="Mode")
        return [pdf_trace, cdf_trace, expectation_trace, mode_trace]

