from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

import numpy as np
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from random_events.variable import *
from random_events.interval import *
from typing_extensions import Union, Iterable, Any, Self, Dict, List, Tuple, DefaultDict
import plotly.graph_objects as go


from ..probabilistic_model import ProbabilisticModel, OrderType, MomentType, CenterType, FullEvidenceType
from ..utils import SubclassJSONSerializer


class UnivariateDistribution(ProbabilisticModel, SubclassJSONSerializer):
    """
    Abstract Base class for Univariate distributions.
    """

    variable: Variable

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return (self.variable, )

    def support(self) -> Event:
        return SimpleEvent({self.variable: self.univariate_support}).as_composite_set()

    @property
    @abstractmethod
    def univariate_support(self) -> AbstractCompositeSet:
        """
        :return: The univariate support of the distribution. This is not an Event.
        """
        raise NotImplementedError

    def log_mode(self) -> Tuple[Event, float]:
        mode, log_likelihood = self.univariate_log_mode()
        return SimpleEvent({self.variable: mode}).as_composite_set(), log_likelihood

    @abstractmethod
    def univariate_log_mode(self) -> Tuple[AbstractCompositeSet, float]:
        """
        :return: The univariate mode of the distribution and its log-likelihood. The mode is not an Event.
        """
        raise NotImplementedError

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

    variable: Continuous

    @property
    @abstractmethod
    def univariate_support(self) -> Interval:
        raise NotImplementedError

    def log_likelihood(self, event: FullEvidenceType) -> float:
        return self.log_pdf(event[0])

    def log_likelihoods(self, events: np.array) -> np.array:
        return self.log_pdfs(events[0, :])

    def pdf(self, value: Union[float, int]) -> float:
        """
        Calculate the probability density function at `value`.
        :param value: The value
        :return: The probability.
        """
        return np.exp(self.log_pdf(value))

    def pdfs(self, values: np.array) -> np.array:
        """
        Calculate the probability density function at `values`.
        :param values: The array of values
        :return: The array of densities.
        """
        return np.exp(self.log_pdfs(values))

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
        return np.array([self.log_pdf(value) for value in values])

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

    def univariate_log_mode(self) -> Tuple[Set, float]:
        max_likelihood = max(self.probabilities.values())
        mode = Set(key for key, value in self.probabilities.items() if value == max_likelihood)
        return mode, np.log(max_likelihood)

    def log_conditional(self, event: Event) -> Tuple[Optional[Self], float]:
        raise NotImplementedError

    def probabilities_for_plotting(self) -> Dict[Union[int, str], float]:
        return {element.name: self.pmf(element) for element in self.variable.domain.simple_sets}

    def univariate_support(self) -> Set:
        return Set(key for key, value in self.probabilities.items() if value > 0)

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        return sum(self.pmf(key) for key in event[self.variable].simple_sets)

    def sample(self, amount: int) -> np.array:
        sample_space = np.array([key.value for key in self.probabilities.keys()])
        sample_probabilities = np.array([value for value in self.probabilities.values()])
        return np.random.choice(sample_space, size=(amount, 1), replace=True, p=sample_probabilities)

    @property
    def representation(self):
        return f"Nominal{self.variable.domain}"


class IntegerDistribution(ContinuousDistribution, DiscreteDistribution):
    """
    Abstract base class for integer distributions. Integer distributions also implement the methods of continuous
    distributions.
    """

    variable: Integer

    def __init__(self, variable: Integer, probabilities: Optional[DefaultDict[Union[int, SetElement], float]]):
        DiscreteDistribution.__init__(self, variable, probabilities)

    def univariate_log_mode(self) -> Tuple[AbstractCompositeSet, float]:
        max_likelihood = max(self.probabilities.values())
        mode = Interval()
        for key, value in self.probabilities.items():
            if value == max_likelihood:
                mode |= singleton(key)
        return mode, np.log(max_likelihood)

    def log_conditional(self, event: Event) -> Tuple[Optional[Self], float]:
        raise NotImplementedError

    def probabilities_for_plotting(self) -> Dict[Union[int, str], float]:
        return self.probabilities

    @property
    def univariate_support(self) -> Interval:
        result = Interval()
        for key, value in self.probabilities.items():
            if value > 0:
                result |= singleton(key)
        return result

    def log_pdf(self, value: Union[float, int]) -> float:
        return np.log(self.pmf(value))

    def cdf(self, value: Union[float, int]) -> float:
        result = 0

        for x, p_x in self.probabilities.items():
            if x <= value:
                result += p_x
            else:
                break

        return result

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        interval: Interval = event[self.variable]
        result = 0

        for x, p_x in self.probabilities.items():
            if x in interval:
                result += p_x

        return result

    def sample(self, amount: int) -> np.array:
        sample_space = np.array(list(self.probabilities.keys()))
        sample_probabilities = np.array([value for value in self.probabilities.values()])
        return np.random.choice(sample_space, size=(amount, 1), replace=True, p=sample_probabilities)

    @property
    def representation(self):
        return f"Ordinal{self.variable.domain}"

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        order = order[self.variable]
        center = center[self.variable]
        result = sum([p_x * (x - center) ** order for x, p_x in self.probabilities.items()])
        return VariableMap({self.variable: result})


class DiracDeltaDistribution(ContinuousDistribution):
    """
    Class for Dirac delta distributions.
    The Dirac measure is used whenever evidence is given as a singleton instance.
    """

    variable: Continuous

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
        self.variable = variable
        self.location = location
        self.density_cap = density_cap

    def log_pdf(self, value: Union[float, int]) -> float:
        return np.log(self.density_cap) if value == self.location else -float("inf")

    def cdf(self, value: Union[float, int]) -> float:
        return 1. if value > self.location else 0.

    @property
    def univariate_support(self) -> Interval:
        return singleton(self.location)

    def sample(self, amount: int) -> np.array:
        return np.full((amount, 1), self.location)

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
        variable = Continuous.from_json(data["variable"])
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

