from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
from random_events.interval import *
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from random_events.variable import *
from typing_extensions import Union, Iterable, Any, Self, Dict, List, Tuple

from ..constants import SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT
from ..interfaces.drawio.drawio import DrawIOInterface
from ..probabilistic_model import ProbabilisticModel, OrderType, MomentType, CenterType
from ..utils import MissingDict, interval_as_array


class UnivariateDistribution(ProbabilisticModel, SubclassJSONSerializer, DrawIOInterface):
    """
    Abstract Base class for Univariate distributions.
    """

    variable: Variable

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return (self.variable,)

    @property
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
        return {**super().to_json(), "variable": self.variable.to_json()}

    def composite_set_from_event(self, event: Event) -> AbstractCompositeSet:
        """
        Extract the composite set from the event that is relevant for this distribution.
        :param event: The event
        :return: The composite set
        """
        return event.marginal(SortedSet(self.variables)).simple_sets[0][self.variable]

    @property
    def abbreviated_symbol(self) -> str:
        return "P"

    @property
    def drawio_style(self) -> Dict[str, Any]:
        return {"style": self.drawio_label, "width": 30, "height": 30, "label": self.abbreviated_symbol}


class ContinuousDistribution(UnivariateDistribution):
    """
    Abstract base class for continuous distributions.
    """

    variable: Continuous

    @property
    @abstractmethod
    def univariate_support(self) -> Interval:
        raise NotImplementedError

    def cdf(self, x: np.array) -> np.array:
        """
        Calculate the cumulative distribution function at x.
        :param x: The data
        :return: The cumulative distribution function at x
        """
        raise NotImplementedError

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        interval: Interval = event[self.variable]
        points = interval_as_array(interval)
        upper_bound_cdf = self.cdf(points[:, (1,)])
        lower_bound_cdf = self.cdf(points[:, (0,)])
        return (upper_bound_cdf - lower_bound_cdf).sum()

    def log_conditional(self, event: Event) -> Tuple[Optional[Self], float]:
        event = event & self.support
        if event.is_empty():
            return None, -np.inf

        interval = self.composite_set_from_event(event)

        if len(interval.simple_sets) == 1:
            return self.log_conditional_from_simple_interval(interval.simple_sets[0])

        else:
            return self.log_conditional_from_interval(interval)

    def log_conditional_from_singleton(self, interval: SimpleInterval) -> Tuple[DiracDeltaDistribution, float]:
        """
        Calculate the conditional distribution given a singleton event with p(event) > 0.

        In this case, the conditional distribution is a Dirac delta distribution and the log-likelihood is chosen
        for the log-probability.

        :param interval: The singleton event
        :return: The conditional distribution and the log-probability of the event.
        """
        log_pdf_value = self.log_likelihood(np.array([[interval.lower]]))[0]
        return DiracDeltaDistribution(self.variable, interval.lower, np.exp(log_pdf_value)), log_pdf_value

    def log_conditional_from_simple_interval(self, interval: SimpleInterval) -> Tuple[Self, float]:
        """
        Calculate the conditional distribution given a simple interval with p(interval) > 0.
        The interval could also be a singleton.

        :param interval: The simple interval
        :return: The conditional distribution and the log-probability of the interval.
        """
        if interval.is_singleton():
            return self.log_conditional_from_singleton(interval)
        return self.log_conditional_from_non_singleton_simple_interval(interval)

    def log_conditional_from_non_singleton_simple_interval(self, interval: SimpleInterval) -> Tuple[Self, float]:
        """
        Calculate the conditional distribution given a non-singleton, simple interval with p(interval) > 0.
        :param interval: The simple interval.
        :return: The conditional distribution and the log-probability of the interval.
        """
        raise NotImplementedError

    def log_conditional_from_interval(self, interval) -> Tuple[Self, float]:
        """
        Calculate the conditional distribution given an interval with p(interval) > 0.
        :param interval: The simple interval
        :return: The conditional distribution and the log-probability of the interval.
        """
        raise NotImplementedError


class ContinuousDistributionWithFiniteSupport(ContinuousDistribution):
    """
    Abstract base class for continuous distributions with finite support.
    """

    interval: SimpleInterval
    """
    The interval of the distribution.
    """

    @property
    def lower(self) -> float:
        return self.interval.lower

    @property
    def upper(self) -> float:
        return self.interval.upper

    @property
    def univariate_support(self) -> Interval:
        return self.interval.as_composite_set()

    def left_included_condition(self, x: np.array) -> np.array:
        """
        Check if x is included in the left bound of the interval.
        :param x: The data
        :return: A boolean array
        """
        return self.interval.lower <= x if self.interval.left == Bound.CLOSED else self.interval.lower < x

    def right_included_condition(self, x: np.array) -> np.array:
        """
         Check if x is included in the right bound of the interval.
         :param x: The data
         :return: A boolean array
         """
        return x < self.interval.upper if self.interval.right == Bound.OPEN else x <= self.interval.upper

    def included_condition(self, x: np.array) -> np.array:
        """
         Check if x is included in interval.
         :param x: The data
         :return: A boolean array
         """
        return self.left_included_condition(x) & self.right_included_condition(x)

    def log_likelihood(self, x: np.array) -> np.array:
        result = np.full(x.shape[:-1], -np.inf)
        include_condition = self.included_condition(x)
        filtered_x = x[include_condition].reshape(-1, 1)
        result[include_condition[:, 0]] = self.log_likelihood_without_bounds_check(filtered_x)
        return result

    @abstractmethod
    def log_likelihood_without_bounds_check(self, x: np.array) -> np.array:
        """
        Evaluate the logarithmic likelihood function at `x` without checking the inclusion into the interval.
        :param x: x where p(x) > 0
        :return: log(p(x))
        """
        raise NotImplementedError


class DiscreteDistribution(UnivariateDistribution):
    """
    Abstract base class for univariate discrete distributions.
    """

    variable: Union[Symbolic, Integer]

    probabilities: MissingDict[int, float] = MissingDict(float)
    """
    A dict that maps from integers to probabilities.
    In Symbolic cases, the integers are obtained by casting the elements, which inherit from int, to integers.
    """

    def __init__(self, variable: Union[Symbolic, Integer],
                 probabilities: Optional[MissingDict[Union[int, SetElement], float]]):
        super().__init__()
        self.variable = variable

        if probabilities is not None:
            self.probabilities = probabilities

    def __eq__(self, other):
        return (isinstance(other,
                           DiscreteDistribution) and self.probabilities == other.probabilities and super().__eq__(
            other))

    def __hash__(self):
        return hash((self.variable, tuple(self.probabilities.items())))

    def log_likelihood(self, events: np.array) -> np.array:
        events = events[:, 0]
        result = np.full(len(events), -np.inf)
        for x, p in self.probabilities.items():
            result[events == x] = np.log(p)
        return result

    def fit(self, data: np.array) -> Self:
        """
        Fit the distribution to the data.

        The probabilities are set equal to the frequencies in the data.

        :param data: The data.
        :return: The fitted distribution
        """
        unique, counts = np.unique(data, return_counts=True)
        probabilities = MissingDict(float)
        for value, count in zip(unique, counts):
            probabilities[int(value)] = count / len(data)
        self.probabilities = probabilities
        return self

    @abstractmethod
    def probabilities_for_plotting(self) -> Dict[Union[int, str], float]:
        """
        :return: The probabilities as dict that can be plotted.
        """
        raise NotImplementedError

    def plot(self, **kwargs) -> List[go.Bar]:
        """
        Plot the distribution.
        """
        probabilities = self.probabilities_for_plotting()
        max_likelihood = max(probabilities.values())
        non_mode_trace = {x: p for x, p in probabilities.items() if p != max_likelihood}
        traces = [go.Bar(x=list(non_mode_trace.keys()), y=list(non_mode_trace.values()), name="Probability")]

        mode = [key for key, value in probabilities.items() if value == max_likelihood]
        traces.append(go.Bar(x=mode, y=[max_likelihood] * len(mode), name="Mode"))
        return traces

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "probabilities": list(self.probabilities.items())}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variable = Variable.from_json(data["variable"])
        probabilities = MissingDict(float)
        for key, value in data["probabilities"]:
            probabilities[key] = value
        return cls(variable, probabilities)

    def normalize(self):
        """
        Normalize the distribution.
        """
        total = sum(self.probabilities.values())
        for key in self.probabilities:
            self.probabilities[key] /= total

    def log_conditional(self, event: Event) -> Tuple[Optional[Self], float]:
        # construct event
        condition = self.composite_set_from_event(event)
        return self.log_conditional_of_composite_set(condition)

    def log_conditional_of_composite_set(self, event: AbstractCompositeSet) -> Tuple[Optional[Self], float]:
        # calculate new probabilities
        new_probabilities = MissingDict(float)
        for x, p_x in self.probabilities.items():
            if x in event:
                new_probabilities[x] = p_x

        # if the event is impossible, return None and 0
        probability = sum(new_probabilities.values())

        if probability == 0:
            return None, -np.inf

        result = self.__class__(self.variable, new_probabilities)
        result.normalize()
        return result, np.log(probability)

    def __copy__(self) -> Self:
        return self.__class__(self.variable, self.probabilities)

    def __repr__(self):
        return f"P({self.variable.name})"

    def sample(self, amount: int) -> np.array:
        sample_space = np.array(list(self.probabilities.keys()))
        sample_probabilities = np.array([value for value in self.probabilities.values()])
        return np.random.choice(sample_space, size=(amount, 1), replace=True, p=sample_probabilities)


class SymbolicDistribution(DiscreteDistribution):
    """
    Class for symbolic (categorical) distributions.
    """

    variable: Symbolic

    def univariate_log_mode(self) -> Tuple[Set, float]:
        clazz = self.variable.domain_type()
        max_likelihood = max(self.probabilities.values())
        mode = Set(*(clazz(key) for key, value in self.probabilities.items() if value == max_likelihood))

        return mode, np.log(max_likelihood)

    def probabilities_for_plotting(self) -> Dict[Union[int, str], float]:
        return {element.name: self.probabilities[int(element)] for element in self.variable.domain.simple_sets}

    @property
    def univariate_support(self) -> Set:
        clazz = self.variable.domain_type()
        return Set(*[clazz(key) for key, value in self.probabilities.items() if value > 0])

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        return sum(self.probabilities[int(key)] for key in event[self.variable].simple_sets)

    @property
    def representation(self):
        return f"Nominal({self.variable.name}, {self.variable.domain.simple_sets[0].all_elements.__name__})"

    @property
    def drawio_label(self):
        return "rounded=1;whiteSpace=wrap;html=1;labelPosition=center;verticalLabelPosition=top;align=center;verticalAlign=bottom;"

    @property
    def image(self):
        return os.path.join(os.path.dirname(__file__), "../../../", "resources", "icons", "defaultIcon.png")


class IntegerDistribution(ContinuousDistribution, DiscreteDistribution):
    """
    Abstract base class for integer distributions. Integer distributions also implement the methods of continuous
    distributions.
    """

    variable: Integer

    def __init__(self, variable: Integer, probabilities: Optional[MissingDict[Union[int, SetElement], float]]):
        DiscreteDistribution.__init__(self, variable, probabilities)

    def log_conditional(self, event: Event) -> Tuple[Optional[Self], float]:
        return DiscreteDistribution.log_conditional(self, event)

    def univariate_log_mode(self) -> Tuple[AbstractCompositeSet, float]:
        max_likelihood = max(self.probabilities.values())
        mode = Interval()
        for key, value in self.probabilities.items():
            if value == max_likelihood:
                mode |= singleton(key)
        return mode, np.log(max_likelihood)

    def probabilities_for_plotting(self) -> Dict[Union[int, str], float]:
        return {x: p_x for x, p_x in self.probabilities.items() if p_x > 0}

    @property
    def univariate_support(self) -> Interval:
        result = Interval()
        for key, value in self.probabilities.items():
            if value > 0:
                result |= singleton(key)
        return result

    def cdf(self, x: np.array) -> np.array:
        result = np.zeros((len(x),))
        maximum_value = max(x)
        for value, p in self.probabilities.items():
            if value > maximum_value:
                break
            else:
                result[x[:, 0] >= value] += p

        return result

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        interval: Interval = event[self.variable]
        result = 0

        for x, p_x in self.probabilities.items():
            if x in interval:
                result += p_x

        return result

    @property
    def representation(self):
        return f"Ordinal({self.variable.name})"

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        order = order[self.variable]
        center = center[self.variable]
        result = sum([p_x * (x - center) ** order for x, p_x in self.probabilities.items()])
        return VariableMap({self.variable: result})

    def plot(self, **kwargs) -> List[go.Bar]:
        height = max(self.probabilities.values()) * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT
        return super().plot() + [self.univariate_expectation_trace(height)]


class DiracDeltaDistribution(ContinuousDistribution):
    """
    Class for Dirac delta distributions.
    The Dirac measure is used whenever evidence is given as a singleton instance.

    https://en.wikipedia.org/wiki/Dirac_delta_function
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

    def __init__(self, variable: Continuous, location: float, density_cap: float = np.inf):
        super().__init__()
        self.variable = variable
        self.location = location
        self.density_cap = density_cap

    def log_likelihood(self, events: np.array) -> np.array:
        result = np.full(len(events), -np.inf)
        result[events[:, 0] == self.location] = np.log(self.density_cap)
        return result

    def cdf(self, x: np.array) -> np.array:
        result = np.zeros((len(x),))
        result[x[:, 0] >= self.location] = 1.
        return result

    @property
    def abbreviated_symbol(self) -> str:
        return "δ"

    @property
    def univariate_support(self) -> Interval:
        return singleton(self.location)

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        interval: Interval = event[self.variable]
        return 1. if self.location in interval else 0.

    def univariate_log_mode(self) -> Tuple[AbstractCompositeSet, float]:
        return self.univariate_support, np.log(self.density_cap)

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
        return (isinstance(other, self.__class__) and super().__eq__(
            other) and self.location == other.location and self.density_cap == other.density_cap)

    def __hash__(self):
        return hash((self.variable, self.location, self.density_cap))

    @property
    def representation(self):
        return f"δ({self.location}, {self.density_cap})"

    def __repr__(self):
        return f"δ({self.variable.name})"

    def __copy__(self):
        return self.__class__(self.variable, self.location, self.density_cap)

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "location": self.location, "density_cap": self.density_cap}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variable = Continuous.from_json(data["variable"])
        location = data["location"]
        density_cap = data["density_cap"]
        return cls(variable, location, density_cap)

    def plot(self, **kwargs) -> List:
        lower_border = self.location - 1
        upper_border = self.location + 1
        pdf_trace = go.Scatter(x=[lower_border, self.location, self.location, self.location, upper_border],
                               y=[0, 0, self.density_cap, 0, 0], mode="lines", name="PDF")
        cdf_trace = go.Scatter(x=[lower_border, self.location, self.location, upper_border], y=[0, 0, 1, 1],
                               mode="lines", name="CDF")
        expectation_trace = go.Scatter(x=[self.location, self.location],
                                       y=[0, self.density_cap * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT],
                                       mode="lines+markers", name="Expectation")
        mode_trace = go.Scatter(x=[self.location, self.location],
                                y=[0, self.density_cap * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT], mode="lines+markers",
                                name="Mode")
        return [pdf_trace, cdf_trace, expectation_trace, mode_trace]
