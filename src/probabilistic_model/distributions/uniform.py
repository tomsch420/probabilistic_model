import random
from typing import Tuple, Optional

import portion
from plotly import graph_objects as go
from random_events.events import Event, EncodedEvent, VariableMap
from random_events.variables import Continuous
from typing_extensions import List, Dict, Any, Self

from probabilistic_model.probabilistic_model import OrderType, CenterType, MomentType
from .distributions import ContinuousDistribution


class UniformDistribution(ContinuousDistribution):
    """
    Class for uniform distributions over the half-open interval [lower, upper).
    """

    interval: portion.Interval
    """
    The interval that the Uniform distribution is defined over.
    """

    def __init__(self, variable: Continuous, interval: portion.Interval):
        super().__init__(variable)
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
        return (isinstance(other, UniformDistribution) and self.interval == other.interval
                and self.variable == other.variable)

    @property
    def representation(self):
        return f"U({self.interval})"

    def __copy__(self):
        return self.__class__(self.variable, self.interval)

    def conditional_from_simple_interval(self, interval: portion.Interval) -> Tuple[Optional[Self], float]:
        return self.__class__(self.variable, interval), self.cdf(interval.upper) - self.cdf(interval.lower)

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "interval": portion.to_data(self.interval)}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variable = Continuous.from_json(data["variable"])
        interval = portion.from_data(data["interval"])
        return cls(variable, interval)

    def plot(self) -> List:
        domain_size = self.domain[self.variable].upper - self.domain[self.variable].lower
        x = [self.domain[self.variable].lower - domain_size * 0.05, self.domain[self.variable].lower, None,
             self.domain[self.variable].lower, self.domain[self.variable].upper, None, self.domain[self.variable].upper,
             self.domain[self.variable].upper + domain_size * 0.05]

        pdf_values = [0, 0, None, self.pdf_value(), self.pdf_value(), None, 0, 0]
        pdf_trace = go.Scatter(x=x, y=pdf_values, mode='lines', name="Probability Density Function")

        cdf_values = [value if value is None else self.cdf(value) for value in x]
        cdf_trace = go.Scatter(x=x, y=cdf_values, mode='lines', name="Cumulative Distribution Function")

        mode, maximum_likelihood = self.mode()
        mode = mode[0][self.variable]

        expectation = self.expectation([self.variable])[self.variable]
        mode_trace = (go.Scatter(x=[mode.lower, mode.lower, mode.upper, mode.upper, ],
                                 y=[0, maximum_likelihood * 1.05, maximum_likelihood * 1.05, 0], mode='lines+markers',
                                 name="Mode", fill="toself"))
        expectation_trace = (
            go.Scatter(x=[expectation, expectation], y=[0, maximum_likelihood * 1.05], mode='lines+markers',
                       name="Expectation"))
        return [pdf_trace, cdf_trace, mode_trace, expectation_trace]

    def __hash__(self):
        return hash((self.variable.name, hash(self.interval)))

    def parameters(self):
        return {"variable": self.variable, "interval": self.interval}

