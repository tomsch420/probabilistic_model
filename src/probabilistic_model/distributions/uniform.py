import os
import random
from typing import Tuple, Optional


import portion
from plotly import graph_objects as go
from random_events.events import Event, EncodedEvent, VariableMap, ComplexEvent
from random_events.variables import Continuous
from sortedcontainers import SortedSet
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
    def domain(self) -> ComplexEvent:
        return ComplexEvent([Event({self.variable: self.interval})])

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
        return self.domain.encode(), self.pdf_value()

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
    def label(self):
        return "rounded=1;whiteSpace=wrap;html=1;labelPosition=center;verticalLabelPosition=top;align=center;verticalAlign=bottom;"
    @property
    def representation(self):
        return f"U({self.interval})"

    @property
    def image(self):
        return os.path.join(os.path.dirname(__file__),"../../../", "resources", "icons", "defaultIcon.png")

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
        domain = self.domain.events[0][self.variable]
        domain_size = domain.upper - domain.lower
        x = [domain.lower - domain_size * 0.05, domain.lower, None,
             domain.lower, domain.upper, None, domain.upper,
             domain.upper + domain_size * 0.05]

        pdf_values = [0, 0, None, self.pdf_value(), self.pdf_value(), None, 0, 0]
        pdf_trace = go.Scatter(x=x, y=pdf_values, mode='lines', name="Probability Density Function")

        cdf_values = [value if value is None else self.cdf(value) for value in x]
        cdf_trace = go.Scatter(x=x, y=cdf_values, mode='lines', name="Cumulative Distribution Function")

        mode, maximum_likelihood = self.mode()
        mode = mode.events[0][self.variable]

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

    def area_validation_metric(self, other: ContinuousDistribution) -> float:
        """
        Calculate the area validation metric of this distribution and another.

        ..math:: \int_{-\infty}^\infty |self(x) - other(x)| dx
        """
        distance = 0.
        if isinstance(other, UniformDistribution):

            # calculate AVM of intersecting part
            intersection = self.interval.intersection(other.interval)

            if not intersection.empty:
                difference_of_pdfs = abs(self.pdf_value() - other.pdf_value())
                distance += difference_of_pdfs * (intersection.upper - intersection.lower)

            # calculate AVM of non-intersecting parts
            difference = self.interval.union(other.interval).difference(intersection)
            for interval in difference:
                pdf_value = self.pdf_value() if interval in self.interval else other.pdf_value()
                distance += pdf_value * (interval.upper - interval.lower)

        else:
            raise NotImplementedError(f"AVM between UniformDistribution and {type(other)} is not known.")
        return 1- distance

    # def event_of_higher_density(self, other: Self, own_node_weights, other_node_weights):
    #     resulting_event = portion.empty()
    #     # calculate AVM of intersecting part
    #     if not isinstance(other, UniformDistribution):
    #         raise NotImplementedError(f"Density between UniformDistribution and {type(other)} is not known.")
    #     intersection = self.interval.intersection(other.interval)
    #     own_weight = sum(own_node_weights.get(hash(self)))
    #     other_weight = sum(other_node_weights.get(hash(other)))
    #
    #     if not intersection.empty and self.pdf_value() * own_weight > other.pdf_value() * other_weight:
    #         #diff_of_pdf = self.pdf_value() - other.pdf_value()
    #         resulting_event = resulting_event.union(portion.closed(intersection.lower, intersection.upper))
    #     difference = self.interval.union(other.interval).difference(intersection)
    #     for interval in difference:
    #         if interval in self.interval:
    #             resulting_event = resulting_event.union(portion.closed(interval.lower, interval.upper))
    #     return Event({self.variable: resulting_event})

    def all_union_of_mixture_points_with(self, other: Self):
        points = SortedSet([self.interval.lower, self.interval.upper, other.interval.lower, other.interval.upper])
        result = [portion.open(lower, upper) for lower, upper in zip(points[:-1], points[1:])]
        return result

    def parameters(self):
        return {"variable": self.variable, "interval": self.interval}

