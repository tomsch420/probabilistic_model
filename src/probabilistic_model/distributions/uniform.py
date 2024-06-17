import numpy as np
import os
import random
from typing import Tuple, Optional

from random_events.product_algebra import SimpleInterval

from .distributions import *
from ..constants import PADDING_FACTOR_FOR_X_AXIS_IN_PLOT
from typing_extensions import List, Dict, Any, Self
from plotly import graph_objects as go

class UniformDistribution(ContinuousDistributionWithFiniteSupport):
    """
    Class for uniform distributions over the half-open interval [lower, upper).
    """

    def __init__(self, variable: Continuous, interval: SimpleInterval):
        super().__init__()
        self.variable = variable
        self.interval = interval

    def log_likelihood_without_bounds_check(self, x: np.array) -> np.array:
        return np.full((len(x),), self.log_pdf_value())

    def cdf(self, x: np.array) -> np.array:
        result = (x - self.lower) / (self.upper - self.lower)
        result = np.minimum(1, np.maximum(0, result))
        return result

    def univariate_log_mode(self) -> Tuple[AbstractCompositeSet, float]:
        return self.interval.as_composite_set(), self.log_pdf_value()

    def log_conditional_from_non_singleton_simple_interval(self, interval: SimpleInterval) -> Tuple[Self, float]:
        probability = self.cdf(interval.upper) - self.cdf(interval.lower)
        return self.__class__(self.variable, interval), np.log(probability)

    def sample(self, amount: int) -> np.array:
        return np.random.uniform(self.lower, self.upper, (amount, 1))

    def pdf_value(self) -> float:
        """
        Calculate the density of the uniform distribution.
        """
        return np.exp(self.log_pdf_value())

    def log_pdf_value(self) -> float:
        """
        Calculate the log-density of the uniform distribution.
        """
        return -np.log(self.upper - self.lower)

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
        return f"U({self.variable.name} | {self.interval})"

    @property
    def image(self):
        return os.path.join(os.path.dirname(__file__),"../../../", "resources", "icons", "defaultIcon.png")

    def __copy__(self):
        return self.__class__(self.variable, self.interval)

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "interval": self.interval.to_json()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variable = Continuous.from_json(data["variable"])
        interval = SimpleInterval.from_json(data["interval"])
        return cls(variable, interval)

    def x_axis_points_for_plotly(self) -> List[Union[None, float]]:
        interval_size = self.upper - self.lower
        x = [self.lower - interval_size * PADDING_FACTOR_FOR_X_AXIS_IN_PLOT, self.lower, None,
             self.lower, self.upper, None,
             self.upper, self.upper + interval_size * PADDING_FACTOR_FOR_X_AXIS_IN_PLOT]
        return x

    def pdf_trace(self) -> go.Scatter:
        pdf_values = [0, 0, None, self.pdf_value(), self.pdf_value(), None, 0, 0]
        pdf_trace = go.Scatter(x=self.x_axis_points_for_plotly(),
                               y=pdf_values, mode='lines', name="Probability Density Function")
        return pdf_trace

    def cdf_trace(self) -> go.Scatter:
        x = self.x_axis_points_for_plotly()
        cdf_values = [value if value is None else self.cdf(value) for value in x]
        cdf_trace = go.Scatter(x=x, y=cdf_values, mode='lines', name="Cumulative Distribution Function")
        return cdf_trace

    def plot(self, **kwargs) -> List:
        pdf_trace = self.pdf_trace()
        cdf_trace = self.cdf_trace()

        height = self.pdf_value() * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT

        mode_trace = (go.Scatter(x=[self.lower, self.lower, self.upper, self.upper],
                                 y=[0, height, height, 0], mode='lines+markers',
                                 name="Mode", fill="toself"))

        expectation = self.expectation([self.variable])[self.variable]
        expectation_trace = (
            go.Scatter(x=[expectation, expectation], y=[0, height], mode='lines+markers',
                       name="Expectation"))
        return [pdf_trace, cdf_trace, mode_trace, expectation_trace]

    def __hash__(self):
        return hash((self.variable.name, hash(self.interval)))



    def all_union_of_mixture_points_with(self, other: Self):
        points = SortedSet([self.interval.lower, self.interval.upper, other.interval.lower, other.interval.upper])
        result = [SimpleInterval(lower, upper) for lower, upper in zip(points[:-1], points[1:])]
        return result
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
        return distance/2