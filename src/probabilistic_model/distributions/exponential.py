import math
from typing import Union, Tuple, Iterable

import numpy as np
import portion
from random_events.events import ComplexEvent, Event, VariableMap
from random_events.variables import Continuous, Integer

from .distributions import ContinuousDistribution
from ..probabilistic_model import MomentType, OrderType, CenterType


class ExponentialDistribution(ContinuousDistribution):
    """
    Shifted Exponential Distribution.
    """

    location: float
    """
    The location (shift) of the exponential distribution
    """

    scale: float
    """
    The scale of the exponential distribution
    """

    def __init__(self, variable: Continuous, location: float, scale: float):
        super().__init__(variable)
        self.location = location
        self.scale = scale

    @property
    def domain(self) -> ComplexEvent:
        return ComplexEvent([Event({self.variable: self.interval})])

    @property
    def interval(self):
        return portion.closedopen(self.location, float("inf"))

    @property
    def rate(self):
        return 1/self.scale

    def _pdf(self, value: Union[float, int]) -> float:
        if value not in self.interval:
            return 0
        return self.rate * np.exp(-self.rate * (value - self.location))

    def _cdf(self, value: float) -> float:
        if value not in self.interval:
            return 0
        return 1 - np.exp(-self.rate * (value - self.location))

    def _mode(self) -> Tuple[ComplexEvent, float]:
        likelihood = self._pdf(self.location)
        event = ComplexEvent([Event({self.variable: self.location})])
        return event, likelihood

    @property
    def representation(self) -> str:
        return f"exp({self.location, self.scale})"

    def raw_moment(self, order: int):
        return math.factorial(order) / (self.rate**order)

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        order = order[self.variable]
        center = center[self.variable]

        if order == 0:
            moment = 1.

        elif order == 1:
            moment = self.raw_moment(1) - center

        elif order == 2:
            moment = self.raw_moment(2) + center**2 - 2 * self.raw_moment(1) * center
        else:
            raise NotImplementedError("Moments above order 2 are not supported.")

        return VariableMap({self.variable: moment})

    def sample(self, amount: int) -> Iterable:
        return np.random.exponential(self.scale, (amount, 1)) + self.location
