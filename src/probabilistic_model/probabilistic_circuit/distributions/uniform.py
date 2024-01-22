import random
from typing import List, Tuple, Optional, Union, Dict, Any

import random_events.variables
from plotly import graph_objects as go
from typing_extensions import Self

import portion
from random_events.events import Event, EncodedEvent, VariableMap
from random_events.variables import Continuous, Variable

from ...probabilistic_circuit import ContinuousDistribution, DeterministicSumUnit, Unit
from ...probabilistic_model import OrderType, CenterType, MomentType
from ...distributions.uniform import UniformDistribution as PMUniformDistribution


class UniformDistribution(PMUniformDistribution, ContinuousDistribution):
    """
    Class for uniform distributions over the half-open interval [lower, upper).
    """

    def __init__(self, variable: Continuous, interval: portion.Interval, parent=None):
        super().__init__(variable, interval)
        ContinuousDistribution.__init__(self, variable, parent=parent)


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

    def to_json(self) -> Dict[str, Any]:
        return {**ContinuousDistribution.to_json(self), "interval": portion.to_data(self.interval)}

    @classmethod
    def from_json_with_variables_and_children(cls, data: Dict[str, Any], variables: List[Variable],
                                              children: List['Unit']) -> Self:
        variable = random_events.variables.Variable.from_json(data["variable"])
        return cls(variable, portion.from_data(data["interval"]))
