from typing import Tuple, Optional
from typing_extensions import Self

import portion
from random_events.events import EncodedEvent
from random_events.variables import Continuous

from ...distributions.uniform import UniformDistribution as PMUniformDistribution


class UniformDistribution(PMUniformDistribution):

    def __init__(self, variable: Continuous, interval: portion.Interval):
        super().__init__(variable, interval)