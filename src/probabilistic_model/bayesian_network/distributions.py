from random_events.events import Event, EncodedEvent
from typing_extensions import Tuple, Dict, Iterable, List

from .bayesian_network import BayesianNetworkMixin
from ..probabilistic_model import ProbabilisticModel
from random_events.variables import Discrete
from ..distributions.distributions import DiscreteDistribution


class RootDistribution(BayesianNetworkMixin, DiscreteDistribution):

    def forward_pass(self, event: EncodedEvent):
        self.forward_message, self.forward_probability = self._conditional(event)


class ConditionalProbabilityTable(BayesianNetworkMixin):

    variables: Tuple[Discrete, ...]
    conditional_probability_distributions: Dict[Tuple, DiscreteDistribution] = dict()

    def __init__(self, variable: Discrete):
        ProbabilisticModel.__init__(self, [variable])

    @property
    def variable(self) -> Discrete:
        return self.variables[0]

    def likelihood(self, event: Iterable) -> float:
        return self._likelihood([variable.encode(value) for variable, value in zip(self.parent_and_node_variables, event)])

    def _likelihood(self, event: Iterable) -> float:
        parent_event = tuple(event[:1])
        node_event = tuple(event[1:])
        return self.conditional_probability_distributions[parent_event]._likelihood(node_event)

    def __repr__(self):
        return f"P({self.variable.name}|{self.parent.variable.name})"

    def to_tabulate(self) -> List[List[str]]:
        table = [[self.parent.variable.name, self.variable.name, repr(self)]]
        for parent_event, distribution in self.conditional_probability_distributions.items():
            for event, probability in zip(self.variable.domain, distribution.weights):
                table.append([str(parent_event[0]), str(event), str(probability)])
        return table
