from collections import UserDict
from typing import Tuple, Optional, Union, List

import networkx as nx
import numpy as np

import pyjuice
import pyjuice.nodes.distributions as dists
import torch
from pyjuice.model.tensorcircuit import SumLayer, TensorCircuit, ProdLayer

from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable
from typing_extensions import Dict, Self

from ..distributions.gaussian import GaussianDistribution
from ..probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit, SumUnit, ProductUnit, \
    ProbabilisticCircuitMixin
from ..distributions.distributions import UnivariateDistribution
from ..probabilistic_model import ProbabilisticModel


class ClassConverter(UserDict):
    """
    Maps classes to their corresponding classes in the pyjuice library.
    """

    def __getitem__(self, item):
        for key in self.data:
            if issubclass(item, key):
                return self.data[key]
        else:
            raise KeyError(f"Could not find class for {item}")


class_map = ClassConverter({SumUnit: SumLayer, ProductUnit: ProdLayer, GaussianDistribution: dists.gaussian.Gaussian})


class TensorProbabilisticCircuit(ProbabilisticModel):
    variable_to_index_map: Dict[Variable, int]
    tensor_circuit: TensorCircuit

    @classmethod
    def from_pc(cls, pc: ProbabilisticCircuit) -> Self:
        result = cls()
        result.variable_to_index_map = {var: i for i, var in enumerate(pc.variables)}

        all_node_depths: Dict[ProbabilisticCircuitMixin, int] = nx.shortest_path_length(pc, source=pc.root)
        total_depth = max(all_node_depths.values())
        # for every depth in reversed order, we create a layer
        layers = []
        for depth in reversed(range(total_depth + 1)):
            # get all nodes at this depth
            nodes = [node for node, d in all_node_depths.items() if d == depth]
            result.construct_layers_from_nodes(nodes)
        return cls()

    def construct_layers_from_nodes(self, nodes: List[ProbabilisticCircuitMixin]):

        # filter univariate distributions
        nodes = [node for node in nodes if isinstance(node, UnivariateDistribution)]
        input_layers = self.construct_input_layer_as_sum(nodes)
        exit()
        # construct maps that hold information about the nodes
        node_types = {index: class_map[type(node)] for index, node in enumerate(nodes)}
        inverse_node_types = {}
        for clazz in set(node_types.values()):
            inverse_node_types[clazz] = [index for index, value in node_types.items() if value == clazz]
            if issubclass(clazz, UnivariateDistribution):
                self.construct_input_layers(nodes)

    def construct_input_layer_as_sum(self, nodes: List[UnivariateDistribution]):
        variables = list(set(node.variable for node in nodes))

        variable_distribution_map = {var: [node for node in nodes if node.variable == var] for var in variables}

        layers = []
        for variable, distributions in variable_distribution_map.items():
            distribution_types = {class_map[type(distribution)] for distribution in distributions}
            layer = self.construct_input_layer_variable(variable, distributions)

    def construct_input_layer_variable(self, variable: Variable, distributions: List[UnivariateDistribution]):
        distribution_type = class_map[type(distributions[0])]

        if distribution_type is dists.gaussian.Gaussian:
            distribution = dists.gaussian.Gaussian()
            params = torch.tensor([[d.location, d.scale**2] for d in distributions])
            params = distribution.init_parameters(len(distributions), params=params.flatten())
            return pyjuice.inputs(var=self.variable_to_index_map[variable],
                                  num_nodes=len(distributions), dist=distribution, params=params)
        else:
            raise NotImplementedError(f"Cannot handle distribution type {distribution_type}")

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return tuple(self.variable_to_index_map.keys())

    def support(self) -> Event:
        pass

    def log_likelihood(self, events: np.array) -> np.array:
        pass

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        pass

    def log_mode(self) -> Tuple[Event, float]:
        pass

    def log_conditional(self, event: Event) -> Tuple[Optional[Union[ProbabilisticModel, Self]], float]:
        pass

    def sample(self, amount: int) -> np.array:
        pass
