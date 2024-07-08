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
from torch import nn
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


def convert_node(node: ProbabilisticCircuitMixin, variable_index_map: Dict[Variable, int]):
    """
    Convert a node to a tensor circuit element.
    """

    if isinstance(node, SumUnit):
        subcircuits = [convert_node(subcircuit, variable_index_map) for subcircuit in node.subcircuits]
        result = pyjuice.summate(*subcircuits, num_nodes=1)
        result.set_params(torch.tensor([weight for weight, _ in node.weighted_subcircuits]))

    elif isinstance(node, GaussianDistribution):
        result = pyjuice.inputs(variable_index_map[node.variable], num_nodes=1,
                                dist=dists.gaussian.Gaussian(node.location, node.scale**2))
    else:
        raise TypeError(f"Could not convert node {node}")

    return result


class TensorProbabilisticCircuit(ProbabilisticModel):
    variable_to_index_map: Dict[Variable, int]
    tensor_circuit: nn.Module

    @classmethod
    def from_pc(cls, pc: ProbabilisticCircuit) -> Self:
        result = cls()
        result.variable_to_index_map = {var: i for i, var in enumerate(pc.variables)}
        tensor_circuit = convert_node(pc.root, result.variable_to_index_map)
        cls.tensor_circuit = pyjuice.compile(pyjuice.merge(tensor_circuit))
        return result

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
