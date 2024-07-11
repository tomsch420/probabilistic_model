from __future__ import annotations

import inspect
from abc import abstractmethod, ABC
from functools import cached_property

import networkx as nx
import torch
import torch.nn as nn
from random_events.interval import SimpleInterval, Bound
from random_events.utils import recursive_subclasses
from random_events.variable import Continuous, Variable
from torch import nextafter
from typing_extensions import List, Tuple, Type, Dict, Union

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit, \
    ProbabilisticCircuitMixin, SumUnit, ProductUnit


def inverse_class_of(clazz: Type[ProbabilisticCircuitMixin]) -> Type[Layer]:
    for subclass in recursive_subclasses(Layer):
        if not inspect.isabstract(subclass):
            if issubclass(clazz, subclass.original_class()):
                return subclass

    raise TypeError(f"Could not find class for {clazz}")


def simple_interval_to_open_tensor(interval: SimpleInterval) -> torch.Tensor:
    """
    Convert a simple interval to a tensor where the first element is the lower bound as if it was open and the
    second is the upper bound as if it was open.

    :param interval: The interval to convert.
    :return: The tensor.
    """
    lower = torch.tensor(interval.lower)
    if interval.left == Bound.CLOSED:
        lower = nextafter(lower, lower - 1)
    upper = torch.tensor(interval.upper)
    if interval.right == Bound.CLOSED:
        upper = nextafter(upper, upper + 1)
    return torch.tensor([lower, upper])


class Layer(nn.Module):
    """
    Abstract class for Layers of a layered circuit.

    Layers have the same scope (set of variables) for every node in them.
    """

    @classmethod
    def original_class(cls) -> Tuple[Type, ...]:
        """
        The original class of the layer.
        """
        return tuple()

    @property
    @abstractmethod
    def variables(self) -> Tuple[Variable, ...]:
        """
        The variables of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        """
        Validate the parameters and their layouts.
        """
        raise NotImplementedError

    @property
    def number_of_nodes(self) -> int:
        """
        The number of nodes in the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log-likelihood of the distribution.

        .. note::
            The shape of the log likelihood depends on the number of samples and nodes.
            The shape of the result is (#samples, #nodes).
            The first dimension indexes the samples, the second the nodes.
        """
        raise NotImplementedError

    @staticmethod
    def from_probabilistic_circuit(pc: ProbabilisticCircuit) -> Layer:
        """
        Convert a probabilistic circuit to a layered circuit.
        The result expresses the same distribution as `pc`.

        :param pc: The probabilistic circuit.
        :return: The layered circuit.
        """

        node_to_depth_map = {node: nx.shortest_path_length(pc, pc.root, node) for node in pc.nodes}
        layer_to_nodes_map = {depth: [node for node, n_depth in node_to_depth_map.items() if depth == n_depth] for depth
                              in set(node_to_depth_map.values())}

        hash_remap = {}
        child_layers = []

        for layer_index, nodes in reversed(layer_to_nodes_map.items()):
            layers = Layer.create_layers_from_nodes(nodes, child_layers, hash_remap)

    @staticmethod
    def create_layers_from_nodes(nodes: List[ProbabilisticCircuitMixin], child_layers,
                                 hash_remap) -> List[Tuple[Layer, Dict[int, int]]]:
        """
        Create a layer from a list of nodes.
        """
        result = []

        unique_types = set(type(node) for node in nodes)
        for unique_type in unique_types:
            nodes_of_current_type = [node for node in nodes if isinstance(node, unique_type)]
            layer_type = inverse_class_of(unique_type)
            scopes = [tuple(node.variables) for node in nodes_of_current_type]
            unique_scopes = set(scopes)
            for scope in unique_scopes:
                nodes_of_current_type_and_scope = [node for node in nodes_of_current_type if
                                                   tuple(node.variables) == scope]
                result.append(layer_type.create_layer_from_nodes_with_same_type_and_scope(
                    nodes_of_current_type_and_scope, child_layers, hash_remap))

        return result

    @classmethod
    @abstractmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[ProbabilisticCircuitMixin],
                                                         child_layers: List[Layer],
                                                         hash_remap: Dict[int, int]) -> Tuple[Layer, Dict[int, int]]:
        """
        Create a layer from a list of nodes with the same type and scope.
        """
        raise NotImplementedError


class InnerLayer(Layer, ABC):
    """
    Class for inner layers
    """

    child_layers: List[Layer]
    """
    The child layers of this layer.
    """

    def __init__(self, child_layers: List[Layer]):
        super().__init__()
        self.child_layers = child_layers

    @cached_property
    def variables(self) -> Tuple[Variable, ...]:
        return tuple(sorted(set().union(*[child_layer.variables for child_layer in self.child_layers])))


class InputLayer(Layer, ABC):
    """
    Abstract base class for input units.
    Input layers should contain only one type of distribution such that the vectorization of the log likelihood
    calculation works without bottleneck statements like if/else or loops.
    """

    variable: Variable
    """
    The variable of the distributions.
    """

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return self.variable,


class UniformLayer(InputLayer):
    """
    A layer that represents a uniform distribution over a single variable.
    """

    variable: Continuous
    """
    The index of the variable that this layer represents.
    """

    lower: torch.Tensor
    """
    The lower bounds of the uniform distributions as a tensor of shape (self.number_of_nodes, ).
    The lower bounds are treated as open intervals.
    """

    upper: torch.Tensor
    """
    The upper bound of the uniform distributions as a tensor of shape (self.number_of_nodes, ).
    The upper bounds are treated as open intervals.
    """

    def __init__(self, variable: Continuous, lower: torch.Tensor, upper: torch.Tensor):
        """
        Initialize the uniform layer.

        :param variable: The variable that this layer represents.
        :param lower: The lower bounds of the uniform distributions.
        :param upper: The upper bounds of the uniform distributions.
        """
        super().__init__()
        self.variable = variable
        self.lower = lower
        self.upper = upper

    @classmethod
    def original_class(cls) -> Tuple[Type, ...]:
        return UniformDistribution,

    def validate(self):
        assert self.lower.shape == self.upper.shape, "The shapes of the lower and upper bounds must match."

    @property
    def number_of_nodes(self) -> int:
        """
        The number of nodes in the layer.
        """
        return len(self.lower)

    def log_pdf_value(self) -> torch.Tensor:
        """
        Calculate the log-density of the uniform distribution.
        """
        return -torch.log(self.upper - self.lower)

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log-likelihood of the uniform distribution.

        :param x: The input tensor of shape (n, 1).
        :return: The log-likelihood of the uniform distribution.
        """
        return torch.where((x > self.lower) & (x < self.upper), self.log_pdf_value(), -torch.inf)

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[UniformDistribution],
                                                         child_layers: List[Layer],
                                                         hash_remap: Dict[int, int]) \
            -> Tuple[Layer, Dict[int, int]]:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        bounds = torch.stack([simple_interval_to_open_tensor(node.interval) for node in nodes])
        result = cls(nodes[0].variable, bounds[:, 0], bounds[:, 1])
        return result, hash_remap


class SumLayer(InnerLayer):
    """
    A layer that represents the sum of multiple other layers.
    """

    child_layers: Union[List[[ProductLayer]], List[InputLayer]]

    log_weights: List[torch.Tensor]
    """
    The logarithmic weights of each edge.
    The list consists of tensor that are interpreted as weights for each child layer.
    
    The first dimension of each tensor must match the number of nodes of this layer and hence has to be 
    constant.
    The second dimension of each tensor must match the number of nodes of the  respective child layer.

    The weights are normalized per row.
    """

    def __init__(self, child_layers: List[Layer], log_weights: List[torch.Tensor]):
        """
        Initialize the sum layer.

        :param child_layers: The child layers of the sum layer.
        :param log_weights: The logarithmic weights of each edge.
        """
        super().__init__(child_layers)
        self.log_weights = log_weights

    def validate(self):
        for log_weights in self.log_weights:
            assert log_weights.shape[0] == self.number_of_nodes, "The number of nodes must match the number of weights."

        for log_weights, child_layer in self.log_weighted_child_layers:
            assert log_weights.shape[
                       1] == child_layer.number_of_nodes, "The number of nodes must match the number of weights."

    @classmethod
    def original_class(cls) -> Tuple[Type, ...]:
        return SumUnit,

    @property
    def log_weighted_child_layers(self):
        yield from zip(self.log_weights, self.child_layers)

    @property
    def concatenated_weights(self) -> torch.Tensor:
        return torch.cat(self.log_weights, dim=1)

    @property
    def log_normalization_constants(self) -> torch.Tensor:
        """
        :return: The normalization constants for every node in this sum layer.
        """
        return torch.logsumexp(self.concatenated_weights, dim=1)

    @property
    def number_of_nodes(self) -> int:
        """
        The number of nodes in the layer.
        """
        return self.log_weights[0].shape[0]

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(len(x), self.number_of_nodes)

        for log_weights, child_layer in self.log_weighted_child_layers:
            # get the log likelihoods of the child nodes
            ll = child_layer.log_likelihood(x)
            # assert ll.shape == (len(x), child_layer.number_of_nodes)

            # expand the log likelihood of the child nodes to the number of nodes in this layer, i.e.
            # (#x, #child_nodes, #nodes)
            ll = ll.unsqueeze(-1).repeat(1, 1, self.number_of_nodes)
            # assert ll.shape == (len(x), child_layer.number_of_nodes, self.number_of_nodes)

            # weight the log likelihood of the child nodes by the weight for each node of this layer
            ll = torch.exp(ll + log_weights.T).sum(dim=1)

            # sum the child layer result
            result += ll

        return torch.log(result) - self.log_normalization_constants.repeat(len(x), 1)

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[SumUnit],
                                                         child_layers: List[Layer],
                                                         hash_remap: Dict[int, int]) -> Tuple[Layer, Dict[int, int]]:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        log_weights = torch.stack([torch.tensor([weight for weight, _ in node.weighted_subcircuits]) for node in nodes])
        print(log_weights)


class ProductLayer(InnerLayer):
    """
    A layer that represents the product of multiple other units.

    Every node in the layer has the same partitioning of variables.
    The n-th child layer has the variables described in the n-th partition.
    """

    child_layers: List[Union[SumLayer, InputLayer]]

    edges: List[torch.Tensor]
    """
    The edges consist of a list of tensors containing integers.
    The outer list describes the edges for each child layer.
    The tensors in the list describe the edges for each node in the child layer.
    The integers are interpreted in such a way that n-th value represents a edge (n, edges[n]).
    
    Due to decomposability and smoothness every product node in this layer has to map to exactly one node in each
    child layer. Nodes in the child layer can be mapped to by multiple nodes in this layer.
    """

    def __init__(self, child_layers: List[Layer], edges: List[torch.Tensor]):
        """
        Initialize the product layer.

        :param child_layers: The child layers of the product layer.
        :param edges: The edges of the product layer.
        """
        super().__init__(child_layers)
        self.edges = edges

    def validate(self):
        for edges, child_layer in zip(self.edges, self.child_layers):
            assert len(edges) == self.number_of_nodes, "The number of nodes must match the number of edges."

    @classmethod
    def original_class(cls) -> Tuple[Type, ...]:
        return ProductUnit,

    @property
    def number_of_nodes(self) -> int:
        return self.edges[0].shape[0]

    @cached_property
    def columns_of_child_layers(self) -> Tuple[Tuple[int, ...], ...]:
        result = []
        for layer in self.child_layers:
            layer_indices = [self.variables.index(variable) for variable in layer.variables]
            result.append(tuple(layer_indices))
        return tuple(result)

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(len(x), self.number_of_nodes)
        for columns, edges, layer in zip(self.columns_of_child_layers, self.edges, self.child_layers):
            # calculate the log likelihood over the columns of the child layer
            ll = layer.log_likelihood(x[:, columns])

            # rearrange the log likelihood to match the edges
            ll = ll[:, edges]  # shape: (#x, #nodes)
            # assert ll.shape == (len(x), self.number_of_nodes)

            result += ll
        return result
