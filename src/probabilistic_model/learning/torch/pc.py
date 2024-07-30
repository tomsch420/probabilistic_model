from __future__ import annotations

import inspect
import math
from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import networkx as nx
import torch
import torch.nn as nn
from random_events.product_algebra import Event, SimpleEvent
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.utils import recursive_subclasses
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import List, Tuple, Type, Dict, Union, Self

from ...probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit, \
    ProbabilisticCircuitMixin, SumUnit, ProductUnit
from ...probabilistic_model import ProbabilisticModel


def inverse_class_of(clazz: Type[ProbabilisticCircuitMixin]) -> Type[Layer]:
    for subclass in recursive_subclasses(Layer):
        if not inspect.isabstract(subclass):
            if issubclass(clazz, subclass.original_class()):
                return subclass

    raise TypeError(f"Could not find class for {clazz}")


class Layer(nn.Module, ProbabilisticModel):
    """
    Abstract class for Layers of a layered circuit.

    Layers have the same scope (set of variables) for every node in them.
    """

    @property
    def support(self) -> Event:
        if self.number_of_nodes == 1:
            return self.support_per_node[0]
        raise ValueError("The support is only defined for layers with one node. Use the support_per_node property "
                         "if you want the support of each node.")

    @property
    @abstractmethod
    def support_per_node(self) -> List[Event]:
        raise NotImplementedError

    @property
    def deterministic(self) -> bool:
        raise NotImplementedError

    @property
    def is_root(self) -> bool:
        return self.number_of_nodes == 1

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

    def probability_of_simple_event(self, event: SimpleEvent) -> torch.Tensor:
        """
        Calculate the probability of a simple event.
        :param event: The event
        :return: The probability of the event for each node in the layer as shape (#nodes, 1).
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

        child_layers = []

        for layer_index, nodes in reversed(layer_to_nodes_map.items()):
            child_layers = Layer.create_layers_from_nodes(nodes, child_layers)
        return child_layers[0].layer

    @staticmethod
    def create_layers_from_nodes(nodes: List[ProbabilisticCircuitMixin], child_layers: List[AnnotatedLayer]) \
            -> List[AnnotatedLayer]:
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
                layer = layer_type.create_layer_from_nodes_with_same_type_and_scope(nodes_of_current_type_and_scope,
                                                                                    child_layers)
                result.append(layer)

        return result

    @classmethod
    @abstractmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[ProbabilisticCircuitMixin],
                                                         child_layers: List[AnnotatedLayer]) -> \
            AnnotatedLayer:
        """
        Create a layer from a list of nodes with the same type and scope.
        """
        raise NotImplementedError

    def log_conditional(self, event: Event) -> Tuple[Optional[Layer], float]:

        # skip trivial case
        if event.is_empty():
            return None, -torch.inf

        # if the event is easy, don't create a proxy node
        elif len(event.simple_sets) == 1:
            conditional, log_probability = self.log_conditional_of_simple_event(event.simple_sets[0])
            return conditional, log_probability.item()

        raise NotImplementedError

    @abstractmethod
    def log_conditional_of_simple_event(self, event: SimpleEvent) -> Tuple[Optional[Self], torch.Tensor]:
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
    Abstract base class for univariate input units.
    Input layers should contain only one type of distribution such that the vectorization of the log likelihood
    calculation works without bottleneck statements like if/else or loops.
    """

    variable: Variable
    """
    The variable of the distributions.
    """

    def __init__(self, variable: Variable):
        super().__init__()
        self.variable = variable

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return self.variable,

    @property
    def support_per_node(self) -> List[Event]:
        return [SimpleEvent({self.variable: us}).as_composite_set()
                for us in self.univariate_support_per_node]

    @property
    @abstractmethod
    def univariate_support_per_node(self) -> List[AbstractCompositeSet]:
        """
        The univariate support of the layer for each node.
        """
        raise NotImplementedError


class SumLayer(InnerLayer):
    """
    A layer that represents the sum of multiple other layers.
    """

    child_layers: Union[List[[ProductLayer]], List[InputLayer]]

    log_weights: List[torch.Tensor]
    """
    The (sparse) logarithmic weights of each edge.
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
    def log_weighted_child_layers(self) -> List[Tuple[torch.Tensor, Union[ProductLayer, InputLayer]]]:
        yield from zip(self.log_weights, self.child_layers)

    @property
    def concatenated_weights(self) -> torch.Tensor:
        return torch.cat(self.log_weights, dim=1)

    @property
    def log_normalization_constants(self) -> torch.Tensor:
        """
        :return: The normalization constants for every node in this sum layer.
        """
        if self.concatenated_weights.is_sparse:
            result = self.concatenated_weights.clone().coalesce()
            result.values().exp_()
            result = result.sum(1)
            result.values().log_()
            return result.to_dense()
        else:
            return torch.logsumexp(self.concatenated_weights, dim=1)

    @property
    def number_of_nodes(self) -> int:
        """
        The number of nodes in the layer.
        """
        return self.log_weights[0].shape[0]

    @property
    def support_per_node(self) -> List[Event]:
        pass

    def log_conditional_of_simple_event(self, event: SimpleEvent) -> Tuple[Optional[Self], torch.Tensor]:
        pass

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(len(x), self.number_of_nodes)

        for log_weights, child_layer in self.log_weighted_child_layers:
            # get the log likelihoods of the child nodes
            ll = child_layer.log_likelihood(x)
            # assert ll.shape == (len(x), child_layer.number_of_nodes)

            # weight the log likelihood of the child nodes by the weight for each node of this layer
            if log_weights.is_sparse:
                cloned_log_weights = log_weights.clone()  # clone the weights
                cloned_log_weights.values().exp_()  # exponent weights
                ll = ll.exp()  # calculate the exponential of the child log likelihoods

                #  calculate the weighted sum in layer
                ll = torch.matmul(ll, cloned_log_weights.T)
            else:
                # expand the log likelihood of the child nodes to the number of nodes in this layer, i.e.
                # (#x, #child_nodes, #nodes)
                ll = ll.unsqueeze(-1)
                # assert ll.shape == (len(x), child_layer.number_of_nodes, 1)

                # (#x, #child_nodes, #number_of_nodes)
                ll = log_weights.T + ll
                ll = torch.exp(ll).sum(dim=1)

            # sum the child layer result
            result += ll

        return torch.log(result) - self.log_normalization_constants

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[SumUnit],
                                                         child_layers: List[AnnotatedLayer]) -> \
            AnnotatedLayer:

        result_hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        variables = tuple(nodes[0].variables)
        number_of_nodes = len(nodes)

        # filter the child layers to only contain layers with the same scope as this one
        filtered_child_layers = [child_layer for child_layer in child_layers if tuple(child_layer.layer.variables) ==
                                 variables]
        log_weights = []

        for child_layer in filtered_child_layers:

            indices = []
            values = []

            for index, node in enumerate(nodes):
                for weight, subcircuit in node.weighted_subcircuits:
                    if hash(subcircuit) in child_layer.hash_remap:
                        indices.append((index, child_layer.hash_remap[hash(subcircuit)]))
                        values.append(math.log(weight))
                        # values.append(weight)

            log_weights.append(torch.sparse_coo_tensor(torch.tensor(indices).T,
                                                       torch.tensor(values),
                                                       (number_of_nodes, child_layer.layer.number_of_nodes),
                                                       is_coalesced=True).double())

        sum_layer = cls([cl.layer for cl in filtered_child_layers], log_weights)
        return AnnotatedLayer(sum_layer, nodes, result_hash_remap)

    @property
    def support(self) -> Event:
        raise NotImplementedError

    def probability_of_simple_event(self, event: SimpleEvent) -> torch.Tensor:
        result = torch.zeros(self.number_of_nodes, 1)
        for log_weights, child_layer in self.log_weighted_child_layers:
            child_layer_prob = child_layer.probability_of_simple_event(event)  # shape: (#child_nodes, 1)
            weights = (torch.exp(log_weights - self.log_normalization_constants.unsqueeze(-1)).
                       to(child_layer_prob.dtype))  # shape: (#nodes, #child_nodes)
            probabilities = torch.matmul(weights, child_layer_prob)
            result += probabilities
        return result

    def log_mode(self) -> Tuple[Event, float]:
        pass

    def log_conditional(self, event: Event) -> Tuple[Optional[Union[ProbabilisticModel, Self]], float]:
        pass

    def sample(self, amount: int) -> torch.Tensor:
        pass


class ProductLayer(InnerLayer):
    """
    A layer that represents the product of multiple other units.

    Every node in the layer has the same partitioning of variables.
    The n-th child layer has the variables described in the n-th partition.
    """

    child_layers: List[Union[SumLayer, InputLayer]]
    """
    The child of a product layer is a list that contains groups sum units with the same scope or groups of input
    units with the same scope.
    """

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

    # @torch.compile
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

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[ProductUnit],
                                                         child_layers: List[AnnotatedLayer]) -> \
            AnnotatedLayer:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        number_of_nodes = len(nodes)

        edges = [torch.full((child_layer.layer.number_of_nodes,), torch.nan).int() for child_layer in child_layers]

        for index, node in enumerate(nodes):
            for child_layer in child_layers:
                cl_variables = SortedSet(child_layer.layer.variables)
                for subcircuit_index, subcircuit in enumerate(node.subcircuits):
                    if cl_variables == subcircuit.variables:
                        edges[subcircuit_index][index] = child_layer.hash_remap[hash(subcircuit)]
        layer = cls([cl.layer for cl in child_layers], edges)
        return AnnotatedLayer(layer, nodes, hash_remap)

    def probability_of_simple_event(self, event: SimpleEvent) -> torch.Tensor:
        result = torch.ones(self.number_of_nodes, 1)
        for edges, layer in zip(self.edges, self.child_layers):
            child_layer_prob = layer.probability_of_simple_event(event)  # shape: (#child_nodes, 1)
            probabilities = child_layer_prob[edges]  # shape: (#nodes, 1)
            result *= probabilities
        return result

    def log_mode(self) -> Tuple[Event, float]:
        pass

    def log_conditional(self, event: Event) -> Tuple[Optional[Union[ProbabilisticModel, Self]], float]:
        pass

    def sample(self, amount: int) -> torch.Tensor:
        pass

    @property
    def support_per_node(self) -> List[Event]:
        pass

    def log_conditional_of_simple_event(self, event: SimpleEvent) -> Tuple[Optional[Self], torch.Tensor]:
        pass



@dataclass
class AnnotatedLayer:
    layer: Layer
    nodes: List[ProbabilisticCircuitMixin]
    hash_remap: Dict[int, int]
