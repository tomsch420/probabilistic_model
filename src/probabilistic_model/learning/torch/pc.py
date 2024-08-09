from __future__ import annotations

import inspect
import math
from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Iterator

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
from ...utils import (remove_rows_and_cols_where_all, add_sparse_edges_dense_child_tensor_inplace,
                      sparse_remove_rows_and_cols_where_all, shrink_index_tensor,
                      embed_sparse_tensors_in_new_sparse_tensor)


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

    def mergeable_with(self, other: Layer):
        return self.variables == other.variables and type(self) == type(other)

    @property
    def support(self) -> Event:
        if self.number_of_nodes == 1:
            return self.support_per_node[0]
        raise ValueError("The support is only defined for layers with one node. Use the support_per_node property "
                         "if you want the support of each node.")

    @property
    @abstractmethod
    def support_per_node(self) -> List[Event]:
        """
        :return: The support of each node in the layer as a list of events.
        """
        raise NotImplementedError

    @property
    def deterministic(self) -> torch.BoolTensor:
        """
        :return: Rather, the layer is deterministic for each node as a boolean tensor.
        """
        raise NotImplementedError

    @property
    def is_root(self) -> bool:
        """
        TODO there could be multiple circuits with one node that are not the root circuit
        :return: Whether the layer is the root layer of the circuit.
        """
        return self.number_of_nodes == 1

    @classmethod
    def original_class(cls) -> Tuple[Type, ...]:
        """
        :return: The tuple of matching classes of the layer in the probabilistic_model.probabilistic_circuit package.
        """
        return tuple()

    @property
    @abstractmethod
    def variables(self) -> Tuple[Variable, ...]:
        """
        :return: The variables of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        """
        Validate the parameters and their layouts.
        """
        raise NotImplementedError

    @property
    @abstractmethod
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

    def probability_of_simple_event(self, event: SimpleEvent) -> torch.DoubleTensor:
        """
        Calculate the probability of a simple event.
        :param event: The simple event.
        :return: The probability of the event for each node in the layer.
        The result is a double tensor with shape (#nodes,).
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
        if event.is_empty():
            conditional, log_prob = self.impossible_condition_result
        elif len(event.simple_sets) == 1:
            conditional, log_prob = self.log_conditional_of_simple_event(event.simple_sets[0])
        else:
            conditional, log_prob = self.log_conditional_of_composite_event(event)
        return conditional, log_prob.item()

    def log_conditional_of_composite_event(self, event: Event) -> Tuple[Optional[Layer], torch.Tensor]:
        """
        :param event: The composite event.
        :return: The conditional distribution and the log probability of the event.
        The log probability is a double tensor with shape (#nodes,).
        """
        # get conditionals of each simple event
        results = [self.log_conditional_of_simple_event(simple_event) for simple_event in event.simple_sets]

        # filter valid input layers
        possible_layers = [(layer, ll) for layer, ll in results if layer is not None]

        # if the result is impossible, return it
        if len(possible_layers) == 0:
            return self.impossible_condition_result

        # create new log weights and new layers inefficiently TODO: some merging of layers if possible
        log_weights = [torch.sparse_coo_tensor(torch.tensor([[0], [0]]), ll) for _, ll in possible_layers]
        layers = [layer for layer, _ in possible_layers]
        # construct result
        resulting_layer = SumLayer(layers, log_weights)

        # calculate log probability
        log_prob = resulting_layer.log_normalization_constants

        return resulting_layer, log_prob

    @abstractmethod
    def log_conditional_of_simple_event(self, event: SimpleEvent) -> Tuple[Optional[Layer], torch.Tensor]:
        """
        :param event: The simple event.
        :return: The conditional distribution and the log probability of the event.
        The log probability is a double tensor with shape (#nodes,).
        """
        raise NotImplementedError

    @abstractmethod
    def merge_with(self, others: List[Self]):
        """
        Merge this layer with another layer inplace.

        :param others: The other layers
        """
        raise NotImplementedError

    @property
    def impossible_condition_result(self) -> Tuple[None, torch.Tensor]:
        """
        :return: The result that a layer yields if it is conditioned on an event E with P(E) = 0
        """
        return None, torch.full((self.number_of_nodes,), -torch.inf, dtype=torch.double)

    @abstractmethod
    def remove_nodes_inplace(self, remove_mask: torch.BoolTensor):
        """
        Remove nodes from the layer inplace.

        Also updates possible child layers if needed.

        :param remove_mask: The mask that indicates which nodes to remove.
        True indicates that the node should be removed.
        """
        raise NotImplementedError

    def __deepcopy__(self):
        raise NotImplementedError


class InnerLayer(Layer, ABC):
    """
    Abstract Base Class for inner layers
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

    Input layers contain only one type of distribution such that the vectorization of the log likelihood
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


class SumLayer(InnerLayer, ABC):
    """
    A layer that represents the sum of multiple other layers.
    """

    child_layers: Union[List[[ProductLayer]], List[InputLayer]]
    """
    Child layers of the sum unit. 
            
    Smoothness Requires that all sums go over the same scope. Hence the child layers have to have the same scope.
    
    In a smooth sum unit the child layers are 
        - product layers if the sum is multivariate
        - input layers if the sum is univariate
    """

    log_weights: List[torch.Tensor]
    """
    The sparse logarithmic weights of each edge.
    The list consists of tensor that are interpreted as weights for each child layer.
    
    The first dimension of each tensor must match the number of nodes of this layer and hence has to be 
    constant.
    The second dimension of each tensor must match the number of nodes of the  respective child layer.
    
    The weights are normalized per row.
    Each weight tensor is of type double.
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
    def log_weighted_child_layers(self) -> Iterator[Tuple[torch.Tensor, Union[ProductLayer, InputLayer]]]:
        """
        :returns: Yields log weights and the child layers zipped together.
        """
        yield from zip(self.log_weights, self.child_layers)

    @property
    def concatenated_weights(self) -> torch.Tensor:
        """
        :return: The concatenated weights of the child layers for each node.
        """
        return torch.cat(self.log_weights, dim=1)

    @property
    def number_of_nodes(self) -> int:
        return self.log_weights[0].shape[0]

    @property
    def support_per_node(self) -> List[Event]:
        pass

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
        raise NotImplementedError

    def sample(self, amount: int) -> torch.Tensor:
        raise NotImplementedError

    def __deepcopy__(self):
        child_layers = [child_layer.__deepcopy__() for child_layer in self.child_layers]
        log_weights = [log_weight.clone() for log_weight in self.log_weights]
        return self.__class__(child_layers, log_weights)

    def mergeable_layer_matrix(self, other: Self) -> torch.Tensor:
        """
        Create a matrix that describes which child layers of this layer can be merged with which child layers of the
        other layer.

        The entry [i, j] is True if the i-th child layer of this layer can be merged with the j-th child layer of the
        other layer.

        :param other: The other layer.
        :return: The mergeable boolean matrix
        """

        # create matrix describing mergeable layers
        mergeable_matrix = torch.zeros((len(self.child_layers), len(other.child_layers)), dtype=torch.bool)

        # fill matrix
        for i, layer in enumerate(self.child_layers):
            for j, other_layer in enumerate(other.child_layers):
                mergeable_matrix[i, j] = layer.mergeable_with(other_layer)
        return mergeable_matrix

    def merge_with_one_layer_inplace(self, other: Self):
        """
        Merge this layer with another layer inplace.

        :param other: The other layer

        """

        mergeable_matrix = self.mergeable_layer_matrix(other)
        assert (mergeable_matrix.sum(dim=1) <= 1).all(), "There must be at most one mergeable layer per layer."

        total_number_of_nodes = self.number_of_nodes + other.number_of_nodes
        new_layers = []
        new_log_weights = []

        for log_weights, layer, mergeable_row in zip(self.log_weights, self.child_layers, mergeable_matrix):

            # if this layer cant be merged with any other layer
            if mergeable_row.sum() == 0:

                # append impossible log weights to match the new number of nodes
                log_weights = torch.sparse_coo_tensor(log_weights.indices(), log_weights.values(),
                                                      (total_number_of_nodes, layer.number_of_nodes),
                                                      dtype=torch.double, is_coalesced=True)

                new_layers.append(layer)
                new_log_weights.append(log_weights)
                continue

            # filter for the mergeable layers
            mergeable_other_layers = [other_layer for other_layer, mergeable
                                      in zip(other.child_layers, mergeable_row) if mergeable]

            # filter for the mergeable log_weights
            mergeable_log_weights = [log_weights] + [other_log_weights for other_log_weights, mergeable
                                                     in zip(other.log_weights, mergeable_row) if mergeable]

            # merge the log_weights
            embedded_log_weights = embed_sparse_tensors_in_new_sparse_tensor(mergeable_log_weights)
            new_log_weights.append(embedded_log_weights)

            # merge the layers
            layer.merge_with(mergeable_other_layers)
            new_layers.append(layer)

        # check if any child_layer from the other layer has not been merged
        for other_log_weights, other_layer, mergeable_col in zip(other.log_weights, other.child_layers,
                                                                 mergeable_matrix.T):
            if mergeable_col.sum() == 0:

                # append impossible log weights to match the new number of nodes
                other_log_weights = torch.sparse_coo_tensor(other_log_weights.indices(), other_log_weights.values(),
                                                            (total_number_of_nodes, other_layer.number_of_nodes),
                                                            dtype=torch.double, is_coalesced=True)
                new_log_weights.append(other_log_weights)
                new_layers.append(other_layer)

        self.log_weights = new_log_weights
        self.child_layers = new_layers

    def merge_with(self, others: List[Self]):
        [self.merge_with_one_layer_inplace(other) for other in others]

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(len(x), self.number_of_nodes, dtype=torch.double)

        for log_weights, child_layer in self.log_weighted_child_layers:
            # get the log likelihoods of the child nodes
            ll = child_layer.log_likelihood(x)
            # assert ll.shape == (len(x), child_layer.number_of_nodes)

            # weight the log likelihood of the child nodes by the weight for each node of this layer
            cloned_log_weights = log_weights.clone()  # clone the weights
            cloned_log_weights.values().exp_()  # exponent weights
            ll = ll.exp()  # calculate the exponential of the child log likelihoods
            #  calculate the weighted sum in layer
            ll = torch.matmul(ll, cloned_log_weights.T)

            # sum the child layer result
            result += ll

        return torch.log(result) - self.log_normalization_constants

    def remove_nodes_inplace(self, remove_mask: torch.BoolTensor):
        keep_mask = ~remove_mask
        keep_indices = keep_mask.nonzero().squeeze(-1)

        # initialize new log weights and child_layers
        new_log_weights = []
        new_child_layers = []

        for log_weights, child_layer in self.log_weighted_child_layers:

            # remove nodes (rows)
            log_weights = log_weights.index_select(0, keep_indices).coalesce()

            # calculate probabilities of child layer nodes
            child_layer_probabilities = log_weights.clone().coalesce()
            child_layer_probabilities = child_layer_probabilities.sum(0)  # shape: (#child_nodes,)
            child_layer_probabilities.values().exp_()
            child_layer_probabilities = child_layer_probabilities.to_dense()

            # check if there is a child that has no incoming edge anymore
            remove_mask: torch.BoolTensor = child_layer_probabilities == 0
            if remove_mask.any():
                child_layer.remove_nodes_inplace(remove_mask)

            if child_layer.number_of_nodes > 0:
                new_log_weights.append(sparse_remove_rows_and_cols_where_all(log_weights, -torch.inf))
                new_child_layers.append(child_layer)

        self.log_weights = new_log_weights
        self.child_layers = new_child_layers

    @property
    def log_normalization_constants(self) -> torch.Tensor:
        result = self.concatenated_weights.clone().coalesce()
        result.values().exp_()
        result = result.sum(1)
        result.values().log_()
        return result.to_dense()

    @property
    def normalized_log_weights(self):
        result = [log_weights.clone().coalesce() for log_weights in self.log_weights]
        log_normalization_constants = self.log_normalization_constants
        for log_weights in result:
            log_weights.values().sub_(log_normalization_constants[log_weights.indices()[0]])
        return result

    def log_conditional_of_simple_event(self, event: SimpleEvent) -> Tuple[Optional[Layer], torch.Tensor]:

        conditional_child_layers = []
        conditional_log_weights = []

        probabilities = torch.zeros(self.number_of_nodes, dtype=torch.double)

        for log_weights, child_layer in self.log_weighted_child_layers:

            # get the conditional of the child layer, log prob shape: (#child_nodes, 1)
            conditional, child_log_prob = child_layer.log_conditional_of_simple_event(event)

            if conditional is None:
                continue

            # clone weights
            log_weights = log_weights.clone().coalesce().double()  # shape: (#nodes, #child_nodes)

            # calculate the weighted sum of the child log probabilities
            add_sparse_edges_dense_child_tensor_inplace(log_weights, child_log_prob)

            # calculate the probabilities of the child nodes in total
            current_probabilities = log_weights.clone().coalesce()
            current_probabilities.values().exp_()
            current_probabilities = current_probabilities.sum(1).to_dense()
            probabilities += current_probabilities

            # update log weights for conditional layer
            log_weights = sparse_remove_rows_and_cols_where_all(log_weights, -torch.inf)

            conditional_child_layers.append(conditional)
            conditional_log_weights.append(log_weights)

        if len(conditional_child_layers) == 0:
            return self.impossible_condition_result

        resulting_layer = SumLayer(conditional_child_layers, conditional_log_weights)
        return resulting_layer, (probabilities.log() - self.log_normalization_constants)


class ProductLayer(InnerLayer):
    """
    A layer that represents the product of multiple other units.
    """

    child_layers: List[Union[SumLayer, InputLayer]]
    """
    The child of a product layer is a list that contains groups sum units with the same scope or groups of input
    units with the same scope.
    """

    edges: torch.Tensor  # SparseTensor[int]
    """
    The edges consist of a sparse matrix containing integers.
    The first dimension describes the edges for each child layer.
    The second dimension describes the edges for each node in the child layer.
    The integers are interpreted in such a way that n-th value represents a edge (n, edges[n]).
    
    Nodes in the child layer can be mapped to by multiple nodes in this layer.
    
    The shape is (#child_layers, #nodes).
    """

    def __init__(self, child_layers: List[Layer], edges: torch.Tensor):
        """
        Initialize the product layer.

        :param child_layers: The child layers of the product layer.
        :param edges: The edges of the product layer.
        """
        super().__init__(child_layers)
        self.edges = edges

    def validate(self):
        assert self.edges.shape == (len(self.child_layers), self.number_of_nodes), \
            (f"The shape of the edges must be {(len(self.child_layers), self.number_of_nodes)} "
             f"but was {self.edges.shape}.")

    @classmethod
    def original_class(cls) -> Tuple[Type, ...]:
        return ProductUnit,

    @property
    def number_of_nodes(self) -> int:
        return self.edges.shape[1]

    def decomposes_as(self, other: Layer) -> bool:
        return set(child_layer.variables for child_layer in self.child_layers) == \
                set(child_layer.variables for child_layer in other.child_layers)

    def mergeable_with(self, other: Layer):
        return super().mergeable_with(other) and self.decomposes_as(other)

    @cached_property
    def columns_of_child_layers(self) -> Tuple[Tuple[int, ...], ...]:
        """
        :return: The indices of the variables for each child layer in the variable-vector of this layer.
        """
        result = []
        for layer in self.child_layers:
            layer_indices = [self.variables.index(variable) for variable in layer.variables]
            result.append(tuple(layer_indices))
        return tuple(result)

    # @torch.compile
    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(len(x), self.number_of_nodes, dtype=torch.double)
        for columns, edges, layer in zip(self.columns_of_child_layers, self.edges, self.child_layers):

            edges = edges.coalesce()
            # calculate the log likelihood over the columns of the child layer
            ll = layer.log_likelihood(x[:, columns])  # shape: (#x, #child_nodes)

            # gather the ll at the indices of the nodes that are required for the edges
            ll = ll[:, edges.values()]  # shape: (#x, #len(edges.values()))
            # assert ll.shape == (len(x), len(edges.values()))

            # add the gathered values to the result where the edges define the indices
            result[:, edges.indices().squeeze(0)] += ll
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
        result = torch.ones(self.number_of_nodes,)
        for edges, layer in zip(self.edges, self.child_layers):
            child_layer_prob = layer.probability_of_simple_event(event)  # shape: (#child_nodes, 1)
            probabilities = child_layer_prob[edges]  # shape: (#nodes, 1)
            result *= probabilities
        return result

    def log_mode(self) -> Tuple[Event, float]:
        pass

    def sample(self, amount: int) -> torch.Tensor:
        pass

    @property
    def support_per_node(self) -> List[Event]:
        pass

    def log_conditional_of_simple_event(self, event: SimpleEvent) -> Tuple[Optional[Self], torch.Tensor]:

        # initialize the conditional child layers and the log probabilities
        log_probabilities = torch.zeros(self.number_of_nodes,)
        conditional_child_layers = []

        # create new edge matrix with nan as default value. nan indicates that an edge got deleted
        remapped_edges = torch.full_like(self.edges, torch.nan, dtype=torch.float)

        # for edge bundle and child layer
        for index, (edges, child_layer) in enumerate(zip(self.edges, self.child_layers)):

            # condition the child layer
            conditional, child_log_prob = child_layer.log_conditional_of_simple_event(event)

            # if it is entirely impossible, this layer also is
            if conditional is None:
                return self.impossible_condition_result

            # create the remapping of the node indices. nan indicates the node got deleted
            new_node_indices = torch.arange(conditional.number_of_nodes)
            layer_remap = torch.full((child_layer.number_of_nodes, ), torch.nan)
            layer_remap[child_log_prob.squeeze() > -torch.inf] = new_node_indices.float()
            
            # update the edges
            remapped_edges[index] = layer_remap[edges]

            # update the log probabilities and child layers
            log_probabilities += child_log_prob[edges]
            conditional_child_layers.append(conditional)

        # get nodes that should be removed as boolean mask
        remove_mask = log_probabilities.squeeze(-1) == -torch.inf  # shape (#nodes, )
        keep_mask = ~remove_mask
        new_edges = torch.full((len(conditional_child_layers), keep_mask.sum()), -1, dtype=torch.long)

        # perform a second pass through the new layers and clean up unused nodes
        for index, edges in enumerate(remapped_edges):
            # update the edges
            new_edges[index] = edges[keep_mask]

        # construct result and clean it up
        result = self.__class__(conditional_child_layers, new_edges)
        result.clean_up_orphans_inplace()
        return result, log_probabilities

    def remove_nodes_inplace(self, remove_mask: torch.BoolTensor):

        # remove nodes from the layer
        self.edges = self.edges[:, ~remove_mask]

        # remove nodes from the child layers
        for index, (edges, child_layer) in enumerate(zip(self.edges, self.child_layers)):

            # create a removal mask for the child layer
            child_layer_remove_mask = torch.ones(child_layer.number_of_nodes).bool()
            child_layer_remove_mask[edges] = False

            # remove nodes from the child layer
            child_layer.remove_nodes_inplace(child_layer_remove_mask)

            # update the edges of this layer
            self.edges[index] = shrink_index_tensor(edges.unsqueeze(-1)).squeeze(-1)

    def merge_with_one_layer_inplace(self, other: Self):
        ...

    def merge_with(self, others: List[Self]):
        raise NotImplementedError

    def clean_up_orphans_inplace(self):
        """
        Clean up the layer inplace by removing orphans in the child layers.
        """
        for index, (edges, child_layer) in enumerate(zip(self.edges, self.child_layers)):

            # mask rather nodes have parent edges or not
            orphans = torch.ones(child_layer.number_of_nodes, dtype=torch.bool)
            orphans[edges] = False

            # if orphans exist
            if orphans.any():

                # remove them from the child layer
                child_layer.remove_nodes_inplace(orphans)

                # update the indices of this layer
                self.edges[index] = shrink_index_tensor(edges.unsqueeze(-1)).squeeze(-1)

    def __deepcopy__(self):
        child_layers = [child_layer.__deepcopy__() for child_layer in self.child_layers]
        edges = self.edges.clone()
        return self.__class__(child_layers, edges)


@dataclass
class AnnotatedLayer:
    layer: Layer
    nodes: List[ProbabilisticCircuitMixin]
    hash_remap: Dict[int, int]
