from __future__ import annotations

import inspect
import math
from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import cached_property

import equinox as eqx
import jax
import numpy as np
import tqdm
from jax import numpy as jnp, tree_flatten
from jax.experimental.sparse import BCOO, bcoo_concatenate
from jaxtyping import Int
from keras.src.legacy.backend import variable
from random_events.product_algebra import SimpleEvent
from random_events.utils import recursive_subclasses, SubclassJSONSerializer
from random_events.variable import Variable
from scipy.sparse import coo_matrix, coo_array, csc_array
from sortedcontainers import SortedSet
from typing_extensions import List, Iterator, Tuple, Union, Type, Dict, Any, Self, Optional

from . import shrink_index_array
from .utils import copy_bcoo, sample_from_sparse_probabilities_csc, sparse_remove_rows_and_cols_where_all
from ..nx.probabilistic_circuit import (SumUnit, ProductUnit, Unit,
                                        ProbabilisticCircuit as NXProbabilisticCircuit)


def inverse_class_of(clazz: Type[Unit]) -> Type[Layer]:
    for subclass in recursive_subclasses(Layer):
        if not inspect.isabstract(subclass):
            if issubclass(clazz, subclass.nx_classes()):
                return subclass

    raise TypeError(f"Could not find class for {clazz}")


class Layer(eqx.Module, SubclassJSONSerializer, ABC):
    """
    Abstract class for Layers of a layered circuit.

    Layers have the same scope (set of variables) for every node in them.
    """

    @abstractmethod
    def log_likelihood_of_nodes_single(self, x: jnp.array) -> jnp.array:
        """
        Calculate the log-likelihood of the distribution.

        :param x: The input vector.
        :return: The log-likelihood of every node in the layer for x.
        """
        raise NotImplementedError

    def log_likelihood_of_nodes(self, x: jnp.array) -> jnp.array:
        """
        Vectorized version of :meth:`log_likelihood_of_nodes_single`
        """
        return jax.vmap(self.log_likelihood_of_nodes_single)(x)

    def cdf_of_nodes_single(self, x: jnp.array) -> jnp.array:
        """
        Calculate the cumulative distribution function of the distribution if applicable.

        :param x: The input vector.
        :return: The cumulative distribution function of every node in the layer for x.
        """
        raise NotImplementedError

    def cdf_of_nodes(self, x: jnp.array) -> jnp.array:
        """
        Vectorized version of :meth:`cdf_of_nodes_single`
        """
        return jax.vmap(self.cdf_of_nodes_single)(x)

    def probability_of_simple_event(self, event: SimpleEvent) -> jnp.array:
        """
        Calculate the probability of a simple event P(E).

        :param event: The simple event to calculate the probability for. It has to contain every variable.
        :return: P(E)
        """
        raise NotImplementedError

    def validate(self):
        """
        Validate the parameters and their layouts.
        """
        raise NotImplementedError

    @property
    def number_of_nodes(self) -> int:
        """
        :return: The number of nodes in the layer.
        """
        raise NotImplementedError

    def all_layers(self) -> List[Layer]:
        """
        :return: A list of all layers in the circuit.
        """
        return [self]

    def all_layers_with_depth(self, depth: int = 0) -> List[Tuple[int, Layer]]:
        """
        :return: A list of tuples of all layers in the circuit with their depth.
        """
        return [(depth, self)]

    @property
    @abstractmethod
    def variables(self) -> jax.Array:
        """
        :return: The variable indices of this layer.
        """
        raise NotImplementedError

    def __deepcopy__(self) -> 'Layer':
        """
        Create a deep copy of the layer.
        """
        raise NotImplementedError

    @classmethod
    def nx_classes(cls) -> Tuple[Type, ...]:
        """
        :return: The tuple of matching classes of the layer in the probabilistic_model.probabilistic_circuit.nx package.
        """
        return tuple()

    def to_nx(self, variables: SortedSet[Variable], progress_bar: Optional[tqdm.tqdm] = None) -> List[
        Unit]:
        """
        Convert the layer to a networkx circuit.
        For every node in this circuit, a corresponding node in the networkx circuit
        is created.
        The nodes all belong to the same circuit.

        :param variables: The variables of the circuit.
        :param progress_bar: A progress bar to show the progress.
        :return: The nodes of the networkx circuit.
        """
        raise NotImplementedError

    @property
    def impossible_condition_result(self) -> Tuple[None, jax.Array]:
        """
        :return: The result that a layer yields if it is conditioned on an event E with P(E) = 0
        """
        return None, jnp.full((self.number_of_nodes,), -jnp.inf, dtype=jnp.float32)

    def log_conditional_of_simple_event(self, event: SimpleEvent, ) -> Tuple[Optional[Self], jax.Array]:
        """
        Calculate the conditional probability distribution given a simple event P(X|E).
        Also return the log probability of E log(P(E)).

        :param event: The event to calculate the conditional distribution for.
        :return: The conditional distribution and the log probability of the event.
        """
        raise NotImplementedError

    @staticmethod
    def create_layers_from_nodes(nodes: List[Unit], child_layers: List[NXConverterLayer],
                                 progress_bar: bool = True) \
            -> List[NXConverterLayer]:
        """
        Create a layer from a list of nodes.
        """
        result = []

        unique_types = set(type(node) if not node.is_leaf else type(node.distribution) for node in nodes)
        for unique_type in unique_types:
            nodes_of_current_type = [node for node in nodes if (isinstance(node, unique_type) if not node.is_leaf
                                                                else isinstance(node.distribution, unique_type))]

            if nodes[0].is_leaf:
                unique_type = type(nodes_of_current_type[0].distribution)

            layer_type = inverse_class_of(unique_type)

            scopes = [tuple(node.variables) for node in nodes_of_current_type]
            unique_scopes = set(scopes)
            for scope in unique_scopes:
                nodes_of_current_type_and_scope = [node for node in nodes_of_current_type if
                                                   tuple(node.variables) == scope]
                layer = layer_type.create_layer_from_nodes_with_same_type_and_scope(nodes_of_current_type_and_scope,
                                                                                    child_layers, progress_bar)
                result.append(layer)

        return result

    @classmethod
    @abstractmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[Unit],
                                                         child_layers: List[NXConverterLayer],
                                                         progress_bar: bool = True) -> \
            NXConverterLayer:
        """
        Create a layer from a list of nodes with the same type and scope.
        """
        raise NotImplementedError

    def partition(self) -> Tuple[Any, Any]:
        """
        Partition the layer into the parameters and the static structure.

        :return: A tuple containing the parameters and the static structure as pytrees.
        """
        return eqx.partition(self, eqx.is_inexact_array)

    @property
    def number_of_trainable_parameters(self):
        """
        :return: The trainable parameters of the layer and all child layers.
        """
        parameters, _ = self.partition()
        flattened_parameters, _ = tree_flatten(parameters)
        number_of_parameters = sum([len(p) for p in flattened_parameters])
        return number_of_parameters

    @property
    def number_of_components(self) -> 0:
        """
        :return: The number of components (leaves + edges) of the entire circuit
        """
        return self.number_of_nodes

    def sample_from_frequencies(self, frequencies: np.array, result: np.array, start_index=0):
        raise NotImplementedError

    def moment_of_nodes(self, order: jax.Array, center: jax.Array):
        """
        Calculate the moment of the nodes.
        The order and center vectors describe the moments for all variables in the entire model. Hence, they should
        never be touched by the forward pass.

        :param order: The order of the moment for each variable.
        :param center: The center of the moment for each variable.
        :return: The moments of the nodes with shape (#nodes, #variables).
        """
        raise NotImplementedError

    def merge_with(self, others: List[Self]) -> Self:
        """
        Merge the layer with others of the same type.
        """
        raise NotImplementedError

    def remove_nodes(self, remove_mask: jax.Array) -> Self:
        """
        Remove nodes from the layer.

        :param remove_mask: A boolean mask of the nodes to remove.
        :return: The layer with the nodes removed.
        """
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

    @property
    @abstractmethod
    def variables(self) -> jax.Array:
        raise NotImplementedError

    def all_layers(self) -> List[Layer]:
        """
        :return: A list of all layers in the circuit.
        """
        result = [self]
        for child_layer in self.child_layers:
            result.extend(child_layer.all_layers())
        return result

    def all_layers_with_depth(self, depth: int = 0) -> List[Tuple[int, Layer]]:
        """
        :return: A list of tuples of all layers in the circuit with their depth.
        """
        result = [(depth, self)]
        for child_layer in self.child_layers:
            result.extend(child_layer.all_layers_with_depth(depth + 1))
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["child_layers"] = [child_layer.to_json() for child_layer in self.child_layers]
        return result

    def clean_up_orphans(self) -> Self:
        """
        Clean up the layer by removing orphans in the child layers.
        """
        raise NotImplementedError

class InputLayer(Layer, ABC):
    """
    Abstract base class for univariate input units.

    Input layers contain only one type of distribution such that the vectorization of the log likelihood
    calculation works without bottleneck statements like if/else or loops.
    """

    _variables: jnp.array = eqx.field(static=True)
    """
    The variable indices of the layer.
    """

    def __init__(self, variable: int):
        super().__init__()
        self._variables = jnp.array([variable])

    @property
    def variables(self) -> jax.Array:
        return self._variables

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["variable"] = self._variables[0].item()
        return result

    @property
    def variable(self):
        return self._variables[0].item()


class SumLayer(InnerLayer):
    log_weights: List[BCOO]
    child_layers: Union[List[[ProductLayer]], List[InputLayer]]

    def __init__(self, child_layers: List[Layer], log_weights: List[BCOO]):
        super().__init__(child_layers)
        self.log_weights = log_weights

    def validate(self):
        for log_weights in self.log_weights:
            assert log_weights.shape[0] == self.number_of_nodes, "The number of nodes must match the number of weights."

        for log_weights, child_layer in self.log_weighted_child_layers:
            assert log_weights.shape[
                       1] == child_layer.number_of_nodes, "The number of nodes must match the number of weights."

    @cached_property
    def variables(self) -> jax.Array:
        return self.child_layers[0].variables

    @classmethod
    def nx_classes(cls) -> Tuple[Type, ...]:
        return SumUnit,

    @property
    def log_weighted_child_layers(self) -> Iterator[Tuple[BCOO, Layer]]:
        """
        :returns: Yields log weights and the child layers zipped together.
        """
        yield from zip(self.log_weights, self.child_layers)

    @property
    def number_of_nodes(self) -> int:
        return self.log_weights[0].shape[0]

    @property
    def number_of_components(self) -> int:
        return sum([cl.number_of_components for cl in self.child_layers]) + sum([lw.nse for lw in self.log_weights])

    @property
    def concatenated_log_weights(self) -> BCOO:
        """
        :return: The concatenated weights of the child layers for each node.
        """
        return bcoo_concatenate(self.log_weights, dimension=1).sort_indices()

    @property
    def log_normalization_constants(self) -> jax.Array:
        result = self.concatenated_log_weights
        result.data = jnp.exp(result.data)
        result = result.sum(1).todense()
        return jnp.log(result)

    @property
    def normalized_weights(self):
        """
        :return: The normalized weights of the child layers for each node.
        """
        result = self.concatenated_log_weights
        z = self.log_normalization_constants
        result.data = jnp.exp(result.data - z[result.indices[:, 0]])
        return result

    def log_likelihood_of_nodes_single(self, x: jax.Array) -> jax.Array:
        result = jnp.zeros(self.number_of_nodes, dtype=jnp.float32)

        for log_weights, child_layer in self.log_weighted_child_layers:
            # get the log likelihoods of the child nodes
            child_layer_log_likelihood = child_layer.log_likelihood_of_nodes_single(x)

            # weight the log likelihood of the child nodes by the weight for each node of this layer
            cloned_log_weights = copy_bcoo(log_weights)  # clone the weights

            # multiply the weights with the child layer likelihood
            cloned_log_weights.data += child_layer_log_likelihood[cloned_log_weights.indices[:, 1]]
            cloned_log_weights.data = jnp.exp(cloned_log_weights.data)  # exponent weights

            # sum the weights for each node
            ll = cloned_log_weights.sum(1).todense()

            # sum the child layer result
            result += ll

        return jnp.where(result > 0, jnp.log(result) - self.log_normalization_constants, -jnp.inf)

    def cdf_of_nodes_single(self, x: jnp.array) -> jnp.array:
        result = jnp.zeros(self.number_of_nodes, dtype=jnp.float32)

        for log_weights, child_layer in self.log_weighted_child_layers:
            # get the cdf of the child nodes
            child_layer_cdf = child_layer.cdf_of_nodes_single(x)

            # weight the cdf of the child nodes by the weight for each node of this layer
            cloned_log_weights = copy_bcoo(log_weights)  # clone the weights

            # multiply the weights with the child layer cdf
            cloned_log_weights.data = jnp.exp(cloned_log_weights.data)  # exponent weights
            cloned_log_weights.data *= child_layer_cdf[cloned_log_weights.indices[:, 1]]

            # sum the weights for each node
            ll = cloned_log_weights.sum(1).todense()

            # sum the child layer result
            result += ll

        # normalize the result
        normalization_constants = jnp.exp(self.log_normalization_constants)
        return result / normalization_constants

    def probability_of_simple_event(self, event: SimpleEvent) -> jnp.array:
        result = jnp.zeros(self.number_of_nodes, dtype=jnp.float32)

        for log_weights, child_layer in self.log_weighted_child_layers:
            # get the probability of the child nodes
            child_layer_prob = child_layer.probability_of_simple_event(event)

            # weight the probability of the child nodes by the weight for each node of this layer
            cloned_log_weights = copy_bcoo(log_weights)  # clone the weights

            # multiply the weights with the child layer cdf
            cloned_log_weights.data = jnp.exp(cloned_log_weights.data)  # exponent weights
            cloned_log_weights.data *= child_layer_prob[cloned_log_weights.indices[:, 1]]

            # sum the weights for each node
            ll = cloned_log_weights.sum(1).todense()

            # sum the child layer result
            result += ll

        # normalize the result
        normalization_constants = jnp.exp(self.log_normalization_constants)
        return result / normalization_constants

    def moment_of_nodes(self, order: jax.Array, center: jax.Array):
        result = jnp.zeros((self.number_of_nodes, len(self.variables)), dtype=jnp.float32)

        for log_weights, child_layer in self.log_weighted_child_layers:
            # get the moment of the child nodes
            moment = child_layer.moment_of_nodes(order, center)  # shape (#child_layer_nodes, #variables)

            # weight the moment of the child nodes by the weight for each node of this layer
            weights = copy_bcoo(log_weights)  # clone the weights, shape (#nodes, #child_layer_nodes)
            weights.data = jnp.exp(weights.data)  # exponent weights

            #  calculate the weighted sum in layer
            moment = weights @ moment

            # sum the child layer result
            result += moment

        return result / jnp.exp(self.log_normalization_constants.reshape(-1, 1))

    def sample_from_frequencies(self, frequencies: np.array, result: np.array, start_index=0):
        node_to_child_frequency_map = self.node_to_child_frequency_map(frequencies)

        # offset for shifting through the frequencies of the node_to_child_frequency_map
        prev_column_index = 0

        consumed_indices = start_index

        for child_layer in self.child_layers:
            # extract the frequencies for the child layer
            current_frequency_block = node_to_child_frequency_map[:,
                                      prev_column_index:prev_column_index + child_layer.number_of_nodes]
            frequencies_for_child_nodes = current_frequency_block.sum(0)
            child_layer.sample_from_frequencies(frequencies_for_child_nodes, result, consumed_indices)
            consumed_indices += frequencies_for_child_nodes.sum()

            # shift the offset
            prev_column_index += child_layer.number_of_nodes

    def node_to_child_frequency_map(self, frequencies: np.array):
        """
        Sample from the exact distribution of the layer by interpreting every node as latent variable.
        This is very slow due to BCOO.sum_duplicates being very slow.

        :param frequencies:
        :param key:
        :return:
        """
        clw = self.normalized_weights
        csr = coo_matrix((clw.data, clw.indices.T), shape=clw.shape).tocsr(copy=False)
        return sample_from_sparse_probabilities_csc(csr, frequencies)

    def log_conditional_of_simple_event(self, event: SimpleEvent, ) -> Tuple[Optional[Self], jax.Array]:
        conditional_child_layers = []
        conditional_log_weights = []

        probabilities = jnp.zeros(self.number_of_nodes, dtype=jnp.float32)

        for log_weights, child_layer in self.log_weighted_child_layers:
            # get the conditional of the child layer
            conditional, child_log_prob = child_layer.log_conditional_of_simple_event(event)
            if conditional is None:
                continue

            # clone weights
            log_weights = copy_bcoo(log_weights)

            # calculate the weighted sum of the child log probabilities
            log_weights.data += child_log_prob[log_weights.indices[:, 1]]

            # skip if this layer is not connected to anything anymore
            if jnp.all(log_weights.data == -jnp.inf):
                continue

            log_weights.data = jnp.exp(log_weights.data)

            # calculate the probabilities of the child nodes in total
            current_probabilities = log_weights.sum(1).todense()
            probabilities += current_probabilities

            log_weights.data = jnp.log(log_weights.data)

            conditional_child_layers.append(conditional)
            conditional_log_weights.append(log_weights)

        if len(conditional_child_layers) == 0:
            return self.impossible_condition_result

        log_probabilities = jnp.log(probabilities)

        concatenated_log_weights = bcoo_concatenate(conditional_log_weights, dimension=1).sort_indices()
        # remove rows and columns where all weights are -inf
        cleaned_log_weights = sparse_remove_rows_and_cols_where_all(concatenated_log_weights, -jnp.inf)

        # normalize the weights
        z = cleaned_log_weights.sum(1).todense()
        cleaned_log_weights.data -= z[cleaned_log_weights.indices[:, 0]]

        # slice the weights for each child layer
        log_weight_slices = jnp.array([0] + [ccl.number_of_nodes for ccl in conditional_child_layers])
        log_weight_slices = jnp.cumsum(log_weight_slices)
        conditional_log_weights = [cleaned_log_weights[:, log_weight_slices[i]:log_weight_slices[i + 1]].sort_indices()
                                   for i in range(len(conditional_child_layers))]

        resulting_layer = SumLayer(conditional_child_layers, conditional_log_weights)
        return resulting_layer, (log_probabilities - self.log_normalization_constants)

    def __deepcopy__(self):
        child_layers = [child_layer.__deepcopy__() for child_layer in self.child_layers]
        log_weights = [copy_bcoo(log_weight) for log_weight in self.log_weights]
        return self.__class__(child_layers, log_weights)

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["log_weights"] = [(lw.data.tolist(), lw.indices.tolist(), lw.shape) for lw in self.log_weights]
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        child_layer = [Layer.from_json(child_layer) for child_layer in data["child_layers"]]
        log_weights = [BCOO((jnp.array(lw[0]), jnp.array(lw[1])), shape=lw[2],
                            indices_sorted=True, unique_indices=True) for lw in data["log_weights"]]
        return cls(child_layer, log_weights)

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[SumUnit],
                                                         child_layers: List[NXConverterLayer],
                                                         progress_bar: bool = True) -> \
            NXConverterLayer:

        result_hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        variables = jnp.array(
            [nodes[0].probabilistic_circuit.variables.index(variable) for variable in nodes[0].variables])

        number_of_nodes = len(nodes)

        # filter the child layers to only contain layers with the same scope as this one
        filtered_child_layers = [child_layer for child_layer in child_layers if (child_layer.layer.variables ==
                                                                                 variables).all()]
        log_weights = []

        # for every possible child layer
        for child_layer in filtered_child_layers:

            # initialize indices and values for sparse weight matrix
            indices = []
            values = []

            # gather indices and log weights
            for index, node in enumerate(tqdm.tqdm(nodes, desc="Calculating weights for sum node")
                                         if progress_bar else nodes):
                for weight, subcircuit in node.weighted_subcircuits:
                    if hash(subcircuit) in child_layer.hash_remap:
                        indices.append((index, child_layer.hash_remap[hash(subcircuit)]))
                        values.append((math.log(weight)))

            # assemble sparse log weight matrix
            log_weights.append(BCOO((jnp.array(values), jnp.array(indices)),
                                    shape=(number_of_nodes,
                                           child_layer.layer.number_of_nodes)))

        sum_layer = cls([cl.layer for cl in filtered_child_layers], log_weights)
        return NXConverterLayer(sum_layer, nodes, result_hash_remap)

    def remove_nodes(self, remove_mask: jax.Array) -> Self:
        new_log_weights = [lw[~remove_mask] for lw in self.log_weights]
        return self.__class__(self.child_layers, new_log_weights)

    def clean_up_orphans(self) -> Self:
        raise NotImplementedError

    def to_nx(self, variables: SortedSet[Variable], progress_bar: Optional[tqdm.tqdm] = None) -> List[
        Unit]:

        variables_ = [variables[i] for i in self.variables]

        if progress_bar:
            progress_bar.set_postfix_str(f"Parsing Sum Layer for variables {variables_}")

        nx_pc = NXProbabilisticCircuit()
        units = [SumUnit() for _ in range(self.number_of_nodes)]
        nx_pc.add_nodes_from(units)

        child_layer_nx = [cl.to_nx(variables, progress_bar) for cl in self.child_layers]

        clw = self.normalized_weights
        csc_weights = coo_matrix((clw.data, clw.indices.T), shape=clw.shape).tocsc(copy=False)

        # offset for shifting through the frequencies of the node_to_child_frequency_map
        prev_column_index = 0

        for child_layer in child_layer_nx:
            # extract the weights for the child layer
            current_weight_block: csc_array = csc_weights[:, prev_column_index:prev_column_index + len(child_layer)]
            current_weight_block: coo_array = current_weight_block.tocoo(False)

            for row, col, weight in zip(current_weight_block.row, current_weight_block.col, current_weight_block.data):
                units[row].add_subcircuit(child_layer[col], weight)

                if progress_bar:
                    progress_bar.update()

            # shift the offset
            prev_column_index += len(child_layer)

        return units


class ProductLayer(InnerLayer):
    """
    A layer that represents the product of multiple other units.
    """

    child_layers: List[Union[SumLayer, InputLayer]]
    """
    The child of a product layer is a list that contains groups sum units with the same scope or groups of input
    units with the same scope.
    """

    edges: Int[BCOO, "len(child_layers), number_of_nodes"] = eqx.field(static=True)
    """
    The edges consist of a sparse matrix containing integers.
    The first dimension describes the edges for each child layer.
    The second dimension describes the edges for each node in the child layer.
    The integers are interpreted in such a way that n-th value represents a edge (n, edges[n]).

    Nodes in the child layer can be mapped to by multiple nodes in this layer.

    The shape is (#child_layers, #nodes).
    """

    def __init__(self, child_layers: List[Layer], edges: BCOO):
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

    @property
    def number_of_nodes(self) -> int:
        return self.edges.shape[1]

    @classmethod
    def nx_classes(cls) -> Tuple[Type, ...]:
        return ProductUnit,

    @property
    def number_of_components(self) -> int:
        return sum([cl.number_of_components for cl in self.child_layers]) + self.edges.nse

    @cached_property
    def variables(self) -> jax.Array:
        child_layer_variables = jnp.concatenate([child_layer.variables for child_layer in self.child_layers])
        max_size = child_layer_variables.shape[0]
        unique_values = jnp.unique(child_layer_variables, size=max_size, fill_value=-1)
        unique_values = unique_values[unique_values >= 0]
        return unique_values.sort()

    def log_likelihood_of_nodes_single(self, x: jax.Array) -> jax.Array:
        result = jnp.zeros(self.number_of_nodes, dtype=jnp.float32)

        for edges, layer in zip(self.edges, self.child_layers):
            # calculate the log likelihood over the columns of the child layer
            ll = layer.log_likelihood_of_nodes_single(x[layer.variables])  # shape: #child_nodes

            # gather the ll at the indices of the nodes that are required for the edges
            ll = ll[edges.data]  # shape: #len(edges.values())

            # add the gathered values to the result where the edges define the indices
            result = result.at[edges.indices[:, 0]].add(ll)

        return result

    def cdf_of_nodes_single(self, x: jnp.array) -> jnp.array:
        result = jnp.ones(self.number_of_nodes, dtype=jnp.float32)

        for edges, layer in zip(self.edges, self.child_layers):
            # calculate the cdf over the columns of the child layer
            cdf = layer.cdf_of_nodes_single(x[layer.variables])  # shape: #child_nodes

            # gather the cdf at the indices of the nodes that are required for the edges
            cdf = cdf[edges.data]  # shape: #len(edges.values())

            # multiply the gathered values by the result where the edges define the indices
            result = result.at[edges.indices[:, 0]].mul(cdf)

        return result

    def probability_of_simple_event(self, event: SimpleEvent) -> jnp.array:
        result = jnp.ones(self.number_of_nodes, dtype=jnp.float32)

        for edges, layer in zip(self.edges, self.child_layers):
            # calculate the cdf over the columns of the child layer
            prob = layer.probability_of_simple_event(event)  # shape: #child_nodes

            # gather the cdf at the indices of the nodes that are required for the edges
            prob = prob[edges.data]  # shape: #len(edges.values())

            # multiply the gathered values by the result where the edges define the indices
            result = result.at[edges.indices[:, 0]].mul(prob)

        return result

    def sample_from_frequencies(self, frequencies: np.array, result: np.array, start_index=0):
        edges_csr = coo_array((self.edges.data, self.edges.indices.T), shape=self.edges.shape).tocsr()
        for row_index, (start, end, child_layer) in enumerate(
                zip(edges_csr.indptr[:-1], edges_csr.indptr[1:], self.child_layers)):
            # get the edges for the current child layer
            row = edges_csr.data[start:end]
            column_indices = edges_csr.indices[start:end]

            frequencies_for_child_layer = np.zeros((child_layer.number_of_nodes,), dtype=np.int32)
            frequencies_for_child_layer[row] = frequencies[column_indices]

            child_layer.sample_from_frequencies(frequencies_for_child_layer, result, start_index)

    def moment_of_nodes(self, order: jax.Array, center: jax.Array):
        result = jnp.full((self.number_of_nodes, self.variables.shape[0]), jnp.nan)
        for edges, layer in zip(self.edges, self.child_layers):
            edges = edges.sum_duplicates(remove_zeros=False)

            # calculate the moments over the columns of the child layer
            child_layer_moment = layer.moment_of_nodes(order, center)

            # gather the moments at the indices of the nodes that are required for the edges
            result = result.at[edges.indices[:, 0], layer.variables].set(child_layer_moment[edges.data][:, 0])

        return result

    def __deepcopy__(self):
        child_layers = [child_layer.__deepcopy__() for child_layer in self.child_layers]
        edges = copy_bcoo(self.edges)
        return self.__class__(child_layers, edges)

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["edges"] = (self.edges.data.tolist(), self.edges.indices.tolist(), self.edges.shape)
        return result

    def log_conditional_of_simple_event(self, event: SimpleEvent, ) -> Tuple[Optional[Self], jax.Array]:

        # initialize the conditional child layers and the log probabilities
        log_probabilities = jnp.zeros(self.number_of_nodes, dtype=jnp.float32)
        conditional_child_layers = []
        remapped_edges = []

        # for edge bundle and child layer
        for index, (edges, child_layer) in enumerate(zip(self.edges, self.child_layers)):
            edges: BCOO
            edges = edges.sum_duplicates(remove_zeros=False)

            # condition the child layer
            conditional, child_log_prob = child_layer.log_conditional_of_simple_event(event)

            # if it is entirely impossible, this layer also is
            if conditional is None:
                continue

            # update the log probabilities and child layers
            log_probabilities = log_probabilities.at[edges.indices[:, 0]].add(child_log_prob[edges.data])
            conditional_child_layers.append(conditional)

            # create the remapping of the node indices. nan indicates the node got deleted
            # enumerate the indices of the conditional child layer nodes
            new_node_indices = jnp.arange(conditional.number_of_nodes)

            # initialize the remapping of the child layer node indices
            layer_remap = jnp.full((child_layer.number_of_nodes,), jnp.nan, dtype=jnp.float32)
            layer_remap = layer_remap.at[child_log_prob > -jnp.inf].set(new_node_indices)

            # update the edges
            remapped_child_edges = layer_remap[edges.data]
            valid_edges = ~jnp.isnan(remapped_child_edges)

            # create new indices for the edges
            new_indices = edges.indices[valid_edges]
            new_indices = jnp.concatenate([jnp.zeros((len(new_indices), 1), dtype=jnp.int32), new_indices],
                                          axis=1)

            new_edges = BCOO((remapped_child_edges[valid_edges].astype(jnp.int32),
                              new_indices),
                             shape=(1, self.number_of_nodes), indices_sorted=True,
                             unique_indices=True)
            remapped_edges.append(new_edges)

        remapped_edges = bcoo_concatenate(remapped_edges, dimension=0).sort_indices()

        # get nodes that should be removed as boolean mask
        remove_mask = log_probabilities == -jnp.inf  # shape (#nodes, )
        keep_mask = ~remove_mask

        # remove the nodes that have -inf log probabilities from remapped_edges
        remapped_edges = coo_array((remapped_edges.data, remapped_edges.indices.T), shape=remapped_edges.shape).tocsc()
        remapped_edges = remapped_edges[:, keep_mask].tocoo()
        remapped_edges = BCOO((remapped_edges.data, jnp.stack((remapped_edges.row, remapped_edges.col)).T),
                              shape=remapped_edges.shape, indices_sorted=True, unique_indices=True)

        # construct result and clean it up
        result = self.__class__(conditional_child_layers, remapped_edges)
        result = result.clean_up_orphans()
        return result, log_probabilities

    def clean_up_orphans(self):
        """
        Clean up the layer by removing orphans in the child layers.
        """
        new_child_layers = []

        for index, (edges, child_layer) in enumerate(zip(self.edges, self.child_layers)):
            edges: BCOO
            edges = edges.sum_duplicates(remove_zeros=False)
            # mask rather nodes have parent edges or not
            orphans = jnp.ones(child_layer.number_of_nodes, dtype=jnp.bool)

            # mark nodes that have parents with False
            data = edges.data
            if len(data) > 0:
                orphans = orphans.at[data].set(False)

            # if orphans exist
            if orphans.any():
                # remove them from the child layer
                child_layer = child_layer.remove_nodes(orphans)
            new_child_layers.append(child_layer)

        # compress edges
        shrunken_indices = shrink_index_array(self.edges.indices)
        new_edges = BCOO((self.edges.data, shrunken_indices), shape=self.edges.shape, indices_sorted=True,
                         unique_indices=True)
        return self.__class__(new_child_layers, new_edges)

    def remove_nodes(self, remove_mask: jax.Array) -> Self:
        new_edges = self.edges[:, ~remove_mask]
        return self.__class__(self.child_layers, new_edges)

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        child_layer = [Layer.from_json(child_layer) for child_layer in data["child_layers"]]
        edges = BCOO((jnp.array(data["edges"][0]), jnp.array(data["edges"][1])), shape=data["edges"][2],
                     indices_sorted=True, unique_indices=True)
        return cls(child_layer, edges)

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[Unit],
                                                         child_layers: List[NXConverterLayer],
                                                         progress_bar: bool = True) -> \
            NXConverterLayer:

        hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        number_of_nodes = len(nodes)

        edge_indices = []
        edge_values = []
        if progress_bar:
            progress_bar = tqdm.tqdm(total=number_of_nodes, desc="Assembling Product Layer")
        # for every node in the nodes for this layer
        for node_index, node in enumerate(nodes):

            # for every child layer
            for child_layer_index, child_layer in enumerate(child_layers):
                cl_variables = SortedSet(
                    [node.probabilistic_circuit.variables[index] for index in child_layer.layer.variables])

                # for every subcircuit
                for subcircuit_index, subcircuit in enumerate(node.subcircuits):
                    # if the scopes are compatible
                    if cl_variables == subcircuit.variables:
                        # add the edge
                        edge_indices.append([child_layer_index, node_index])
                        edge_values.append(child_layer.hash_remap[hash(subcircuit)])
            if progress_bar:
                progress_bar.update(1)

        # assemble sparse edge tensor
        edges = (BCOO((jnp.array(edge_values), jnp.array(edge_indices)), shape=(len(child_layers), number_of_nodes)).
                 sort_indices().sum_duplicates(remove_zeros=False))
        layer = cls([cl.layer for cl in child_layers], edges)
        return NXConverterLayer(layer, nodes, hash_remap)

    def to_nx(self, variables: SortedSet[Variable], progress_bar: Optional[tqdm.tqdm] = None) -> List[
        Unit]:

        variables_ = [variables[i] for i in self.variables]
        if progress_bar:
            progress_bar.set_postfix_str(f"Parsing Product Layer of variables {variables_}")

        nx_pc = NXProbabilisticCircuit()
        units = [ProductUnit() for _ in range(self.number_of_nodes)]
        nx_pc.add_nodes_from(units)

        child_layer_nx = [cl.to_nx(variables, progress_bar) for cl in self.child_layers]

        for (row, col), data in zip(self.edges.indices, self.edges.data):
            units[col].add_subcircuit(child_layer_nx[row][data])
            if progress_bar:
                progress_bar.update()

        return units


@dataclass
class NXConverterLayer:
    """
    Class used for conversion from a probabilistic circuit in networkx to a layered circuit in jax.
    """
    layer: Layer
    nodes: List[Unit]
    hash_remap: Dict[int, int]
