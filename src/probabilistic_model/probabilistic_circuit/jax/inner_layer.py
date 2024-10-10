from __future__ import annotations

import datetime
import inspect
import math
from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import cached_property

import equinox as eqx
import jax
import tqdm
from jax import numpy as jnp, tree_flatten
from jax.experimental.sparse import BCOO, bcoo_concatenate, BCSR
from jaxtyping import Int
from networkx.algorithms.operators.binary import difference
from random_events.product_algebra import SimpleEvent
from random_events.utils import recursive_subclasses, SubclassJSONSerializer
from scipy.sparse import coo_matrix, csr_matrix
from sortedcontainers import SortedSet
from typing_extensions import List, Iterator, Tuple, Union, Type, Dict, Any, Optional, Self

from . import create_bcoo_indices_from_row_lengths, embed_sparse_array_in_nan_array, \
    sample_from_sparse_probabilities_bcsr
from .utils import copy_bcoo, sample_from_sparse_probabilities
from ..nx.probabilistic_circuit import SumUnit, ProductUnit, ProbabilisticCircuitMixin


def inverse_class_of(clazz: Type[ProbabilisticCircuitMixin]) -> Type[Layer]:
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

    @staticmethod
    def create_layers_from_nodes(nodes: List[ProbabilisticCircuitMixin], child_layers: List[NXConverterLayer],
                                 progress_bar: bool = True) \
            -> List[NXConverterLayer]:
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
                                                                                    child_layers, progress_bar)
                result.append(layer)

        return result

    @classmethod
    @abstractmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[ProbabilisticCircuitMixin],
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

    def sample_from_frequencies(self, frequencies: jax.Array, key: jax.random.PRNGKey) -> BCOO:
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

    def sample_from_frequency_block(self, frequency_block: BCOO, child_layer: Layer,
                                    key: jax.random.PRNGKey) -> Optional[BCOO]:
        """
        Get the samples from the frequency-block of the child layer.

        :param frequency_block: A parse tensor that maps the nodes of this layer to the frequencies of the
        child layer nodes. An element at position (i, j) describes that node i of this layer request
        frequency_block[i, j] many samples from the child node j.
        :param child_layer: The child layer to sample from.
        :param key: The random key to use for sampling.
        :return: The samples for this layer.
        The result is a sparse matrix with shape (#nodes, max(frequency_block.sum(1)), #childer_layer.variables).
        The result is None if no samples are requested.
        """

        # calculate the total number of samples requested for each node of the child layer
        frequencies_for_child_nodes = frequency_block.sum(0).todense()


        if all(frequencies_for_child_nodes == 0):
            return None

        # calculate total samples requested for each node of this layer
        frequencies = frequency_block.sum(1).todense()

        # calculate the row (node) in this layer the samples in samples_from_child_layer.values() belong to.
        # the node_ownership should contain the node index in this layer for each sample
        # Example: [1, 1, 0] means that the first two samples belong to node 1 and the last sample belongs to node 0.
        transposed_frequency_block = frequency_block.T.sort_indices()


        node_ownership = jnp.repeat(transposed_frequency_block.indices[:, 1], transposed_frequency_block.data)

        # sample the child layer
        samples_from_child_layer = child_layer.sample_from_frequencies(frequencies_for_child_nodes, key)

        # arg-sort the node ownership to get the indices of the samples in the correct order
        arg_sorted_indices = jnp.argsort(node_ownership)

        # reorder the samples from the child layer
        samples_from_child_layer = samples_from_child_layer.data[arg_sorted_indices]

        samples_of_block = BCOO((samples_from_child_layer, create_bcoo_indices_from_row_lengths(frequencies)),
                                shape=(self.number_of_nodes, jnp.max(frequencies), len(child_layer.variables)),
                                indices_sorted=True, unique_indices=True)

        return samples_of_block

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["child_layers"] = [child_layer.to_json() for child_layer in self.child_layers]
        return result


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

    def sample_from_frequencies(self, frequencies: jax.Array, key: jax.random.PRNGKey):

        node_to_child_frequency_map = self.node_to_child_frequency_map_exact(frequencies, key)

        # offset for shifting through the frequencies of the node_to_child_frequency_map
        prev_column_index = 0

        all_samples = []

        for child_layer in self.child_layers:

            # extract the frequencies for the child layer
            current_frequency_block = node_to_child_frequency_map[:,
                                      prev_column_index:prev_column_index + child_layer.number_of_nodes]
            samples = self.sample_from_frequency_block(current_frequency_block, child_layer, key)

            if samples is not None:
                all_samples.append(samples)

            # shift the offset
            prev_column_index += child_layer.number_of_nodes

        # concatenate the samples
        catted_samples = bcoo_concatenate(all_samples, dimension=1).sort_indices()

        # build result
        new_indices = create_bcoo_indices_from_row_lengths(frequencies)
        result = BCOO((catted_samples.data, new_indices),
                      shape=(self.number_of_nodes, jnp.max(frequencies), len(self.variables)),
                      indices_sorted=True, unique_indices=True)

        return result

    def node_to_child_frequency_map_exact(self, frequencies: jax.Array, key: jax.random.PRNGKey):
        """
        Sample from the exact distribution of the layer by interpreting every node as latent variable.
        This is very slow due to BCOO.sum_duplicates being very slow.

        :param frequencies:
        :param key:
        :return:
        """
        clw = self.normalized_weights
        csr = coo_matrix((clw.data, clw.indices.T), shape=(clw.shape[0], clw.shape[1])).tocsr(copy=False)
        return sample_from_sparse_probabilities_bcsr(csr, clw.indices, frequencies, key)

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



    def sample_from_frequencies(self, frequencies: jax.Array, key: jax.random.PRNGKey) -> BCOO:
        concatenated_samples_per_variable = [jnp.full((0, 1), jnp.nan) for _ in self.variables]
        edges_bcsr = BCSR.from_bcoo(self.edges)
        for row_index, (start, end, child_layer) in enumerate(zip(edges_bcsr.indptr[:-1], edges_bcsr.indptr[1:], self.child_layers)):

            # get the log-probabilities of the current row
            row = edges_bcsr.data[start:end]
            column_indices = edges_bcsr.indices[start:end]
            frequencies_for_child_layer = BCOO(
                (frequencies[column_indices], jnp.array([column_indices, row]).T),
                        shape=(self.number_of_nodes, child_layer.number_of_nodes),
                        indices_sorted=True, unique_indices=True)

            # sample from the child layer
            current_samples = self.sample_from_frequency_block(frequencies_for_child_layer, child_layer, key)

            if current_samples is None:
                continue

            # write samples in the correct columns for the result
            for column in child_layer.variables:
                concatenated_samples_per_variable[column] = (
                    jnp.concatenate((concatenated_samples_per_variable[column], current_samples.data[:, (column,)])))

        # assemble the result
        result_indices = create_bcoo_indices_from_row_lengths(frequencies)
        result_values = jnp.concatenate(concatenated_samples_per_variable, axis=-1)

        result = BCOO((result_values, result_indices),
                      shape=(self.number_of_nodes, jnp.max(frequencies), len(self.variables)),
                      indices_sorted=True, unique_indices=True)

        return result

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

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        child_layer = [Layer.from_json(child_layer) for child_layer in data["child_layers"]]
        edges = BCOO((jnp.array(data["edges"][0]), jnp.array(data["edges"][1])), shape=data["edges"][2],
                     indices_sorted=True, unique_indices=True)
        return cls(child_layer, edges)

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[ProbabilisticCircuitMixin],
                                                         child_layers: List[NXConverterLayer],
                                                         progress_bar: bool = True) -> \
            NXConverterLayer:

        hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        number_of_nodes = len(nodes)

        edge_indices = []
        edge_values = []

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


@dataclass
class NXConverterLayer:
    """
    Class used for conversion from a probabilistic circuit in networkx to a layered circuit in jax.
    """
    layer: Layer
    nodes: List[ProbabilisticCircuitMixin]
    hash_remap: Dict[int, int]
