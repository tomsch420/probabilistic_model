from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import cached_property

import jax
from jaxtyping import Array, Float, Int
from jax import numpy as jnp
import equinox as eqx
from jax.experimental.sparse import BCOO, bcoo_concatenate, bcoo_reduce_sum
from typing_extensions import List, Iterator, Tuple, Union, Type, Dict
from jax.scipy.special import logsumexp
from .utils import copy_bcoo
from ..nx.probabilistic_circuit import SumUnit, ProductUnit, ProbabilisticCircuitMixin


class Layer(eqx.Module, ABC):
    """
    Abstract class for Layers of a layered circuit.

    Layers have the same scope (set of variables) for every node in them.
    """

    @abstractmethod
    def log_likelihood_of_nodes(self, x: jnp.array) -> jnp.array:
        """
        Calculate the log-likelihood of the distribution.

        .. Note::
            The shape of the log likelihood depends on the number of samples and nodes.
            The shape of the result is (#samples, #nodes).
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


class InputLayer(Layer, ABC):
    """
    Abstract base class for univariate input units.

    Input layers contain only one type of distribution such that the vectorization of the log likelihood
    calculation works without bottleneck statements like if/else or loops.
    """

    _variables: jnp.array
    """
    The variable indices of the layer.
    """

    def __init__(self, variable: int):
        super().__init__()
        self._variables = jnp.array([variable])

    @property
    def variables(self) -> jax.Array:
        return self._variables


class SumLayer(InnerLayer):

    log_weights: List[BCOO]
    child_layers: List[Layer]

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
        return bcoo_concatenate(self.log_weights, dimension=1)

    @property
    def log_normalization_constants(self) -> jax.Array:
        result = self.concatenated_log_weights
        result.data = jnp.exp(result.data)
        result = result.sum(1).todense()
        return jnp.log(result)

    @jax.jit
    def log_likelihood_of_nodes(self, x: jax.Array) -> jax.Array:
        result = jnp.zeros((len(x), self.number_of_nodes))

        for log_weights, child_layer in self.log_weighted_child_layers:
            # get the log likelihoods of the child nodes
            ll = child_layer.log_likelihood_of_nodes(x)
            # assert ll.shape == (len(x), child_layer.number_of_nodes)

            # weight the log likelihood of the child nodes by the weight for each node of this layer
            cloned_log_weights = copy_bcoo(log_weights)  # clone the weights
            cloned_log_weights.data = jnp.exp(cloned_log_weights.data)  # exponent weights
            ll = jnp.exp(ll)  # calculate the exponential of the child log likelihoods
            #  calculate the weighted sum in layer
            ll = ll @ cloned_log_weights.T

            # sum the child layer result
            result += ll

        return jnp.log(result) - self.log_normalization_constants

    def __deepcopy__(self):
        child_layers = [child_layer.__deepcopy__() for child_layer in self.child_layers]
        log_weights = [copy_bcoo(log_weight) for log_weight in self.log_weights]
        return self.__class__(child_layers, log_weights)


class ProductLayer(InnerLayer):
    """
    A layer that represents the product of multiple other units.
    """

    child_layers: List[Union[SumLayer, InputLayer]]
    """
    The child of a product layer is a list that contains groups sum units with the same scope or groups of input
    units with the same scope.
    """

    edges: Int[BCOO, "len(child_layers), number_of_nodes"]
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
        return jnp.unique(jnp.concatenate([layer.variables for layer in self.child_layers])).sort()

    @jax.jit
    def log_likelihood_of_nodes(self, x: jax.Array) -> jax.Array:
        result = jnp.zeros((len(x), self.number_of_nodes))

        for edges, layer in zip(self.edges, self.child_layers):
            # calculate the log likelihood over the columns of the child layer
            ll = layer.log_likelihood_of_nodes(x[:, layer.variables])  # shape: (#x, #child_nodes)

            # gather the ll at the indices of the nodes that are required for the edges
            ll = ll[:, edges.data]  # shape: (#x, #len(edges.values()))
            # assert ll.shape == (len(x), len(edges.values()))

            # add the gathered values to the result where the edges define the indices
            result = result.at[:, edges.indices[:, 0]].add(ll)

        return result

    def __deepcopy__(self):
        child_layers = [child_layer.__deepcopy__() for child_layer in self.child_layers]
        edges = copy_bcoo(self.edges)
        return self.__class__(child_layers, edges)

@dataclass
class NXConverterLayer:
    """
    Class used for conversion from a probabilistic circuit in networkx to a layered circuit in jax.
    """
    layer: Layer
    nodes: List[ProbabilisticCircuitMixin]
    hash_remap: Dict[int, int]