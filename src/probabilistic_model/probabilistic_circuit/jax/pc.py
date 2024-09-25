from abc import abstractmethod, ABC
from functools import cached_property

import jax

from jax import numpy as jnp
import equinox as eqx
from jax.experimental.sparse import BCOO, bcoo_concatenate, bcoo_reduce_sum
from typing_extensions import List, Iterator, Tuple
from jax.scipy.special import logsumexp
from .utils import copy_bcoo


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


class InnerLayer(Layer):
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
    def variables(self) -> jax.Array:
        return jnp.array(sorted(set().union(*[child_layer.variables for child_layer in self.child_layers])))


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


class SumLayer(eqx.Module):
    log_weights: List[BCOO]
    child_layers: List[eqx.Module]

    def __init__(self, child_layers: List[eqx.Module], log_weights: List[BCOO]):
        super().__init__()
        self.log_weights = log_weights
        self.child_layers = child_layers


    def validate(self):
        for log_weights in self.log_weights:
            assert log_weights.shape[0] == self.number_of_nodes, "The number of nodes must match the number of weights."

        for log_weights, child_layer in self.log_weighted_child_layers:
            assert log_weights.shape[
                       1] == child_layer.number_of_nodes, "The number of nodes must match the number of weights."

    @property
    def log_weighted_child_layers(self) -> Iterator[Tuple[BCOO, eqx.Module]]:
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

    @eqx.filter_jit
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