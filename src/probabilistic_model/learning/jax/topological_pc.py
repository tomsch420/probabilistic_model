from __future__ import annotations

from abc import abstractmethod, ABC

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy.stats.norm


class TopologicalCircuit:
    """
    Implements the topological Approach to circuit from Anji using JAX.
    """


class Layer(nn.Module):

    @abstractmethod
    def log_likelihood(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.log_likelihood(*args, **kwargs)


class IntermediateLayer(Layer, ABC):

    next_layer: Layer

    edge_mask: jax.Array
    """
    A boolean array that tells us which nodes are connected to which.
    The shape of the array is (n, m) where n is the number of nodes in the current layer and m is the number of nodes 
    in the next layer.
    """


class SumLayer(IntermediateLayer):
    """
    ..Note::

    A Sum Layer cannot change the dimensionality of the input from next_layer.
    For instance, if an input of shape (100, 2) comes in, the output shape must be (100, 2).

    """

    log_weights: jax.Array
    """
    The log weights of the edges between the current layer and the next layer.
    The shape of the array is (n, m) where n is the number of nodes in the current layer and m is the number of nodes
    in the next layer.
    """

    def log_likelihood(self, x: jax.Array) -> jax.Array:
        # apply mask to weights
        masked_weights = self.edge_mask * self.log_weights

        # normalize weights
        masked_weights = masked_weights - jax.scipy.special.logsumexp(masked_weights, axis=-1)

        # calculate weighted log likelihoods
        return jax.scipy.special.logsumexp(masked_weights + self.next_layer.log_likelihood(x), axis=-1)


class ProductLayer(IntermediateLayer):

    def log_likelihood(self, x: jax.Array) -> jax.Array:
        return self.next_layer.log_likelihood(x) + jnp.sum(jnp.where(self.edge_mask, 0.0, -jnp.inf), axis=-1)


class UniformLayer(Layer):
    lower: jax.Array
    upper: jax.Array

    def log_likelihood(self, x: jax.Array) -> jax.Array:
        include_condition = jnp.logical_and(self.lower <= x, x <= self.upper)
        log_likelihoods = jnp.where(include_condition, -jnp.log(self.upper - self.lower), -jnp.inf)
        return log_likelihoods


class NormalLayer(Layer):
    location: jax.Array
    scale: jax.Array
    min_scale: jax.Array = 1e-6

    def log_likelihood(self, x: jax.Array) -> jax.Array:
        return jax.scipy.stats.norm.logpdf(x, loc=self.location, scale=jnp.exp(self.scale) + self.min_scale)
