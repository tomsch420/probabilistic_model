from abc import ABC
from typing import List

import jax
from jax import numpy as jnp
from typing_extensions import Tuple, Type

from .inner_layer import InputLayer, NXConverterLayer
from ..nx.distributions import DiracDeltaDistribution
from ..nx.probabilistic_circuit import ProbabilisticCircuitMixin


class ContinuousLayer(InputLayer, ABC):
    """
    Abstract base class for continuous univariate input units.
    """


class ContinuousLayerWithFiniteSupport(ContinuousLayer, ABC):
    """
    Abstract class for continuous univariate input units with finite support.
    """

    interval: jax.Array
    """
    The interval of the distribution as a array of shape (num_nodes, 2).
    The first column contains the lower bounds and the second column the upper bounds.
    The intervals are treated as open intervals (>/< comparator).
    """

    def __init__(self, variable: int, interval):
        super().__init__(variable)
        self.interval = interval

    @property
    def lower(self) -> jax.Array:
        return self.interval[:, 0]

    @property
    def upper(self) -> jax.Array:
        return self.interval[:, 1]

    def left_included_condition(self, x: jax.Array) -> jax.Array:
        """
        Check if x is included in the left bound of the intervals.
        :param x: The data
        :return: A boolean array of shape (#x, #nodes)
        """
        return self.lower < x

    def right_included_condition(self, x: jax.Array) -> jax.Array:
        """
         Check if x is included in the right bound of the intervals.
         :param x: The data
         :return: A boolean array of shape (#x, #nodes)
         """
        return x < self.upper

    def included_condition(self, x: jax.Array) -> jax.Array:
        """
         Check if x is included in the interval.
         :param x: The data
         :return: A boolean array of shape (#x, #nodes)
         """
        return self.left_included_condition(x) & self.right_included_condition(x)


class DiracDeltaLayer(ContinuousLayer):

    location: jax.Array
    """
    The locations of the Dirac delta distributions.
    """

    density_cap: jax.Array
    """
    The density caps of the Dirac delta distributions.
    This value will be used to replace infinity in likelihoods.
    """

    def __init__(self, variable_index, location, density_cap):
        super().__init__(variable_index)
        self.location = location
        self.density_cap = density_cap

    @property
    def number_of_nodes(self):
        return len(self.location)

    @jax.jit
    def log_likelihood_of_nodes(self, x: jax.Array) -> jax.Array:
        return jnp.where(x == self.location, jnp.log(self.density_cap), -jnp.inf)

    @classmethod
    def nx_classes(cls) -> Tuple[Type, ...]:
        return DiracDeltaDistribution,

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(cls, nodes: List[DiracDeltaDistribution],
                                                         child_layers: List[NXConverterLayer],
                                                         progress_bar: bool = True) -> \
            NXConverterLayer:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        locations = jnp.array([node.location for node in nodes], dtype=jnp.double)
        density_caps = jnp.array([node.density_cap for node in nodes], dtype=jnp.double)
        result = cls(nodes[0].probabilistic_circuit.variables.index(nodes[0].variable), locations, density_caps)
        return NXConverterLayer(result, nodes, hash_remap)