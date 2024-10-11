from abc import ABC
from typing import List, Dict, Any

import jax
import numpy as np
from jax import numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
from random_events.interval import Interval
from random_events.product_algebra import SimpleEvent
from typing_extensions import Tuple, Type, Self

from . import create_bcoo_indices_from_row_lengths, create_bcsr_indices_from_row_lengths
from .inner_layer import InputLayer, NXConverterLayer
from ..nx.distributions import DiracDeltaDistribution
import equinox as eqx
from .utils import simple_interval_to_open_array


class ContinuousLayer(InputLayer, ABC):
    """
    Abstract base class for continuous univariate input units.
    """

    def probability_of_simple_event(self, event:SimpleEvent) -> jnp.array:
        interval: Interval = list(event.values())[self.variables[0]]
        points = jnp.array([simple_interval_to_open_array(i) for i in interval.simple_sets])
        upper_bound_cdf = self.cdf_of_nodes(points[:, (1,)])
        lower_bound_cdf = self.cdf_of_nodes(points[:, (0,)])
        return (upper_bound_cdf - lower_bound_cdf).sum(axis=0)


class ContinuousLayerWithFiniteSupport(ContinuousLayer, ABC):
    """
    Abstract class for continuous univariate input units with finite support.
    """

    interval: jax.Array = eqx.field(static=True)
    """
    The interval of the distribution as a array of shape (num_nodes, 2).
    The first column contains the lower bounds and the second column the upper bounds.
    The intervals are treated as open intervals (>/< comparator).
    """

    def __init__(self, variable: int, interval: jax.Array):
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

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["interval"] = self.interval.tolist()
        return result


class DiracDeltaLayer(ContinuousLayer):

    location: jax.Array = eqx.field(static=True)
    """
    The locations of the Dirac delta distributions.
    """

    density_cap: jax.Array = eqx.field(static=True)
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

    def log_likelihood_of_nodes(self, x: jax.Array) -> jax.Array:
        return jax.vmap(self.log_likelihood_of_nodes_single)(x)

    def log_likelihood_of_nodes_single(self, x: jax.Array) -> jax.Array:
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
        locations = jnp.array([node.location for node in nodes], dtype=jnp.float32)
        density_caps = jnp.array([node.density_cap for node in nodes], dtype=jnp.float32)
        result = cls(nodes[0].probabilistic_circuit.variables.index(nodes[0].variable), locations, density_caps)
        return NXConverterLayer(result, nodes, hash_remap)

    def sample_from_frequencies(self, frequencies: np.array, result: np.array, start_index = 0):
        values = self.location.repeat(frequencies).reshape(-1, 1)
        result[start_index:start_index + len(values), self.variables] = values

    def cdf_of_nodes_single(self, x: jnp.array) -> jnp.array:
        return jnp.where(x < self.location, 0., 1.)

    def moment_of_nodes(self, order: jax.Array, center: jax.Array):
        order = order[self.variables[0]]
        center = center[self.variables[0]]
        if order == 0:
            result = jnp.ones(self.number_of_nodes)
        elif order == 1:
            result = self.location - center
        else:
            result = jnp.zeros(self.number_of_nodes)
        return result.reshape(-1, 1)

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["location"] = self.location.tolist()
        result["density_cap"] = self.density_cap.tolist()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(data["variable"], jnp.array(data["location"]), jnp.array(data["density_cap"]))