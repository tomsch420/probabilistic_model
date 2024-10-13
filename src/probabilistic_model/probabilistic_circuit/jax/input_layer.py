from __future__ import annotations
from abc import ABC
from typing import List, Dict, Any

import jax
import numpy as np
from jax import numpy as jnp
from jax.experimental.array_api import reshape
from jax.experimental.sparse import BCOO, BCSR
from random_events.interval import Interval, SimpleInterval
from random_events.product_algebra import SimpleEvent
from typing_extensions import Tuple, Type, Self, Union, Optional

from . import create_bcoo_indices_from_row_lengths, create_bcsr_indices_from_row_lengths, \
    embed_sparse_array_in_nan_array
from .inner_layer import InputLayer, NXConverterLayer, SumLayer
from ..nx.distributions import DiracDeltaDistribution
import equinox as eqx
from .utils import simple_interval_to_open_array, remove_rows_and_cols_where_all


class ContinuousLayer(InputLayer, ABC):
    """
    Abstract base class for continuous univariate input units.
    """

    def probability_of_simple_event(self, event:SimpleEvent) -> jax.Array:
        interval: Interval = list(event.values())[self.variables[0]]
        return self.probability_of_interval(interval)

    def probability_of_interval(self, interval: Interval) -> jnp.array:
        points = jnp.array([simple_interval_to_open_array(i) for i in interval.simple_sets])
        upper_bound_cdf = self.cdf_of_nodes(points[:, (1,)])
        lower_bound_cdf = self.cdf_of_nodes(points[:, (0,)])
        return (upper_bound_cdf - lower_bound_cdf).sum(axis=0)

    def probability_of_simple_interval(self, interval: SimpleInterval) -> jax.Array:
        points = simple_interval_to_open_array(interval)
        upper_bound_cdf = self.cdf_of_nodes_single(points[1])
        lower_bound_cdf = self.cdf_of_nodes_single(points[0])
        return upper_bound_cdf - lower_bound_cdf

    def log_conditional_of_simple_event(self, event: SimpleEvent) -> Tuple[Optional[Union[Self, DiracDeltaLayer]], jax.Array]:
        if event.is_empty():
            return self.impossible_condition_result

        interval: Interval = list(event.values())[self.variable]

        if interval.is_singleton():
            return self.log_conditional_from_singleton(interval.simple_sets[0])

        if len(interval.simple_sets) == 1:
            return self.log_conditional_from_simple_interval(interval.simple_sets[0])
        else:
            return self.log_conditional_from_interval(interval)

    def log_conditional_from_singleton(self, singleton: SimpleInterval) -> Tuple[DiracDeltaLayer, jax.Array]:
        """
        Calculate the conditional distribution given a singleton interval.

        In this case, the conditional distribution is a Dirac delta distribution and the log-likelihood is chosen
        instead of the log-probability.

        This method returns a Dirac delta layer that has at most the same number of nodes as the input layer.

        :param singleton: The singleton event
        :return: The dirac delta layer and the log-likelihoods with shape (something <= #singletons, 1).
        """
        value = singleton.lower
        log_likelihoods = self.log_likelihood_of_nodes(
            jnp.array(value).reshape(-1, 1))[:, 0]  # shape: (#nodes, )

        possible_indices = (log_likelihoods > -jnp.inf).nonzero()[0]  # shape: (#dirac-nodes, )
        filtered_likelihood = log_likelihoods[possible_indices]
        locations = jnp.full_like(filtered_likelihood, value)
        layer = DiracDeltaLayer(self.variable, locations, jnp.exp(filtered_likelihood))
        return layer, log_likelihoods

    def log_conditional_from_simple_interval(self, interval: SimpleInterval) -> Tuple[Self, jax.Array]:
        """
        Calculate the conditional distribution given a simple interval with p(interval) > 0.
        The interval could also be a singleton.

        :param interval: The simple interval
        :return: The conditional distribution and the log-probability of the interval.
        """
        raise NotImplementedError

    def log_conditional_from_interval(self, interval: Interval) -> Tuple[SumLayer, jax.Array]:
        """
        Calculate the conditional distribution given an interval with p(interval) > 0.

        :param interval: The simple interval
        :return: The conditional distribution and the log-probability of the interval.
        """

        # get conditionals of each simple interval
        results = [self.log_conditional_from_simple_interval(simple_interval) for simple_interval in
                   interval.simple_sets]

        layers, log_probs = zip(*results)

        # stack the log probabilities
        stacked_log_probabilities = jnp.stack(log_probs, axis=1)  # shape: (#simple_intervals, #nodes)

        # calculate the log probabilities of the entire interval
        exp_stacked_log_probabilities = jnp.exp(stacked_log_probabilities)
        summed_exp_stacked_log_probabilities = jnp.sum(exp_stacked_log_probabilities, axis=1)
        total_log_probabilities = jnp.log(summed_exp_stacked_log_probabilities)  # shape: (#nodes, 1)


        # create new input layer
        possible_layers = [layer for layer in layers if layer is not None]
        input_layer = possible_layers[0]
        input_layer = input_layer.merge_with(possible_layers[1:])

        # remove the rows that are entirely -inf and normalize weights
        bcoo_data = remove_rows_and_cols_where_all(exp_stacked_log_probabilities/
                                                   summed_exp_stacked_log_probabilities.reshape(-1, 1),
                                                   0)

        log_weights = BCOO.fromdense(bcoo_data)
        log_weights.data = jnp.log(log_weights.data)

        resulting_layer = SumLayer([input_layer], [log_weights])
        return resulting_layer, total_log_probabilities


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

    def __deepcopy__(self):
        return self.__class__(self.variables[0].item(), self.interval.copy())


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

    def validate(self):
        assert len(self.location) == len(self.density_cap), "The number of locations and density caps must match."

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

    def log_conditional_from_simple_interval(self, interval: SimpleInterval) -> Tuple[Self, jax.Array]:
        probabilities = jnp.log(self.probability_of_simple_interval(interval))

        valid_probabilities = probabilities > -jnp.inf

        if not valid_probabilities.any():
            return self.impossible_condition_result

        result = self.__class__(self.variable, self.location[valid_probabilities],
                                self.density_cap[valid_probabilities])
        return result, probabilities

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["location"] = self.location.tolist()
        result["density_cap"] = self.density_cap.tolist()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(data["variable"], jnp.array(data["location"]), jnp.array(data["density_cap"]))

    def merge_with(self, others: List[Self]) -> Self:
        return self.__class__(self.variable, jnp.concatenate([self.location] + [other.location for other in others]),
                                jnp.concatenate([self.density_cap] + [other.density_cap for other in others]))