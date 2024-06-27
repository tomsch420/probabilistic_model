from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import List

import jax
import networkx as nx
from random_events.interval import SimpleInterval
from random_events.variable import Continuous
from typing_extensions import Self, Type, Dict
from typing import NamedTuple

from probabilistic_model.probabilistic_circuit.probabilistic_circuit import (SumUnit as PCSumUnit,
                                                                             ProbabilisticCircuit as PMProbabilisticCircuit,
                                                                             ProductUnit as PCProductUnit,
                                                                             ProbabilisticCircuitMixin)
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution as PCUniformDistribution
import equinox
import jax.numpy as jnp
from jax import Array
from random_events.utils import recursive_subclasses


def inverse_class_of(clazz: Type[ProbabilisticCircuitMixin]) -> Type[ModuleMixin]:
    for subclass in recursive_subclasses(ModuleMixin):
        if issubclass(clazz, subclass.origin_class()):
            return subclass
    raise TypeError(f"Could not find class for {clazz}")


class TunableParameters(NamedTuple):
    """
    Class that describes the tunable parameters of the module
    """
    ...


StateDictType = Dict[int, TunableParameters]


class ModuleMixin:
    """
    Mixin for JAX modules that are capable of being converted to the original probabilistic circuit module.
    JAX modules are limited in functionality, as only the log_likelihood method is supported and automatically
    differentiable.
    """

    probabilistic_circuit: ProbabilisticCircuit

    tunable_parameters: TunableParameters
    """
    The instance of parameters
    """

    @staticmethod
    @abstractmethod
    def origin_class() -> Type[ProbabilisticCircuitMixin]:
        """
        The original class of the module.

        :return: The original class of the module.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_unit(cls, unit: ProbabilisticCircuitMixin, probabilistic_circuit: ProbabilisticCircuit) -> Self:
        """
        Create a new instance of this class from a unit.

        :param unit: The unit to read the parameters from.
        :param probabilistic_circuit: The probabilistic circuit where the unit should be added.
        :return: The jax version of the unit.
        """
        raise NotImplementedError

    def log_likelihood(self, x: Array):
        """
        Calculate p(x)
        :param x:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def log_likelihood_given_state(self, state_dict: StateDictType,  x: Array):
        """
        Calculate p(x|theta=state)
        :param x: The data
        :param state_dict: The state (parameters)
        :return: The log likelihood of each datapoint
        """
        raise NotImplementedError

    def average_negative_log_likelihood(self, state_dict: StateDictType, x: Array):
        log_likelihood = self.log_likelihood_given_state(state_dict, x)
        return jnp.mean(log_likelihood)

    def __hash__(self):
        return id(self)


class UniformDistribution(ModuleMixin, PCUniformDistribution):

    tunable_parameters = TunableParameters()

    def __init__(self, variable: Continuous, interval: SimpleInterval, probabilistic_circuit: ProbabilisticCircuit):
        super().__init__(variable, interval)
        self.probabilistic_circuit = probabilistic_circuit

    @staticmethod
    def origin_class() -> Type[PCUniformDistribution]:
        return PCUniformDistribution

    @classmethod
    def from_unit(cls, unit: PCUniformDistribution, probabilistic_circuit: ProbabilisticCircuit) -> Self:
        return cls(unit.variable, unit.interval, probabilistic_circuit)

    def log_pdf_value(self) -> Array:
        return -jnp.log(self.upper - self.lower)

    def log_likelihood_without_bounds_check(self, x: Array) -> Array:
        return jnp.full((x.shape[:-1]), self.log_pdf_value())

    def log_likelihood(self, x: Array) -> Array:
        result = jnp.full(x.shape[:-1], -jnp.inf)
        include_condition = self.included_condition(x)
        filtered_x = x[include_condition].reshape(-1, 1)
        likelihoods = self.log_likelihood_without_bounds_check(filtered_x)
        result = result.at[include_condition[:, 0]].set(likelihoods)
        return result

    def log_likelihood_given_state(self, state_dict: StateDictType, x: Array):
        return self.log_likelihood(x)


class SumUnit(PCSumUnit, ModuleMixin):

    class TunableParameters(NamedTuple):
        log_weights: Array

        @property
        def weights(self) -> Array:
            weights = jnp.exp(self.log_weights)
            return weights / weights.sum()

    probabilistic_circuit: ProbabilisticCircuit
    tunable_parameters: TunableParameters

    def __init__(self, initial_weights: Array, probabilistic_circuit: ProbabilisticCircuit):
        super().__init__()
        self.tunable_parameters = self.TunableParameters(jnp.log(initial_weights))
        self.probabilistic_circuit = probabilistic_circuit

    @cached_property
    def subcircuits(self) -> List[ModuleMixin]:
        return super().subcircuits

    @staticmethod
    def origin_class() -> Type[PCSumUnit]:
        return PCSumUnit

    @property
    def weights(self) -> Array:
        weights = jnp.exp(self.tunable_parameters.log_weights)
        return weights / weights.sum()

    def log_likelihood(self, events: Array) -> Array:
        return self.log_likelihood_given_state({hash(self): self.tunable_parameters}, events)

    def log_likelihood_given_state(self, state_dict: StateDictType,  x: Array):
        result = jnp.zeros(x.shape[:-1])
        for weight, subcircuit in zip(state_dict[hash(self)].weights, self.subcircuits):
            subcircuit_likelihood = jnp.exp(subcircuit.log_likelihood(x))
            result += weight * subcircuit_likelihood
        return jnp.log(result)

    def __call__(self, x):
        return self.log_likelihood(x)

    @classmethod
    def from_unit(cls, unit: PCSumUnit, probabilistic_circuit: ProbabilisticCircuit) -> Self:
        result = cls(unit.weights, probabilistic_circuit)
        return result

    def __hash__(self):
        return id(self)


class ProductUnit(ModuleMixin, PCProductUnit):

    @staticmethod
    def origin_class() -> Type[PCProductUnit]:
        return PCProductUnit


class ProbabilisticCircuit(PMProbabilisticCircuit):

    @cached_property
    def root(self) -> ModuleMixin:
        return super().root

    @classmethod
    def from_probabilistic_circuit(cls, probabilistic_circuit: PMProbabilisticCircuit) -> Self:

        node_remap: Dict = dict()

        result = cls()
        for node in probabilistic_circuit.nodes:
            jax_node = inverse_class_of(type(node)).from_unit(unit=node, probabilistic_circuit=result)
            nx.DiGraph.add_node(result, jax_node)
            node_remap[node] = jax_node

        for edge in probabilistic_circuit.edges:
            result.add_edge(node_remap[edge[0]], node_remap[edge[1]])
        return result

    def log_likelihood(self, events: Array) -> Array:
        return self.root.log_likelihood(events)

    def log_likelihood_given_state(self, state_dict: Dict[int, TunableParameters],  x: Array):
        return self.root.log_likelihood_given_state(state_dict, x)

    def tunable_parameters(self) -> Dict[int, TunableParameters]:
        return {hash(node): node.tunable_parameters for node in self.nodes}
