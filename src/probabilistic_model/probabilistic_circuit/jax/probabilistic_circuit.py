from __future__ import annotations

from abc import abstractmethod

import networkx as nx
import numpy as np
from equinox import AbstractVar
from networkx.classes.digraph import _CachedPropertyResetterPred
from random_events.interval import SimpleInterval
from random_events.variable import Continuous
from typing_extensions import Self, Type, Dict

from probabilistic_model.probabilistic_circuit.probabilistic_circuit import (SmoothSumUnit,
                                                                             ProbabilisticCircuit as PMProbabilisticCircuit,
                                                                             DecomposableProductUnit,
                                                                             ProbabilisticCircuitMixin)
from probabilistic_model.probabilistic_circuit.distributions import UniformDistribution as PCUniformDistribution
import equinox
import jax.numpy as jnp
from jax import Array
from random_events.utils import recursive_subclasses


class ProbabilisticCircuit(PMProbabilisticCircuit):

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


def inverse_class_of(clazz: Type[ProbabilisticCircuitMixin]) -> Type[ModuleMixin]:
    for subclass in recursive_subclasses(ModuleMixin):
        if issubclass(clazz, subclass.origin_class()):
            return subclass
    raise TypeError(f"Could not find class for {clazz}")


class ModuleMixin:
    """
    Mixin for JAX modules that are capable of being converted to the original probabilistic circuit module.
    JAX modules are limited in functionality, as only the log_likelihood method is supported and automatically
    differentiable.
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


class UniformDistribution(PCUniformDistribution, equinox.Module, ModuleMixin):

    variable: Continuous
    interval: SimpleInterval
    probabilistic_circuit: ProbabilisticCircuit = equinox.field(static=True)

    def __init__(self, variable: Continuous, interval: SimpleInterval, probabilistic_circuit: ProbabilisticCircuit):
        self.variable = variable
        self.interval = interval
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

    def __hash__(self):
        return id(self)


class SumUnit(equinox.Module, SmoothSumUnit, ModuleMixin):

    _weights: Array
    probabilistic_circuit: ProbabilisticCircuit = equinox.field(static=True)

    def __init__(self, initial_weights: Array, probabilistic_circuit: ProbabilisticCircuit):
        super().__init__()
        self._weights = initial_weights
        self.probabilistic_circuit = probabilistic_circuit

    @staticmethod
    def origin_class() -> Type[SmoothSumUnit]:
        return SmoothSumUnit

    @property
    def weights(self) -> Array:
        exp_weights = jnp.exp(self._weights)
        return exp_weights / exp_weights.sum()

    def log_likelihood(self, events: Array) -> Array:
        result = jnp.zeros(events.shape[:-1])
        for weight, subcircuit in zip(self.weights, self.subcircuits):
            subcircuit_likelihood = jnp.exp(subcircuit.log_likelihood(events))
            result += weight * subcircuit_likelihood
        return jnp.log(result)

    def __call__(self, x):
        return self.log_likelihood(x)

    @classmethod
    def from_unit(cls, unit: SmoothSumUnit, probabilistic_circuit: ProbabilisticCircuit) -> Self:
        result = cls(jnp.log(unit.weights), probabilistic_circuit)
        return result

    def __hash__(self):
        return id(self)


class ProductUnit(equinox.Module, DecomposableProductUnit, ModuleMixin):
    probabilistic_circuit: ProbabilisticCircuit = equinox.field(static=True)

    @staticmethod
    def origin_class() -> Type[DecomposableProductUnit]:
        return DecomposableProductUnit


