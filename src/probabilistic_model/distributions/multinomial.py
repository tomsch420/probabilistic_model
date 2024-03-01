import itertools
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np

from random_events.variables import Discrete, Integer, Symbolic, Variable
from random_events.events import EncodedEvent

from ..probabilistic_model import ProbabilisticModel
from ..utils import SubclassJSONSerializer
from typing_extensions import Self, Any

from ..probabilistic_circuit.probabilistic_circuit import (ProbabilisticCircuit, DeterministicSumUnit,
                                                          DecomposableProductUnit)
from ..probabilistic_circuit.distributions.distributions import SymbolicDistribution, IntegerDistribution


class MultinomialDistribution(ProbabilisticModel, SubclassJSONSerializer):
    """
    A multinomial distribution over discrete random variables.
    """

    variables: Tuple[Discrete, ...]
    """
    The variables in the distribution.
    """

    probabilities: np.ndarray
    """
    The probability mass function. The dimensions correspond to the variables in the same order.
    The first dimension indexes over the first variable and so on. If no probabilities are provided in the constructor,
    the probabilities are initialized with ones.
    """

    def __init__(self, variables: Iterable[Discrete], probabilities: Optional[np.ndarray] = None):
        super().__init__(variables)

        shape = tuple(len(variable.domain) for variable in self.variables)

        if probabilities is None:
            probabilities = np.ones(shape)

        if shape != probabilities.shape:
            raise ValueError("The number of variables must match the number of dimensions in the probability array."
                             "Variables: {}".format(self.variables), "Dimensions: {}".format(probabilities.shape))

        self.probabilities = probabilities

    def marginal(self, variables: Iterable[Discrete]) -> 'MultinomialDistribution':

        # calculate which variables to marginalize over as the difference between variables and self.variables
        axis = tuple(self.variables.index(variable) for variable in self.variables if variable not in variables)

        # marginalize the probabilities over the axis
        probabilities = np.sum(self.probabilities, axis=axis)

        return MultinomialDistribution(variables, probabilities)

    def _mode(self) -> Tuple[List[EncodedEvent], float]:
        likelihood = np.max(self.probabilities)
        events = np.transpose(np.asarray(self.probabilities == likelihood).nonzero())
        mode = [EncodedEvent(zip(self.variables, event)) for event in events.tolist()]
        return mode, likelihood

    def __copy__(self) -> 'MultinomialDistribution':
        """
        :return: a shallow copy of the distribution.
        """
        return MultinomialDistribution(self.variables, self.probabilities)

    def __eq__(self, other: 'MultinomialDistribution') -> bool:
        """Compare self with other and return the boolean result.

        Two discrete random variables are equal only if the probability mass
        functions are equal and the order of dimensions are equal.

        """
        return (isinstance(other, self.__class__) and self.variables == other.variables and
                self.probabilities.shape == other.probabilities.shape and
                np.allclose(self.probabilities, other.probabilities))

    def __str__(self):
        return "P({}): \n".format(", ".join(var.name for var in self.variables)) + str(self.probabilities)

    def to_tabulate(self) -> List[List[str]]:
        """
        :return: a pretty table of the distribution.
        """
        columns = [[var.name for var in self.variables] + ["P"]]
        events: List[List] = list(list(event) for event in itertools.product(*[var.domain for var in self.variables]))

        for idx, event in enumerate(events):
            events[idx].append(self.likelihood(event))
        table = columns + events
        return table

    def _probability(self, event: EncodedEvent) -> float:
        indices = tuple(event[variable] for variable in self.variables)
        return self.probabilities[np.ix_(*indices)].sum()

    def _likelihood(self, event: List[int]) -> float:
        return float(self.probabilities[tuple(event)])

    def _conditional(self, event: EncodedEvent) -> Tuple[Optional[Self], float]:
        indices = tuple(event[variable] for variable in self.variables)
        indices = np.ix_(*indices)
        probabilities = np.zeros_like(self.probabilities)
        probabilities[indices] = self.probabilities[indices]
        return MultinomialDistribution(self.variables, probabilities), self.probabilities[indices].sum()

    def normalize(self) -> Self:
        """
        Normalize the distribution.
        :return: The normalized distribution
        """
        normalized_probabilities = self.probabilities / np.sum(self.probabilities)
        return MultinomialDistribution(self.variables, normalized_probabilities)

    def as_probabilistic_circuit(self) -> DeterministicSumUnit:
        """
        Convert this distribution to a probabilistic circuit. A deterministic sum unit with decomposable children is
        used to describe every state. The size of the circuit is equal to the size of `self.probabilities`.

        :return: The distribution as a probabilistic circuit.
        """
        # initialize the result as a deterministic sum unit
        result = DeterministicSumUnit()

        # iterate through all states of this distribution
        for event in itertools.product(*[variable.domain for variable in self.variables]):

            # create a product unit for the current state
            product_unit = DecomposableProductUnit()

            # iterate through all variables
            for variable, value in zip(self.variables, event):

                # create probabilities for the current variables state as one hot encoding
                weights = [0.] * len(variable.domain)
                weights[variable.encode(value)] = 1.

                # create a distribution for the current variable
                if isinstance(variable, Integer):
                    distribution = IntegerDistribution(variable, weights)
                elif isinstance(variable, Symbolic):
                    distribution = SymbolicDistribution(variable, weights)
                else:
                    raise ValueError(f"Variable type {type(variable)} not supported.")

                # mount the distribution to the product unit
                product_unit.add_subcircuit(distribution)

            # calculate the probability of the current state
            probability = self.likelihood(event)

            # mount the product unit to the result
            result.add_subcircuit(product_unit, probability)

        return result

    def encode_full_evidence_event(self, event: Iterable) -> List[int]:
        """
        Encode a full evidence event into a list of integers.
        :param event: The event to encode.
        :return: The encoded event.
        """
        return [variable.encode(value) for variable, value in zip(self.variables, event)]

    def fit(self, data: Iterable[Iterable[Any]]) -> Self:
        """
        Fit the distribution to the data.
        :param data: The data to fit the distribution to.
        :return: The fitted distribution.
        """
        encoded_data = np.zeros((len(data), len(self.variables)), dtype=int)
        for index, sample in enumerate(data):
            indices = self.encode_full_evidence_event(sample)
            encoded_data[index] = indices

        return self._fit(encoded_data)

    def _fit(self, data: np.ndarray) -> Self:
        probabilities = np.zeros_like(self.probabilities)
        uniques, counts = np.unique(data, return_counts=True, axis=0)

        for unique, count in zip(uniques.astype(int), counts):
            probabilities[tuple(unique)] = count

        self.probabilities = probabilities / probabilities.sum()
        return self

    def to_json(self) -> Dict[str, Any]:
        return {**SubclassJSONSerializer.to_json(self),
                "variables": [variable.to_json() for variable in self.variables],
                "probabilities": self.probabilities.tolist()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variables = [Variable.from_json(variable) for variable in data["variables"]]
        probabilities = np.array(data["probabilities"])
        return cls(variables, probabilities)