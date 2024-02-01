import itertools
from typing import Iterable, List, Optional, Tuple

import numpy as np

from random_events.variables import Discrete
from random_events.events import EncodedEvent

from ..probabilistic_model import ProbabilisticModel
from typing_extensions import Self


class MultinomialDistribution(ProbabilisticModel):
    """
    A multinomial distribution over discrete random variables.
    """

    variables: Tuple[Discrete]
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

    def normalize(self) -> 'MultinomialDistribution':
        """
        Normalize the distribution.
        :return: The normalized distribution
        """
        normalized_probabilities = self.probabilities / np.sum(self.probabilities)
        return MultinomialDistribution(self.variables, normalized_probabilities)