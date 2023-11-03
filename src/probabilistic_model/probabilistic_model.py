import abc
from typing import Tuple, Iterable, List

from random_events.variables import Variable
from random_events.events import Event, EncodedEvent


class ProbabilisticModel(abc.ABC):
    """
    Abstract base class for probabilistic models.

    The definition of events follows the definition of events in the random_events package.
    The definition of functions is motivated by the background knowledge provided in the probabilistic circuits.

    This class can be used as an interface to any kind of probabilistic model, tractable or not.
    The methods follow the pattern that methods that bein with `_` use a preprocessed version of the original method.
    This is useful to separating the process of parsing user inputs and the actual calculations.
    """

    variables: Tuple[Variable]
    """The variables involved in the model."""

    def __init__(self, variable: Iterable[Variable]):
        """
        Initialize the model.

        :param variable: The variables in the model.
        """
        self.variables = tuple(sorted(variable))

    def _likelihood(self, event: Iterable) -> float:
        """
        Calculate the likelihood of a preprocessed event.
        The likelihood as a full evidence query, i.e. an assignment to all variables in the model
        :param event: The event is some iterable that represents a value for each variable in the model.
        :return: The likelihood of the event.
        """
        raise NotImplementedError

    def likelihood(self, event: Iterable) -> float:
        """
        Calculate the likelihood of an event.
        The likelihood as a full evidence query, i.e. an assignment to all variables in the model

        The event belongs to the class of full evidence queries.

        .. note:: You can read more about queries of this class in Definition 1 of this
            `article <http://starai.cs.ucla.edu/papers/ProbCirc20.pdf>`_.


        :param event: The event is some iterable that represents a value for each variable in the model.
        :return: The likelihood of the event.
        """
        raise self._likelihood([variable.encode(value) for variable, value in zip(self.variables, event)])

    def _probability(self, event: EncodedEvent) -> float:
        """
        Calculate the probability of a preprocessed event P(E).

        :param event: The event to calculate the probability of.
        :return: The probability of the model.
        """
        raise NotImplementedError

    def probability(self, event: Event) -> float:
        """
        Calculate the probability of an event P(E).
        The event belongs to the class of marginal queries.

        .. note:: You can read more about queries of this class in Definition 11 of this
            `article <http://starai.cs.ucla.edu/papers/ProbCirc20.pdf>`_.

        :param event: The event to calculate the probability of.
        :return: The probability of the model.
        """
        return self.probability(event.encode())

    def _mode(self) -> Tuple[Iterable[EncodedEvent], float]:
        """
        Calculate the mode of the model.
        As there may exist multiple modes, this method returns an iterable of modes and their likelihood.

        :return: The internal representation of the mode of the model amd the likelihood.
        """
        raise NotImplementedError

    def mode(self) -> Tuple[List[Event], float]:
        """
        Calculate the mode of the model.
        As there may exist multiple modes, this method returns an iterable of modes and their likelihood.
        The event belongs to the map query class.

        .. note:: You can read more about queries of this class in Definition 26 of this
            `article <http://starai.cs.ucla.edu/papers/ProbCirc20.pdf>`_.

        :return: The mode of the model.
        """
        mode, likelihood = self._mode()
        return list(event.decode() for event in mode), likelihood

    def marginal(self, variables: Iterable[Variable]) -> 'ProbabilisticModel':
        """
        Calculate the marginal distribution of a set of variables.

        :param variables: The variables to calculate the marginal distribution on.
        :return: The marginal distribution of the variables.
        """
        raise NotImplementedError

    def _conditional(self, event: EncodedEvent) -> 'ProbabilisticModel':
        """
        Calculate the conditional distribution of the model given a preprocessed event.

        :param event: The event to condition on.
        :return: The conditional distribution of the model.
        """
        raise NotImplementedError

    def conditional(self, event: Event) -> 'ProbabilisticModel':
        """
        Calculate the conditional distribution of the model given an event.

        The event belongs to the class of marginal queries.

        .. note:: You can read more about queries of this class in Definition 11 of this
            `article <http://starai.cs.ucla.edu/papers/ProbCirc20.pdf>`_.

        :param event: The event to condition on.
        :return: The conditional distribution of the model.
        """
        return self._conditional(event.encode())

    def sample(self) -> Iterable:
        """
        Sample from the model.
        :return: The samples
        """
        raise NotImplementedError
