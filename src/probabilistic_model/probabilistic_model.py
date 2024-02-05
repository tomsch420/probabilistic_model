import abc
from typing import Tuple, Iterable, List, Optional, Union, TYPE_CHECKING

from random_events.events import Event, EncodedEvent, VariableMap
from random_events.variables import Variable, Integer, Continuous
from typing_extensions import Self

# # Type hinting for Python 3.7 to 3.9
if TYPE_CHECKING:
    OrderType = VariableMap[Union[Integer, Continuous], int]
    CenterType = VariableMap[Union[Integer, Continuous], float]
    MomentType = VariableMap[Union[Integer, Continuous], float]
else:
    OrderType = VariableMap
    CenterType = VariableMap
    MomentType = VariableMap


class ProbabilisticModel(abc.ABC):
    """
    Abstract base class for probabilistic models.

    The definition of events follows the definition of events in the random_events package.
    The definition of functions is motivated by the background knowledge provided in the probabilistic circuits.

    This class can be used as an interface to any kind of probabilistic model, tractable or not.
    The methods follow the pattern that methods that begin with `_` use a preprocessed version of the original method.
    This is useful to separating the process of parsing user inputs and the actual calculations.
    """

    _variables: Tuple[Variable, ...]
    """The variables involved in the model."""

    def __init__(self, variables: Optional[Iterable[Variable]]):
        """
        Initialize the model.

        :param variables: The variables in the model.
        """

        if variables is not None:
            self._variables = tuple(sorted(variables))

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return self._variables

    @variables.setter
    def variables(self, variables: Iterable[Variable]):
        self._variables = tuple(sorted(variables))

    def preprocess_event(self, event: Event) -> EncodedEvent:
        """
        Preprocess an event to the internal representation of the model.
        Furthermore, all variables that are in this model are assigned to some value.
        If the value is specified in the event, the value is used; otherwise the domain of the variable is used.

        :param event: The event to preprocess.
        :return: The preprocessed event.
        """
        return (Event({variable: variable.domain for variable in self.variables}) & event).encode()

    def _likelihood(self, event: Iterable) -> float:
        """
        Calculate the likelihood of a preprocessed event.
        The likelihood as a full evidence query, i.e., an assignment to all variables in the model

        :param event: The event is some iterable that represents a value for each variable in the model.
        :return: The likelihood of the event.
        """
        raise NotImplementedError

    def likelihood(self, event: Iterable) -> float:
        """
        Calculate the likelihood of an event.
        The likelihood is a full evidence query, i.e., an assignment to all variables in the model

        The event belongs to the class of full evidence queries.

        .. Note:: You can read more about queries of this class in Definition 1 in :cite:p:`choi2020probabilistic`
            or watch the `video tutorial <https://youtu.be/2RAG5-L9R70?si=TAfIX2LmOWM-Fd2B&t=785>`_.
            :cite:p:`youtube2020probabilistic`


        :param event: The event is some iterable that represents a value for each variable in the model.
        :return: The likelihood of the event.
        """

        return self._likelihood([variable.encode(value) for variable, value in zip(self.variables, event)])

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

        .. Note:: You can read more about queries of this class in Definition 11 in :cite:p:`choi2020probabilistic`
            or watch the `video tutorial <https://youtu.be/2RAG5-L9R70?si=8aEGIqmoDTiUR2u6&t=1089>`_.
            :cite:p:`youtube2020probabilistic`

        :param event: The event to calculate the probability of.
        :return: The probability of the model.
        """
        return self._probability(self.preprocess_event(event))

    def _mode(self) -> Tuple[Iterable[EncodedEvent], float]:
        """
        Calculate the mode of the model.
        As there may exist multiple modes, this method returns an Iterable of modes and their likelihood.

        :return: The internal representation of the mode and the likelihood.
        """
        raise NotImplementedError

    def mode(self) -> Tuple[List[Event], float]:
        """
        Calculate the mode of the model.
        As there may exist multiple modes, this method returns an Iterable of modes and their likelihood.
        The event belongs to the map query class.

        .. Note:: You can read more about queries of this class in Definition 26 in :cite:p:`choi2020probabilistic`
            or watch the `video tutorial <https://youtu.be/2RAG5-L9R70?si=FjREKNtAV0owm27A&t=1962>`_.
            :cite:p:`youtube2020probabilistic`

        :return: The mode of the model and the likelihood.
        """
        mode, likelihood = self._mode()
        return list(event.decode() for event in mode), likelihood

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        """
        Calculate the marginal distribution of a set of variables.

        :param variables: The variables to calculate the marginal distribution on.
        :return: The marginal distribution of the variables.
        """
        raise NotImplementedError

    def _conditional(self, event: EncodedEvent) -> Tuple[Optional[Self], float]:
        """
        Calculate the conditional distribution of the model given a preprocessed event.

        If the event is impossible, the conditional distribution is None and the probability is 0.

        :param event: The event to condition on.
        :return: The conditional distribution of the model and the probability of the event.
        """
        raise NotImplementedError

    def conditional(self, event: Event) -> Tuple[Optional[Self], float]:
        """
        Calculate the conditional distribution of the model given an event.

        The event belongs to the class of marginal queries.

        If the event is impossible, the conditional distribution is None and the probability is 0.

        .. Note:: You can read more about queries of this class in Definition 11 in :cite:p:`choi2020probabilistic`_
            or watch the `video tutorial <https://youtu.be/2RAG5-L9R70?si=8aEGIqmoDTiUR2u6&t=1089>`_.
            :cite:p:`youtube2020probabilistic`

        :param event: The event to condition on.
        :return: The conditional distribution of the model and the probability of the event.
        """
        return self._conditional(self.preprocess_event(event))

    def sample(self, amount: int) -> Iterable:
        """
        Sample from the model.

        :param amount: The number of samples to draw.
        :return: The samples
        """
        raise NotImplementedError

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        """
        Calculate the (centralized) moment of the distribution.

        .. math::

            \int_{-\infty}^{\infty} (x - center)^{order} pdf(x) dx

        .. Note:: You can read more about queries of this class in Definition 22 in :cite:p:`choi2020probabilistic`_.
            :cite:p:`youtube2020probabilistic`


        :param order: The orders of the moment as a variable map for every continuous and integer variable.
        :param center: The center of the moment as a variable map for every continuous and integer variable.
        :return: The moments of the variables in `order`.
        """
        raise NotImplementedError

    def expectation(self, variables: Iterable[Union[Integer, Continuous]]) -> MomentType:
        """
        Calculate the expectation of the numeric variables in `variables`.

        :param variables: The variable to calculate the expectation of.
        :return: The expectation of the variable.
        """
        order = VariableMap({variable: 1 for variable in variables})
        center = VariableMap({variable: 0 for variable in variables})
        return self.moment(order, center)

    def variance(self, variables: Iterable[Union[Integer, Continuous]]) -> MomentType:
        """
        Calculate the variance of the numeric variables in `variables`.

        :param variables: The variable to calculate the variance of.
        :return: The variance of the variable.
        """
        order = VariableMap({variable: 2 for variable in variables})
        center = self.expectation(variables)
        return self.moment(order, center)


class ProbabilisticModelWrapper:
    """
    Wrapper class for probabilistic models.
    """

    model: ProbabilisticModel
    """The model that is wrapped."""

    def likelihood(self, event: Iterable) -> float:
        return self.model.likelihood(event)

    def _likelihood(self, event: Iterable) -> float:
        return self.model._likelihood(event)

    def probability(self, event: Event) -> float:
        return self.model.probability(event)

    def _probability(self, event: EncodedEvent) -> float:
        return self.model._probability(event)

    def mode(self) -> Tuple[List[Event], float]:
        return self.model.mode()

    def _mode(self) -> Tuple[Iterable[EncodedEvent], float]:
        return self.model._mode()

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        return self.model.marginal(variables)

    def conditional(self, event: Event) -> Tuple[Optional[Self], float]:
        return self.model.conditional(event)

    def _conditional(self, event: EncodedEvent) -> Tuple[Optional[Self], float]:
        return self.model._conditional(event)

    def sample(self, amount: int) -> Iterable:
        return self.model.sample(amount)

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        return self.model.moment(order, center)
