import abc

import numpy as np
from typing_extensions import Tuple, Iterable, List, Optional, Union, TYPE_CHECKING, Self

from random_events.sigma_algebra import *
from random_events.product_algebra import *
from random_events.variable import *
from random_events.set import *

# Type definitions
FullEvidenceType = np.array  # [Union[float, int, SetElement]]

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

    """

    @property
    def representation(self) -> str:
        """
        The symbol used to represent this distribution.
        """
        return self.__class__.__name__

    @property
    @abc.abstractmethod
    def variables(self) -> Tuple[Variable, ...]:
        """
        :return: The variables of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def support(self) -> Event:
        """
        :return: The support of the model.
        """
        raise NotImplementedError

    def likelihood(self, events: np.array) -> np.array:
        """
        Calculate the likelihood of an array of events.

        The likelihood is a full evidence query, i.e., an assignment to all variables in the model.
        The order of elements in the event has to correspond to the order of variables in the model.

        The event belongs to the class of full evidence queries.

        .. Note:: You can read more about queries of this class in Definition 1 in :cite:p:`choi2020probabilistic`
            or watch the `video tutorial <https://youtu.be/2RAG5-L9R70?si=TAfIX2LmOWM-Fd2B&t=785>`_.
            :cite:p:`youtube2020probabilistic`

        :param events: The array of full evidence event. The shape of the array has to be (n, len(self.variables)).
        :return: The likelihoods of the event as an array of shape (n, ).
        """
        return np.exp(self.log_likelihood(events))

    @abstractmethod
    def log_likelihood(self, events: np.array) -> np.array:
        """
        Calculate the log-likelihood of an event.

        Check the documentation of `likelihood` for more information.

        :param events: The full evidence event
        :return: The log-likelihood of the event.
        """
        raise NotImplementedError

    def probability(self, event: Event) -> float:
        """
        Calculate the probability of an event.
        The event is richly described by the random_events package.

        :param event: The event.
        :return: The probability of the event.
        """
        return sum(self.probability_of_simple_event(simple_set) for simple_set in event.simple_sets)

    @abstractmethod
    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        """
        Calculate the probability of a simple event.

        The event belongs to the class of marginal queries.

        .. Note:: You can read more about queries of this class in Definition 11 in :cite:p:`choi2020probabilistic`
            or watch the `video tutorial <https://youtu.be/2RAG5-L9R70?si=8aEGIqmoDTiUR2u6&t=1089>`_.
            :cite:p:`youtube2020probabilistic`

        :param event: The event.
        :return: The probability of the event.
        """
        raise NotImplementedError

    def mode(self) -> Tuple[Event, float]:
        """
        Calculate the mode of the model.
        The mode is the **set** of most likely events.

        The calculation belongs to the map query class.

        .. Note:: You can read more about queries of this class in Definition 26 in :cite:p:`choi2020probabilistic`
            or watch the `video tutorial <https://youtu.be/2RAG5-L9R70?si=FjREKNtAV0owm27A&t=1962>`_.
            :cite:p:`youtube2020probabilistic`

        :return: The mode and its likelihood.
        """
        mode, log_likelihood = self.log_mode()
        return mode, np.exp(log_likelihood)

    @abstractmethod
    def log_mode(self) -> Tuple[Event, float]:
        """
        Calculate the mode of the model.

        Check the documentation of `mode` for more information.

        :return: The mode and its log-likelihood.
        """
        raise NotImplementedError

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        """
        Calculate the marginal distribution of a set of variables.

        :param variables: The variables to calculate the marginal distribution on.
        :return: The marginal distribution over the variables.
        """
        raise NotImplementedError

    def conditional(self, event: Event) -> Tuple[Optional[Self], float]:
        """
        Calculate the conditional distribution P(*| event) and the probability of the event.

        If the event is impossible, the conditional distribution is None and the probability is 0.

        :param event: The event to condition on.
        :return: The conditional distribution and the probability of the event.
        """
        conditional, log_probability = self.log_conditional(event)
        return conditional, np.exp(log_probability)

    @abstractmethod
    def log_conditional(self, event: Event) -> Tuple[Optional[Self], float]:
        """
        Calculate the conditional distribution P(*| event) and the probability of the event.

        Check the documentation of `conditional` for more information.

        :param event: The event to condition on.
        :return: The conditional distribution and the log-probability of the event.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, amount: int) -> np.array:
        """
        Sample from the model.

        :param amount: The number of samples to draw.
        :return: The samples.
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
