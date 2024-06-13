from __future__ import annotations

import abc

from random_events.interval import closed
from random_events.product_algebra import *
from random_events.set import *
from random_events.variable import *

from .constants import *

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
        for simple_event in event.simple_sets:
            simple_event.fill_missing_variables(self.variables)
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

    def conditional(self, event: Event) -> Tuple[Optional[Union[ProbabilisticModel, Self]], float]:
        """
        Calculate the conditional distribution P(*| event) and the probability of the event.

        If the event is impossible, the conditional distribution is None and the probability is 0.

        :param event: The event to condition on.
        :return: The conditional distribution and the probability of the event.
        """
        for simple_event in event.simple_sets:
            simple_event.fill_missing_variables(self.variables)
        conditional, log_probability = self.log_conditional(event)
        return conditional, np.exp(log_probability)

    @abstractmethod
    def log_conditional(self, event: Event) -> Tuple[Optional[Union[ProbabilisticModel, Self]], float]:
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

    def expectation(self, variables: Iterable[Variable]) -> MomentType:
        """
        Calculate the expectation of the numeric variables in `variables`.

        :param variables: The variable to calculate the expectation of.
        :return: The expectation of the variable.
        """
        order = VariableMap({variable: 1 for variable in variables})
        center = VariableMap({variable: 0 for variable in variables})
        return self.moment(order, center)

    def variance(self, variables: Iterable[Variable]) -> MomentType:
        """
        Calculate the variance of the numeric variables in `variables`.

        :param variables: The variable to calculate the variance of.
        :return: The variance of the variable.
        """
        order = VariableMap({variable: 2 for variable in variables})
        center = self.expectation(variables)
        return self.moment(order, center)

    def universal_simple_event(self) -> SimpleEvent:
        """
        :return: A simple event that contains every possible value.
        """
        return SimpleEvent({variable: variable.domain for variable in self.variables})

    def plotly_layout(self) -> Dict[str, Any]:
        """
        Create a layout for the plotly plot.

        :return: The layout.
        """
        if len(self.variables) == 1:
            return self.plotly_layout_1d()
        elif len(self.variables) == 2:
            return self.plotly_layout_2d()
        elif len(self.variables) == 3:
            return self.plotly_layout_3d()
        else:
            raise NotImplementedError("Plotting is only supported for models with up to three variables.")

    def plotly_layout_1d(self) -> Dict[str, Any]:
        """
        :return: The layout argument for plotly figures as dict
        """
        return {"title": f"{self.representation}", "xaxis": {"title": self.variables[0].name}}

    def plotly_layout_2d(self) -> Dict[str, Any]:
        """
        :return: The layout argument for plotly figures as dict
        """
        return {"title": f"{self.representation}", "xaxis": {"title": self.variables[0].name},
                "yaxis": {"title": self.variables[1].name}}

    def plotly_layout_3d(self) -> Dict[str, Any]:
        """
        :return: The layout argument for plotly figures as dict
        """
        return {"title": f"{self.representation}",
                "scene": {"xaxis": {"title": self.variables[0].name}, "yaxis": {"title": self.variables[1].name},
                          "zaxis": {"title": self.variables[2].name}}}

    def plot(self, number_of_samples: int = 1000) -> List:
        """
        Generate traces that can be plotted with plotly.
        :return: The traces.
        """
        if len(self.variables) == 1:
            return self.plot_1d(number_of_samples)
        elif len(self.variables) == 2:
            return self.plot_2d(number_of_samples)
        elif len(self.variables) == 3:
            return self.plot_3d(number_of_samples)
        else:
            raise NotImplementedError("Plotting is only supported for models with up to three variables.")

    def plot_1d(self, number_of_samples: int) -> List:
        """
        Plot a one-dimensional model using samples.
        :param number_of_samples: The number of samples to draw.
        :return: The traces.
        """
        samples = np.sort(self.sample(number_of_samples), axis=0)
        likelihood = self.likelihood(samples)

        try:
            mode, maximum_likelihood = self.mode()
        except NotImplementedError:
            mode, maximum_likelihood = None, max(likelihood)

        height = maximum_likelihood * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT

        pdf_trace = go.Scatter(x=samples[:, 0], y=likelihood, mode="lines", legendgroup="PDF", name=PDF_TRACE_NAME,
                               line=dict(color=PDF_TRACE_COLOR))

        mode_traces = self.univariate_mode_traces(mode, height)
        return ([pdf_trace, self.univariate_expectation_trace(height)] + mode_traces +
                self.univariate_complement_of_support_trace(min(samples)[0], max(samples)[0]))

    def univariate_expectation_trace(self, height: float) -> go.Scatter:
        """
        Create a trace for the expectation of the model in 1d.
        :param height: The height of the trace.
        :return: The trace.
        """
        mean = self.expectation(self.variables)[self.variables[0]]
        mean_trace = go.Scatter(x=[mean, mean], y=[0, height], mode="lines+markers", name=EXPECTATION_TRACE_NAME,
                                marker=dict(color=EXPECTATION_TRACE_COLOR), line=dict(color=EXPECTATION_TRACE_COLOR))
        return mean_trace

    def univariate_mode_traces(self, mode: Optional[Event], height: float):
        if mode is None:
            return []

        interval = mode.simple_sets[0][self.variables[0]]
        x_values = []
        y_values = []
        for simple_interval in interval.simple_sets:
            simple_interval: SimpleInterval
            x_values += (
                [simple_interval.lower, simple_interval.lower, simple_interval.upper, simple_interval.upper, None])
            y_values += ([0, height, height, 0, None])
        return [go.Scatter(x=x_values, y=y_values, mode="lines+markers", name=MODE_TRACE_NAME, fill="toself",
                           line=dict(color=MODE_TRACE_COLOR))]

    def univariate_complement_of_support_trace(self, min_of_samples: float, max_of_samples: float) -> List:
        """
        Create a trace for the complement of the support of the model in 1d.
        :param min_of_samples: The minimum value of the samples.
        :param max_of_samples: The maximum value of the samples.
        :return: A list of traces for the support of the model.
        """
        supporting_interval: Interval = self.support().simple_sets[0][self.variables[0]]
        complement_of_support = supporting_interval.complement()
        limiting_interval = closed(min_of_samples - min_of_samples * PADDING_FACTOR_FOR_X_AXIS_IN_PLOT,
                                   max_of_samples + max_of_samples * PADDING_FACTOR_FOR_X_AXIS_IN_PLOT)
        limited_complement_of_support = complement_of_support & limiting_interval
        traces = SimpleEvent({self.variables[0]: limited_complement_of_support}).plot()
        for trace in traces:
            trace.update(name=PDF_TRACE_NAME, marker=dict(color=PDF_TRACE_COLOR))
        return traces

    def plot_2d(self, number_of_samples: int) -> List:
        """
        Plot a two-dimensional model.
        :param number_of_samples: The number of samples to draw.
        :return: The traces.
        """
        samples = self.sample(number_of_samples)
        likelihood = self.likelihood(samples)
        expectation = self.expectation(self.variables)
        likelihood_trace = go.Scatter(x=samples[:, 0], y=samples[:, 1], mode="markers", marker=dict(color=likelihood))
        expectation_trace = go.Scatter(x=[expectation[self.variables[0]]], y=[expectation[self.variables[1]]],
                                       mode="markers", marker=dict(color=EXPECTATION_TRACE_COLOR))
        return [likelihood_trace, expectation_trace] + self.multivariate_mode_traces()

    def plot_3d(self, number_of_samples: int) -> List:
        """
        Plot a three-dimensional model using samples.
        :param number_of_samples: The number of samples to draw.
        :return: The traces.s
        """
        samples = self.sample(number_of_samples)
        likelihood = self.likelihood(samples)
        expectation = self.expectation(self.variables)
        likelihood_trace = go.Scatter3d(x=samples[:, 0], y=samples[:, 1], z=samples[:, 2], mode="markers",
                                        marker=dict(color=likelihood))
        expectation_trace = go.Scatter3d(x=[expectation[self.variables[0]]], y=[expectation[self.variables[1]]],
                                         z=[expectation[self.variables[2]]], mode="markers",
                                         name=EXPECTATION_TRACE_NAME, marker=dict(color=EXPECTATION_TRACE_COLOR))

        return [likelihood_trace, expectation_trace] + self.multivariate_mode_traces()

    def multivariate_mode_traces(self):
        """
        :return: traces for the mode of a multivariate model.
        """
        try:
            mode, _ = self.mode()
            mode_traces = mode.plot(color=MODE_TRACE_COLOR)
            for trace in mode_traces:
                trace.update(name=MODE_TRACE_NAME)
        except NotImplementedError:
            mode_traces = []
        return mode_traces
