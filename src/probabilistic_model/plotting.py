from abc import ABC

import numpy as np
import plotly.graph_objs as go
from random_events.interval import Interval, SimpleInterval
from random_events.product_algebra import Event
from random_events.variable import Continuous
from typing_extensions import Tuple, List, Optional

from .constants import *
from .probabilistic_model import ProbabilisticModel


class SampleBasedPlotMixin(ProbabilisticModel, ABC):
    """
    Mixin class for plotting models that contain only continuous variables using samples.
    """

    variables: Tuple[Continuous]

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the cumulative distribution function of the model.
        """
        raise NotImplementedError

    def composite_set_from_event(self, event: Event) -> Interval:
        """
        Convert an event to a composite set.
        """
        raise NotImplementedError

    def plot(self, number_of_samples: int = 1000) -> List:
        """
        Plot the model using samples.
        """

        if not all(isinstance(variable, Continuous) for variable in self.variables):
            raise ValueError("All variables must be continuous to support sample based plotting.")

        if len(self.variables) == 1:
            return self.plot_1d(number_of_samples)
        elif len(self.variables) == 2:
            return self.plot_2d(number_of_samples)
        elif len(self.variables) == 3:
            return self.plot_3d(number_of_samples)
        else:
            raise NotImplementedError("Cannot plot models with more than 3 variables")

    def plot_mode_1d(self, mode: Optional[Event], height: float) -> List:
        if mode is None:
            return []

        interval = self.composite_set_from_event(mode)
        x_values = []
        y_values = []
        for simple_interval in interval.simple_sets:
            simple_interval: SimpleInterval
            x_values += (
                [simple_interval.lower, simple_interval.lower, simple_interval.upper, simple_interval.upper, None])
            y_values += ([0, height, height, 0, None])
        return [go.Scatter(x=x_values, y=y_values, mode="lines+markers", name=MODE_TRACE_NAME, fill="toself",
                           line=dict(color=MODE_TRACE_COLOR))]

    def plot_1d(self, number_of_samples: int) -> List:
        samples = np.sort(self.sample(number_of_samples), axis=0)
        likelihood = self.likelihood(samples)
        samples = samples[:, 0]

        try:
            mode, maximum_likelihood = self.mode()
        except NotImplementedError:
            mode, maximum_likelihood = None, max(likelihood)

        height = maximum_likelihood * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT

        pdf_trace = go.Scatter(x=samples, y=likelihood, mode="lines", legendgroup="PDF", name=PDF_TRACE_NAME,
                               line=dict(color=PDF_TRACE_COLOR))

        try:
            cdf = self.cdf(samples)
            cdf_trace = go.Scatter(x=samples, y=cdf, mode="lines", name=CDF_TRACE_NAME, legendgroup="CDF",
                                   line=dict(color=CDF_TRACE_COLOR))
        except NotImplementedError:
            cdf_trace = None

        mode_traces = self.plot_mode_1d(mode, height)
        return ([pdf_trace, cdf_trace, self.expectation_trace_1d(height)] + mode_traces +
                self.complement_of_support_trace_1d(min(samples), max(samples)))

    def expectation_trace_1d(self, height: float) -> go.Scatter:
        mean = self.expectation(self.variables)[self.variables[0]]
        mean_trace = go.Scatter(x=[mean, mean], y=[0, height], mode="lines+markers", name=EXPECTATION_TRACE_NAME,
                                marker=dict(color=EXPECTATION_TRACE_COLOR), line=dict(color=EXPECTATION_TRACE_COLOR))
        return mean_trace

    def complement_of_support_trace_1d(self, min_of_samples: float, max_of_samples: float) -> List:
        """
        Create a trace for the complement of the support of the model in 1d.
        :param min_of_samples: The minimum value of the samples.
        :param max_of_samples: The maximum value of the samples.
        :return: A list of traces for the support of the model.
        """
        supporting_interval: Interval = self.support().simple_sets[0][self.variables[0]]
        complement_of_support = supporting_interval.complement()

        range_of_samples = max_of_samples - min_of_samples
        pdf_x_values = []
        pdf_y_values = []
        cdf_x_values = []
        cdf_y_values = []

        for simple_interval in complement_of_support.simple_sets:
            simple_interval: SimpleInterval

            # if it is the leftmost interval
            if simple_interval.lower == -np.inf:
                left_padded_value = simple_interval.upper - (PADDING_FACTOR_FOR_X_AXIS_IN_PLOT * range_of_samples)
                pdf_x_values += [left_padded_value, simple_interval.upper, None]
                pdf_y_values += [0, 0, None]
                cdf_x_values += [left_padded_value, simple_interval.upper, None]
                cdf_y_values += [0, 0, None]

            # if it is the rightmost interval
            elif simple_interval.upper == np.inf:
                right_padded_value = simple_interval.lower + (PADDING_FACTOR_FOR_X_AXIS_IN_PLOT * range_of_samples)
                pdf_x_values += [simple_interval.lower, right_padded_value, None]
                pdf_y_values += [0, 0, None]
                cdf_x_values += [simple_interval.lower, right_padded_value, None]
                cdf_y_values += [1, 1, None]

            # if it is an interval in the middle
            else:
                pdf_x_values += [simple_interval.lower, simple_interval.upper, None]
                pdf_y_values += [0, 0, None]

        return [go.Scatter(x=pdf_x_values, y=pdf_y_values, mode="lines", legendgroup="PDF", showlegend=False,
                           line=dict(color=PDF_TRACE_COLOR)),
                go.Scatter(x=cdf_x_values, y=cdf_y_values, mode="lines", legendgroup="CDF", showlegend=False,
                           line=dict(color=CDF_TRACE_COLOR))]

    def plot_2d(self, number_of_samples: int) -> List:
        raise NotImplementedError

    def plot_3d(self, number_of_samples: int) -> List:
        raise NotImplementedError
