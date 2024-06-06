from abc import abstractmethod, ABC

import numpy as np
from random_events.interval import Interval, SimpleInterval
from random_events.product_algebra import Event
from random_events.variable import Continuous
from typing_extensions import Tuple, List, Optional

from .probabilistic_model import ProbabilisticModel
import plotly.graph_objs as go
from .constants import *


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

    def plot_mode_1d(self, mode: Optional[Event], maximum_likelihood: float) -> List:
        if mode is None:
            return []

        interval = self.composite_set_from_event(mode)
        x_values = []
        y_values = []
        for simple_interval in interval.simple_sets:
            simple_interval: SimpleInterval
            x_values += ([simple_interval.lower, simple_interval.lower, simple_interval.upper, simple_interval.upper, None])
            y_values += ([0, maximum_likelihood, maximum_likelihood, 0, None])
        return [go.Scatter(x=x_values, y=y_values, mode="lines", name="Mode", fill="toself")]

    def plot_1d(self, number_of_samples: int) -> List:
        samples = np.sort(self.sample(number_of_samples), axis=0)
        likelihood = self.likelihood(samples)
        samples = samples[:, 0]
        cdf = self.cdf(samples)
        mean = self.expectation(self.variables)[self.variables[0]]

        try:
            mode, maximum_likelihood = self.mode()
        except NotImplementedError:
            mode, maximum_likelihood = None, max(likelihood)

        height = maximum_likelihood * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT

        pdf_trace = go.Scatter(x=samples, y=likelihood, mode="lines", name="PDF")
        cdf_trace = go.Scatter(x=samples, y=cdf, mode="lines", name="CDF")
        mean_trace = go.Scatter(x=[mean, mean], y=[0, height], mode="lines+markers", name="Expectation")
        mode_traces = self.plot_mode_1d(mode, height)
        return [pdf_trace, cdf_trace, mean_trace] + mode_traces

    def plot_2d(self, number_of_samples: int) -> List:
        raise NotImplementedError

    def plot_3d(self, number_of_samples: int) -> List:
        raise NotImplementedError
