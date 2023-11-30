from typing import List, Iterable, Any

import numpy as np
import pandas as pd
from random_events.variables import Variable, Continuous as REContinuous, Integer as REInteger, Symbolic


def infer_variables_from_dataframe(data: pd.DataFrame, scale_continuous_types: bool = True) -> List[Variable]:
    """
    Infer the variables from a dataframe.
    The variables are inferred by the column names and types of the dataframe.

    :param data: The dataframe to infer the variables from.
    :param scale_continuous_types: Whether to scale numeric types.
    :return: The inferred variables.
    """
    result = []

    for column, datatype in zip(data.columns, data.dtypes):

        # handle continuous variables
        if datatype in [float]:

            minimal_distance_between_values = np.diff(np.sort(data[column].unique())).min()
            mean = data[column].mean()
            std = data[column].std()

            if scale_continuous_types:
                variable = ScaledContinuous(column, mean, std, minimal_distance_between_values)
            else:
                variable = Continuous(column, mean, std, minimal_distance_between_values)

        # handle discrete variables
        elif datatype in [object, int]:

            unique_values = data[column].unique()

            if datatype == int:
                mean = data[column].mean()
                std = data[column].std()
                variable = Integer(column, unique_values, mean, std)
            elif datatype == object:
                variable = Symbolic(column, unique_values)
            else:
                raise ValueError(f"Datatype {datatype} of column {column} is not supported.")

        else:
            raise ValueError(f"Datatype {datatype} of column {column} is not supported.")

        result.append(variable)

    return result


class Integer(REInteger):
    mean: float
    """
    Mean of the random variable.
    """

    std: float
    """
    Standard Deviation of the random variable.
    """

    def __init__(self, name: str, domain: Iterable, mean, std):
        super().__init__(name, domain)
        self.mean = mean
        self.std = std


class Continuous(REContinuous):
    """
    Base class for continuous variables in JPTs. This class does not standardize the data,
    but needs to know mean and std anyway.
    """

    mean: float
    """
    Mean of the random variable.
    """

    std: float
    """
    Standard Deviation of the random variable.
    """

    minimal_distance: float
    """
    The minimal distance between two values of the variable.
    """

    min_likelihood_improvement: float
    """
    The minimum likelihood improvement passed to the Nyga Distributions.
    """

    min_samples_per_quantile: int
    """
    The minimum number of samples per quantile passed to the Nyga Distributions.
    """

    def __init__(self, name: str, mean: float, std: float, minimal_distance: float = 1.,
                 min_likelihood_improvement: float = 0.1, min_samples_per_quantile: int = 10):
        super().__init__(name)
        self.mean = mean
        self.std = std
        self.minimal_distance = minimal_distance
        self.min_likelihood_improvement = min_likelihood_improvement
        self.min_samples_per_quantile = min_samples_per_quantile


class ScaledContinuous(Continuous):
    """
    A continuous variable that is standardized.
    """

    def __init__(self, name: str, mean: float, std: float, minimal_distance: float = 1.):
        super().__init__(name, mean, std, minimal_distance)

    def encode(self, value: Any):
        return (value - self.mean) / self.std

    def decode(self, value: float) -> float:
        return value * self.std + self.mean

    def __str__(self):
        return f"{self.__class__.__name__}({self.name}, {self.mean}, {self.std}, {self.minimal_distance})"
