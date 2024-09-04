from __future__ import annotations

import time
import types
from collections import defaultdict
from functools import wraps

import numpy as np
from random_events.interval import SimpleInterval, Interval
from random_events.utils import recursive_subclasses
from typing_extensions import Type


def simple_interval_as_array(interval: SimpleInterval) -> np.ndarray:
    """
    Convert a simple interval to a numpy array.
    :param interval:  The interval
    :return:  [lower, upper] as numpy array
    """
    return np.array([interval.lower, interval.upper])


def interval_as_array(interval: Interval) -> np.ndarray:
    """
    Convert an interval to a numpy array.
    The resulting array has shape (n, 2) where n is the number of simple intervals in the interval.
    The first column contains the lower bounds and the second column the upper bounds of the simple intervals.
    :param interval: The interval
    :return:  as numpy array
    """
    return np.array([simple_interval_as_array(simple_interval) for simple_interval in interval.simple_sets])


def type_converter(abstract_type: Type, package: types.ModuleType):
    """
    Convert a type to a different type from a target sub-package that inherits from this type.

    :param abstract_type: The type to convert
    :param package: The sub-package to search in for that type

    :return: The converted type
    """
    for subclass in recursive_subclasses(abstract_type):
        if subclass.__module__.startswith(package.__name__):
            return subclass

    raise ValueError("Could not find type {} in package {}".format(abstract_type, package))


class MissingDict(defaultdict):
    """
    A defaultdict that returns the default value when the key is missing and does **not** add the key to the dict.
    """

    def __missing__(self, key):
        return self.default_factory()


def timeit(func):
    """
    Decorator to measure the time a function takes to execute.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{func.__name__} took {total_time} s')
        return result

    return timeit_wrapper
