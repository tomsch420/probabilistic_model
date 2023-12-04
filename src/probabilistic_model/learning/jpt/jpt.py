import math
from collections import deque
from typing import Tuple, Union, Optional, List, Iterable, Dict, Any
from typing_extensions import Self

import numpy as np
import pandas as pd
import portion

from random_events.events import VariableMap, Event, EncodedEvent
from random_events.variables import Variable

from .variables import Continuous, Integer, Symbolic
from ..nyga_distribution import NygaDistribution
from ...probabilistic_circuit.distributions import DiracDeltaDistribution, SymbolicDistribution, IntegerDistribution
from ...probabilistic_circuit.units import DeterministicSumUnit, DecomposableProductUnit, Unit
from jpt.learning.impurity import Impurity
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class JPT(DeterministicSumUnit):

    targets: Tuple[Variable]
    """
    The variables to optimize for.
    """

    features: Tuple[Variable]
    """
    The variables that are used to craft criteria.
    """

    _min_samples_leaf: Union[int, float]
    """
    The minimum number of samples to create another sum node. If this is smaller than one, it will be reinterpreted
    as fraction w. r. t. the number of samples total.
    """

    min_impurity_improvement: float
    """
    The minimum impurity improvement to create another sum node.
    """

    max_leaves: Union[int, float]
    """
    The maximum number of leaves.
    """

    max_depth: Union[int, float]
    """
    The maximum depth of the tree.
    """

    dependencies: VariableMap
    """
    The dependencies between the variables.
    """

    total_samples: int = 1
    """
    The total amount of samples that were used to fit the model.
    """

    indices: Optional[np.ndarray] = None
    impurity: Optional[Impurity] = None
    c45queue: deque = deque()
    weights: List[float]

    def __init__(self, variables: Iterable[Variable], targets: Optional[Iterable[Variable]] = None,
                 features: Optional[Iterable[Variable]] = None, min_samples_leaf: Union[int, float] = 1,
                 min_impurity_improvement: float = 0.0, max_leaves: Union[int, float] = float("inf"),
                 max_depth: Union[int, float] = float("inf"), dependencies: Optional[VariableMap] = None, ):

        super().__init__(variables, weights=[])
        self.set_targets_and_features(targets, features)
        self._min_samples_leaf = min_samples_leaf
        self.min_impurity_improvement = min_impurity_improvement
        self.max_leaves = max_leaves
        self.max_depth = max_depth

        if dependencies is None:
            self.dependencies = VariableMap({var: list(self.targets) for var in self.features})
        else:
            self.dependencies = dependencies

    def set_targets_and_features(self, targets: Optional[Iterable[Variable]],
                                 features: Optional[Iterable[Variable]]) -> None:
        """
        Set the targets and features of the model.
        If only one of them is provided, the other is set as the complement of the provided one.
        If none are provided, both of them are set as the variables of the model.
        If both are provided, they are taken as given.

        :param targets: The targets of the model.
        :param features: The features of the model.
        :return: None
        """
        # if targets are not specified
        if targets is None:

            # and features are not specified
            if features is None:
                self.targets = self.variables
                self.features = self.variables

            # and features are specified
            else:
                self.targets = tuple(sorted(set(self.variables) - set(features)))
                self.features = tuple(sorted(features))

        # if targets are specified
        else:
            # and features are not specified
            if features is None:
                self.targets = tuple(sorted(set(targets)))
                self.features = tuple(sorted(set(self.variables) - set(targets)))

            # and features are specified
            else:
                self.targets = tuple(sorted(set(targets)))
                self.features = tuple(sorted(set(features)))

    def __eq__(self, other):
        return (isinstance(other, JPT) and
                self.variables == other.variables and
                self.targets == other.targets and
                self.features == other.features and
                self.children == other.children and
                self.min_impurity_improvement == other.min_impurity_improvement and
                self.max_depth == other.max_depth)

    @property
    def min_samples_leaf(self):
        """
        The minimum number of samples to create another sum node.
        """
        if self._min_samples_leaf < 1.:
            return math.ceil(self._min_samples_leaf * self.total_samples)
        else:
            return self._min_samples_leaf

    @property
    def numeric_variables(self):
        return [variable for variable in self.variables if isinstance(variable, (Continuous, Integer))]

    @property
    def numeric_targets(self):
        return [variable for variable in self.targets if isinstance(variable, (Continuous, Integer))]

    @property
    def numeric_features(self):
        return [variable for variable in self.features if isinstance(variable, (Continuous, Integer))]

    @property
    def symbolic_variables(self):
        return [variable for variable in self.variables if isinstance(variable, Symbolic)]

    @property
    def symbolic_targets(self):
        return [variable for variable in self.targets if isinstance(variable, Symbolic)]

    @property
    def symbolic_features(self):
        return [variable for variable in self.features if isinstance(variable, Symbolic)]

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the data to be used in the model.

        :param data: The data to preprocess.
        :return: The preprocessed data.
        """

        result = np.zeros(data.shape)

        for variable_index, variable in enumerate(self.variables):
            column = data[variable.name]
            column = variable.encode_many(column)
            result[:, variable_index] = column

        return result

    def fit(self, data: pd.DataFrame) -> 'JPT':
        """
        Fit the model to the data.

        :param data: The data to fit the model to.
        :return: The fitted model.
        """

        preprocessed_data = self.preprocess_data(data)

        self.total_samples = len(preprocessed_data)

        self.indices = np.ascontiguousarray(np.arange(preprocessed_data.shape[0], dtype=np.int64))
        self.impurity = self.construct_impurity()
        self.impurity.setup(preprocessed_data, self.indices)

        self.c45queue.append((preprocessed_data, 0, len(preprocessed_data), 0))

        while self.c45queue:
            self.c45(*self.c45queue.popleft())

        return self

    def c45(self, data: np.ndarray, start: int, end: int, depth: int):
        """
        Construct a DecisionNode or DecomposableProductNode from the data.

        :param data: The data to calculate the impurity from.
        :param start: Starting index in the data.
        :param end: Ending index in the data.
        :param depth: The current depth of the induction
        :return: The constructed decision tree node
        """

        number_of_samples = end - start

        # if the inducing in this step results in inadmissible nodes, skip the impurity calculation
        if depth >= self.max_depth or number_of_samples < 2 * self.min_samples_leaf:
            max_gain = -float("inf")
        else:
            max_gain = self.impurity.compute_best_split(start, end)

        # if the max gain is insufficient
        if max_gain <= self.min_impurity_improvement:

            # create decomposable product node
            leaf_node = self.create_leaf_node(data[self.indices[start:end]])
            self.weights.append(number_of_samples/len(data))
            leaf_node.parent = self

            # terminate the induction
            return

        # if the max gain is sufficient
        split_pos = self.impurity.best_split_pos

        # increase the depth
        new_depth = depth + 1

        # append the new induction steps
        self.c45queue.append((data, start, start + split_pos + 1, new_depth))
        self.c45queue.append((data, start + split_pos + 1, end, new_depth))

    def get_split_value(self, data: np.ndarray, start: int) -> Tuple[Event, Event]:
        """
        Get the split value from the impurity and data.
        Return events describing the split.

        :param data: The data to calculate the split value from.
        :param start: The starting index in the data.

        :return: The splitting events left and right.
        """
        split_position = self.impurity.best_split_pos
        split_variable_index = self.impurity.best_var
        split_variable = self.variables[split_variable_index]

        if isinstance(split_variable, Continuous):
            split_value = (data[self.indices[start + split_position], split_variable_index] +
                           data[self.indices[start + split_position + 1], split_variable_index]) / 2
            left_event = Event({split_variable: portion.closedopen(-np.inf, split_value)})
            right_event = Event({split_variable: portion.closed(split_value, np.inf)})
        else:
            split_value = int(data[self.indices[start + split_position], split_variable_index])
            left_event = EncodedEvent({split_variable: split_value}).decode()
            right_event = Event() - left_event
        return left_event, right_event

    def create_leaf_node(self, data: np.ndarray) -> DecomposableProductUnit:
        result = DecomposableProductUnit(self.variables)

        for index, variable in enumerate(self.variables):
            if isinstance(variable, Continuous):
                distribution = NygaDistribution(variable,
                                                min_likelihood_improvement=variable.min_likelihood_improvement,
                                                min_samples_per_quantile=variable.min_samples_per_quantile)
                distribution._fit(data[:, index].tolist())

                if isinstance(distribution.children[0], DiracDeltaDistribution):
                    distribution.children[0].density_cap = 1/variable.minimal_distance

            elif isinstance(variable, Symbolic):
                distribution = SymbolicDistribution(variable, weights=[1/len(variable.domain)]*len(variable.domain))
                distribution._fit(data[:, index].tolist())

            elif isinstance(variable, Integer):
                distribution = IntegerDistribution(variable, weights=[1/len(variable.domain)]*len(variable.domain))
                distribution._fit(data[:, index].tolist())
            else:
                raise ValueError(f"Variable {variable} is not supported.")

            distribution.parent = result

        return result

    def construct_impurity(self) -> Impurity:
        min_samples_leaf = self.min_samples_leaf

        numeric_vars = (
            np.array([index for index, variable in enumerate(self.variables) if variable in self.numeric_targets], dtype=int))
        symbolic_vars = np.array(
            [index for index, variable in enumerate(self.variables) if variable in self.symbolic_targets], dtype=int)

        invert_impurity = np.array([0] * len(self.symbolic_targets), dtype=int)

        n_sym_vars_total = len(self.symbolic_variables)
        n_num_vars_total = len(self.numeric_variables)

        numeric_features = np.array(
            [index for index, variable in enumerate(self.variables) if variable in self.numeric_features], dtype=int)
        symbolic_features = np.array(
            [index for index, variable in enumerate(self.variables) if variable in self.symbolic_features], dtype=int)

        symbols = np.array([len(variable.domain) for variable in self.symbolic_variables])
        max_variances = np.array([variable.std ** 2 for variable in self.numeric_variables])

        dependency_indices = dict()

        for variable, dep_vars in self.dependencies.items():
            # get the index version of the dependent variables and store them
            idx_var = self.variables.index(variable)
            idc_dep = [self.variables.index(var) for var in dep_vars]
            dependency_indices[idx_var] = idc_dep

        return Impurity(min_samples_leaf, numeric_vars, symbolic_vars, invert_impurity, n_sym_vars_total,
                        n_num_vars_total, numeric_features, symbolic_features, symbols, max_variances,
                        dependency_indices)

    def plot(self) -> go.Figure:
        """
        Plot the model.
        """
        subplot_titles = [distribution.__class__.__name__ for child in self.children for distribution in child.children]
        figure = make_subplots(rows=len(self.children), cols=len(self.variables),
                               row_titles=[f"P(Leaf = {child_index}) = {weight}" for weight, child_index
                                           in zip(self.weights, range(len(self.children)))],
                               subplot_titles=subplot_titles)

        for child_index, child in enumerate(self.children):
            child: DecomposableProductUnit

            for distribution_index, distribution in enumerate(child.children):
                traces: List[go.Scatter] = distribution.plot()
                legend_group = child_index * len(self.variables) + distribution_index + 1
                traces = [trace.update(legendgroup=legend_group)
                          for trace in traces]
                figure.add_traces(traces, rows=child_index + 1, cols=distribution_index + 1)
                figure.update_xaxes(title_text=distribution.variable.name,
                                    row=child_index + 1,
                                    col=distribution_index + 1)

        figure.update_layout(height=300*len(self.children), width=600*len(self.variables),
                             title=f"Joint Probability Tree over {len(self.variables)} variables",)

        return figure

    def _variable_dependencies_to_json(self) -> Dict[str, List[str]]:
        """
        Convert the variable dependencies to a json compatible format.
        The result maps variable names to lists of variable names.
        """
        return {variable.name: [dependency.name for dependency in dependencies]
                for variable, dependencies in self.dependencies.items()}

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["targets"] = [variable.name for variable in self.targets]
        result["features"] = [variable.name for variable in self.features]
        result["_min_samples_leaf"] = self._min_samples_leaf
        result["min_impurity_improvement"] = self.min_impurity_improvement
        result["max_leaves"] = self.max_leaves
        result["max_depth"] = self.max_depth
        result["dependencies"] = self._variable_dependencies_to_json()
        result["total_samples"] = self.total_samples
        return result

    @classmethod
    def from_json_with_variables_and_children(cls, data: Dict[str, Any],
                                              variables: List[Variable],
                                              children: List[Unit]) -> Self:
        targets = [variable for variable in variables if variable.name in data["targets"]]
        features = [variable for variable in variables if variable.name in data["features"]]
        dependencies = VariableMap()
        for variable in variables:
            dependencies[variable] = [dep_var for dep_var in variables if dep_var.name
                                      in data["dependencies"][variable.name]]
        result = cls(variables=variables, targets=targets, features=features,
                     min_samples_leaf=data["_min_samples_leaf"],
                     min_impurity_improvement=data["min_impurity_improvement"], max_leaves=data["max_leaves"],
                     max_depth=data["max_depth"], dependencies=dependencies)
        result.total_samples = data["total_samples"]
        result.children = children
        result.weights = data["weights"]
        return result
