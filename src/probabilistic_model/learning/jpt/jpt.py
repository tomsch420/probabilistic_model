import itertools
import math
from collections import deque
from datetime import datetime
from typing import Tuple, Union, Optional, List, Iterable, Dict, Any
from sortedcontainers import SortedSet
import random_events.variables
import tqdm
import networkx as nx
from matplotlib import pyplot as plt
from typing_extensions import Self

import numpy as np
import pandas as pd
import portion

from random_events.events import VariableMap, Event, EncodedEvent
from random_events.variables import Variable, Discrete

from .variables import Continuous, Integer, Symbolic
from ..nyga_distribution import NygaDistribution
from ...probabilistic_circuit.distributions.distributions import (DiracDeltaDistribution,
                                                                  SymbolicDistribution,
                                                                  IntegerDistribution)
from ...probabilistic_circuit.probabilistic_circuit import (DeterministicSumUnit,
                                                            DecomposableProductUnit as PMDecomposableProductUnit)
from jpt.learning.impurity import Impurity
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class DecomposableProductUnit(PMDecomposableProductUnit):
    sample_indices: List[int]
    """
    The indices of the samples of the training dataset that are used to fit a product unit.
    """

    total_samples: int
    """
    The number of samples that are used to form this product unit.
    """

    # def area_validation_metric(self, other: Self) -> float:
    #     avm = 1.
    #     for own_subcircuit in self.subcircuits:
    #         for other_subcircuit in other.subcircuits:
    #             if other_subcircuit.variables == own_subcircuit.variables:
    #                 # 0 means completely different, 1 means identical
    #                 avm *= own_subcircuit.area_validation_metric(other_subcircuit)
    #     return avm

    def events_of_higher_density(self, other: Self, own_node_weights, other_node_weights):
        own_result = Event()
        other_result = Event()
        combination_map = {}

        for own_univariate_unit in self.subcircuits:
            other_univariate_unit = [s for s in other.subcircuits if s.variables == own_univariate_unit.variables][0]
            variable = own_univariate_unit.variables[0]
            if isinstance(variable, random_events.variables.Continuous):
                own_univariate_unit: NygaDistribution
                other_univariate_unit: NygaDistribution
                all_mixture_points = own_univariate_unit.all_union_of_mixture_points_with(other_univariate_unit)
                combination_map[variable] = all_mixture_points
            elif isinstance(variable, random_events.variables.Symbolic):
                own_univariate_unit: SymbolicDistribution
                other_univariate_unit: SymbolicDistribution
                support = own_univariate_unit.domain | other_univariate_unit.domain
                combination_map[variable] = support
            else:
                raise NotImplementedError("Unknown Node Type")

        # number_of_combinations = math.prod([len(value) for value in combination_map.values()])
        own_weight = sum(own_node_weights.get(hash(self)))
        other_weight = sum(other_node_weights.get(hash(other)))
        for combination in itertools.product(*combination_map.values()): #tqdm.tqdm(itertools.product(*combination_map.values()), total=number_of_combinations):
            full_evidence_state = list(((element.upper - element.lower) / 2) + element.lower  if isinstance(element, portion.Interval) else element
                                   for element in combination)
            likelihood_in_self = self.likelihood(full_evidence_state) * own_weight
            likelihood_in_other = other.likelihood(full_evidence_state) * other_weight

            if likelihood_in_self > likelihood_in_other:
                if not own_result:
                    own_result = Event({variable: value for variable, value in zip(self.variables, combination)})
                else:
                    own_result = own_result.union(Event({variable: value for variable, value in zip(self.variables, combination)}))
            elif likelihood_in_other > likelihood_in_self:
                if not other_result:
                    other_result = Event({variable: value for variable, value in zip(other.variables, combination)})
                else:
                    other_result = other_result.union(Event({variable: value for variable, value in zip(other.variables, combination)}))

        return own_result, other_result


class JPT(DeterministicSumUnit):
    targets: Tuple[Variable, ...]
    """
    The variables to optimize for.
    """

    features: Tuple[Variable, ...]
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

    keep_sample_indices: bool = False
    """
    Rather to store the sample indices in the leaves or not.
    """

    variables_from_init: Tuple[Variable, ...]
    """
    The variables from initialization. Since variables will be overwritten as soon as the model is learned,
    we need to store the variables from initialization here.
    """

    def __init__(self, variables: Iterable[Variable], targets: Optional[Iterable[Variable]] = None,
                 features: Optional[Iterable[Variable]] = None, min_samples_leaf: Union[int, float] = 1,
                 min_impurity_improvement: float = 0.0, max_leaves: Union[int, float] = float("inf"),
                 max_depth: Union[int, float] = float("inf"), dependencies: Optional[VariableMap] = None, ):
        super().__init__()
        self.variables_from_init = tuple(sorted(variables))
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
                self.targets = self.variables_from_init
                self.features = self.variables_from_init

            # and features are specified
            else:
                self.targets = tuple(sorted(set(self.variables_from_init) - set(features)))
                self.features = tuple(sorted(features))

        # if targets are specified
        else:
            # and features are not specified
            if features is None:
                self.targets = tuple(sorted(set(targets)))
                self.features = tuple(sorted(set(self.variables_from_init) - set(targets)))

            # and features are specified
            else:
                self.targets = tuple(sorted(set(targets)))
                self.features = tuple(sorted(set(features)))

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
        return [variable for variable in self.variables_from_init if isinstance(variable, (Continuous, Integer))]

    @property
    def numeric_targets(self):
        return [variable for variable in self.targets if isinstance(variable, (Continuous, Integer))]

    @property
    def numeric_features(self):
        return [variable for variable in self.features if isinstance(variable, (Continuous, Integer))]

    @property
    def symbolic_variables(self):
        return [variable for variable in self.variables_from_init if isinstance(variable, Symbolic)]

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

        for variable_index, variable in enumerate(self.variables_from_init):
            column = data[variable.name]
            if not isinstance(variable, Integer):
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
            self.mount(leaf_node)
            weight = number_of_samples / len(data)
            self.probabilistic_circuit.add_edge(self, leaf_node, weight=weight)

            if self.keep_sample_indices:
                leaf_node.sample_indices = self.indices[start:end]

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
        """
        Create a fully decomposable product node from a 2D data array.

        :param data: The preprocessed data to use for training
        :return: The leaf node.
        """
        result = DecomposableProductUnit()
        result.total_samples = len(data)

        for index, variable in enumerate(self.variables_from_init):
            if isinstance(variable, Continuous):
                distribution = NygaDistribution(variable,
                                                min_likelihood_improvement=variable.min_likelihood_improvement,
                                                min_samples_per_quantile=variable.min_samples_per_quantile)
                distribution._fit(data[:, index].tolist())

                if isinstance(distribution.subcircuits[0], DiracDeltaDistribution):
                    distribution.subcircuits[0].density_cap = 1 / variable.minimal_distance

            elif isinstance(variable, Symbolic):
                distribution = SymbolicDistribution(variable, weights=[1 / len(variable.domain)] * len(variable.domain))
                distribution._fit(data[:, index].tolist())

            elif isinstance(variable, Integer):
                distribution = IntegerDistribution(variable, weights=[1 / len(variable.domain)] * len(variable.domain))
                distribution.fit(data[:, index].tolist())
            else:
                raise ValueError(f"Variable {variable} is not supported.")

            result.mount(distribution)
            result.probabilistic_circuit.add_edge(result, distribution)

        return result

    def construct_impurity(self) -> Impurity:
        min_samples_leaf = self.min_samples_leaf

        numeric_vars = (
            np.array([index for index, variable in enumerate(self.variables_from_init)
                      if variable in self.numeric_targets], dtype=int))
        symbolic_vars = np.array(
            [index for index, variable in enumerate(self.variables_from_init)
             if variable in self.symbolic_targets], dtype=int)

        invert_impurity = np.array([0] * len(self.symbolic_targets), dtype=int)

        n_sym_vars_total = len(self.symbolic_variables)
        n_num_vars_total = len(self.numeric_variables)

        numeric_features = np.array(
            [index for index, variable in enumerate(self.variables_from_init)
             if variable in self.numeric_features], dtype=int)
        symbolic_features = np.array(
            [index for index, variable in enumerate(self.variables_from_init)
             if variable in self.symbolic_features], dtype=int)

        symbols = np.array([len(variable.domain) for variable in self.symbolic_variables])
        max_variances = np.array([variable.std ** 2 for variable in self.numeric_variables])

        dependency_indices = dict()

        for variable, dep_vars in self.dependencies.items():
            # get the index version of the dependent variables and store them
            idx_var = self.variables_from_init.index(variable)
            idc_dep = [self.variables_from_init.index(var) for var in dep_vars]
            dependency_indices[idx_var] = idc_dep

        return Impurity(min_samples_leaf, numeric_vars, symbolic_vars, invert_impurity, n_sym_vars_total,
                        n_num_vars_total, numeric_features, symbolic_features, symbols, max_variances,
                        dependency_indices)

    def plot(self, sample_amount: int = 5000) -> go.Figure:
        """
        Plot the model.
        """
        subplot_titles = [distribution.__class__.__name__ for child in self.subcircuits
                          for distribution in child.subcircuits]
        figure = make_subplots(rows=len(self.subcircuits), cols=len(self.variables),
                               row_titles=[f"P(Leaf = {child_index}) = {weight}" for weight, child_index
                                           in zip(self.weights, range(len(self.subcircuits)))],
                               subplot_titles=subplot_titles)

        for child_index, child in enumerate(self.subcircuits):
            child: DecomposableProductUnit

            for distribution_index, distribution in enumerate(child.subcircuits):
                traces: List[go.Scatter] = distribution.plot()
                legend_group = child_index * len(self.variables) + distribution_index + 1
                traces = [trace.update(legendgroup=legend_group)
                          for trace in traces]
                figure.add_traces(traces, rows=child_index + 1, cols=distribution_index + 1)
                figure.update_xaxes(title_text=distribution.variable.name,
                                    row=child_index + 1,
                                    col=distribution_index + 1)

        figure.update_layout(height=300 * len(self.subcircuits), width=600 * len(self.variables),
                             title=f"Joint Probability Tree over {len(self.variables)} variables", )

        return figure

    def _variable_dependencies_to_json(self) -> Dict[str, List[str]]:
        """
        Convert the variable dependencies to a json compatible format.
        The result maps variable names to lists of variable names.
        """
        return {variable.name: [dependency.name for dependency in dependencies]
                for variable, dependencies in self.dependencies.items()}

    def empty_copy(self):
        result = self.__class__(variables=self.variables_from_init,
                                targets=self.targets,
                                features=self.features,
                                min_samples_leaf=self.min_samples_leaf,
                                min_impurity_improvement=self.min_impurity_improvement,
                                max_depth=self.max_depth,
                                dependencies=self.dependencies)
        return result

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["variables_from_init"] = [variable.to_json() for variable in self.variables_from_init]
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
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        sum_unit = DeterministicSumUnit._from_json(data)
        variables = [Variable.from_json(variable) for variable in data["variables_from_init"]]
        result = cls(variables, min_samples_leaf=data["_min_samples_leaf"],
                     min_impurity_improvement=data["min_impurity_improvement"],
                     max_leaves=data["max_leaves"], max_depth=data["max_depth"])
        for weight, subcircuit in sum_unit.weighted_subcircuits:
            result.mount(subcircuit)
            result.probabilistic_circuit.add_edge(result, subcircuit, weight=weight)
        return result

    def marginal(self, variables: Iterable[Variable], simplify_if_univariate=True, as_deterministic_sum=False) \
            -> Optional[Self]:
        """
        Marginalize the model to the given variables.
        :param variables: The variables to marginalize to.
        :param simplify_if_univariate: If the result is univariate, simplify it to a univariate distribution.
        :param as_deterministic_sum: If the result is univariate and discrete, return it as a deterministic sum
        instead of the distribution itself.
        :return: The marginal JPT.
        """
        result = super().marginal(variables)

        if result is None or len(result.variables) > 1 or not simplify_if_univariate:
            return result

        variable = result.variables[0]

        if isinstance(variable, Continuous):
            distribution = NygaDistribution.from_uniform_mixture(result)

        elif isinstance(variable, Discrete):
            weights = [result.probability(Event({variable: value})) for value in variable.domain]
            if isinstance(variable, Symbolic):
                distribution = SymbolicDistribution(variable, weights=weights)
            elif isinstance(variable, Integer):
                distribution = IntegerDistribution(variable, weights=weights)
            else:
                raise NotImplementedError(f"Variable {variable} is not supported.")
            if as_deterministic_sum:
                distribution = distribution.as_deterministic_sum()
        else:
            raise NotImplementedError(f"Variable {variable} not supported.")

        return distribution

    def area_validation_metric(self, other: Self) -> float:

        p_event, q_event = JPT.events_of_higher_density(self, other)
        result = (self.probability(p_event) - other.probability(p_event)
                  + other.probability(q_event) - self.probability(q_event))

        return result/2

        #
        # event_p = Event()
        # event_q = Event()
        #
        # for own_pro_unit in self.subcircuits:
        #     own_pro_unit: DecomposableProductUnit
        #     for other_pro_unit in other.subcircuits:
        #         other_pro_unit: DecomposableProductUnit
        #         if own_pro_unit.variables != other_pro_unit.variables:
        #             continue
        #         intersection = own_pro_unit.domain.intersection(other_pro_unit.domain)
        #         if not intersection.is_empty():
        #             intersection_point = []
        #             for var in intersection.variables:
        #                 var_lower = intersection.events[0][var].lower
        #                 var_upper = intersection.events[0][var].upper
        #                 point = var_lower + (var_upper - var_lower) / 2
        #                 intersection_point.append(point)
        #             if own_pro_unit.likelihood(intersection_point) > other_pro_unit.likelihood(intersection_point):
        #                 event_p = event_p.union(intersection)
        #             else:
        #                 event_q = event_q.union(intersection)
        #
        #         difference = own_pro_unit.domain.union(other_pro_unit.domain).difference(intersection) \
        #             if not intersection.is_empty() else own_pro_unit.domain.union(other_pro_unit.domain)
        #         #could be more events
        #         disjunct_events_li = []
        #         for diff_events in difference.events:
        #             eve_li = []
        #             for var in difference.variables:
        #                 portions = []
        #                 for p in diff_events[var]:
        #                     portions.append(Event({var: p}))
        #
        #                 if not eve_li:
        #                     eve_li = portions
        #                 else:
        #                     temp_list = []
        #                     for i in range(len(eve_li)):
        #                         for j in range(len(portions)):
        #                             dim_event = eve_li[i]
        #                             dim_event[var] = portions[j][var]
        #                             temp_list.append(dim_event)
        #                     eve_li = temp_list
        #             disjunct_events_li.extend(eve_li)
        #
        #         for d_event in disjunct_events_li:
        #             if (own_pro_unit.domain.intersection(d_event).is_empty()
        #                     and not other_pro_unit.domain.intersection(d_event).is_empty()):
        #                 event_q = event_q.union(d_event)
        #             elif (not own_pro_unit.domain.intersection(d_event).is_empty()
        #                     and other_pro_unit.domain.intersection(d_event).is_empty()):
        #                 event_p = event_p.union(d_event)
        # print(self.probability(event_p), other.probability(event_p))
        # print(self.probability(event_q), other.probability(event_q))
        # go.Figure(event_p.plot()).show()
        # go.Figure(event_q.plot()).show()
        # return (self.probability(event_p) - other.probability(event_p)
        #         + self.probability(event_q) - other.probability(event_q))


    def events_of_higher_density(self, other: Self):
        own_node_weights = self.probabilistic_circuit.nodes_weights()
        other_node_weights = other.probabilistic_circuit.nodes_weights()
        own_result = Event()
        other_result = Event()
        for own_pro_unit in self.subcircuits:
            own_pro_unit: DecomposableProductUnit
            for other_pro_unit in other.subcircuits:
                other_pro_unit: DecomposableProductUnit
                own_result_part, other_result_part = own_pro_unit.events_of_higher_density(other_pro_unit, own_node_weights, other_node_weights)
                if own_result.is_empty():
                    own_result = own_result_part
                else:
                    own_result = own_result.union(own_result_part)
                if other_result.is_empty():
                    other_result = other_result_part
                else:
                    other_result = other_result.union(other_result_part)


        return own_result, other_result