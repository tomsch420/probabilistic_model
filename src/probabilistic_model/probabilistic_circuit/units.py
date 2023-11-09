import itertools
import random
import copy
from typing import Iterable, Tuple, List, Union, Optional, Any

from anytree import NodeMixin
from random_events.events import EncodedEvent, VariableMap
from random_events.variables import Variable, Integer, Continuous

from probabilistic_model.probabilistic_model import ProbabilisticModel
from typing_extensions import Self


class Unit(ProbabilisticModel, NodeMixin):
    """
    Abstract class for nodes used in a probabilistic circuit
    """

    def __init__(self, variables: Iterable[Variable], parent: 'Unit' = None):
        self.parent = parent
        super().__init__(variables)
        NodeMixin.__init__(self)

    def variable_indices_of_child(self, child: 'Unit') -> List[int]:
        """
        Get the list of the variables' indices in self that are also in child.

        :param child: The child to check for.
        :return: The indices
        """
        return list(index for index, variable in enumerate(self.variables) if variable in child.variables)

    def __add__(self, other) -> Union['SumUnit', 'SmoothSumUnit']:
        if not isinstance(other, Unit):
            raise ValueError(f"Cannot add a Probabilistic Circuit with {type(other)}.")

        joined_variables = set(self.variables).union(other.variables)

        # if the sum is smooth
        if set(self.variables) == set(other.variables):
            # create a smooth sum unit
            result = SmoothSumUnit(variables=sorted(joined_variables), weights=[.5, .5])
        else:
            # create an ordinary sum unit
            result = SumUnit(variables=sorted(joined_variables), weights=[.5, .5])

        result.children = [self, other]
        return result

    def __mul__(self, other) -> Union['ProductUnit', 'DecomposableProductUnit']:
        if not isinstance(other, Unit):
            raise ValueError(f"Cannot add a Probabilistic Circuit with {type(other)}.")

        joined_variables = set(self.variables).union(other.variables)

        # check if product is decomposable
        if set(self.variables).intersection(other.variables) == set():
            result = DecomposableProductUnit(variables=sorted(joined_variables))
        else:
            result = ProductUnit(variables=sorted(joined_variables))
        result.children = [self, other]
        return result

    def _copy_children(self) -> Iterable['Unit']:
        """
        Copy the children of this unit.

        :return: The copied children.
        """
        return [copy.copy(child) for child in self.children]

    def __copy__(self):
        result = self.__class__(self.variables)
        result.children = self._copy_children()
        return result

    def _parameter_copy(self):
        """
        :return: a new instance of this class taking over only the parameters but not the children.
        """
        return self.__class__(self.variables)

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:

        # calculate intersection of variables
        variable_intersection = set(self.variables).intersection(set(variables))

        # if the intersection is empty, return None
        if len(variable_intersection) == 0:
            return None

        # create parameter copy
        result = self._parameter_copy()

        # update variables
        result.variables = variable_intersection

        # list for marginalized children
        marginal_children = []

        # recurse into children
        for child in self.children:

            # marginalize current child
            marginal_child = child.marginal(variable_intersection)

            # if the result is None, skip it
            if marginal_child is None:
                continue

            # append marginal distribution
            marginal_children.append(marginal_child)

        # update children
        result.children = marginal_children
        return result

    def filter_variable_map_by_self(self, variable_map: VariableMap):
        """
        Filter a variable map by the variables of this unit.
        :param variable_map: The map to filter
        :return: The map filtered by the variables of this unit.
        """
        return variable_map.__class__({variable: value for variable, value in variable_map.items()
                                       if variable in self.variables})


class SumUnit(Unit):
    """
    Abstract class for sum units.
    """

    weights: Iterable
    """The weights of the convex sum unit."""

    def __init__(self, variables: Iterable[Variable], weights: Iterable, parent: 'Unit' = None):
        super().__init__(variables, parent)
        self.weights = weights

    def normalize(self) -> Self:
        """
        Normalize the weights of this sum unit to a convex sum.

        :return: The normalized sum unit.
        """
        sum_of_weights = sum(self.weights)
        normalized_weights = [weight / sum_of_weights for weight in self.weights]
        result = self.__class__(self.variables, normalized_weights)
        result.children = self.children
        return result

    def __copy__(self):
        result = self.__class__(self.variables, copy.copy(self.weights))
        result.children = self._copy_children()
        return result

    def _parameter_copy(self):
        """
        :return: a new instance of this class taking over only the parameters but not the children.
        """
        return self.__class__(self.variables, self.weights)

    def __str__(self):
        return ("(" + " + ".join([f"{weight} * {str(child)}" for weight, child in zip(self.weights, self.children)]) +
                ")")

    def __repr__(self):
        return "+"


class SmoothSumUnit(SumUnit):
    """
    Smooth sum unit used in a probabilistic circuit
    """

    def __init__(self, variables: Iterable[Variable], weights: Iterable, parent: 'Unit' = None):
        super().__init__(variables, weights, parent)

    def _likelihood(self, event: Iterable) -> float:
        return sum([weight * child._likelihood(event) for weight, child in zip(self.weights, self.children)])

    def _probability(self, event: EncodedEvent) -> float:
        return sum([weight * child._probability(event) for weight, child in zip(self.weights, self.children)])

    def sample(self, amount: int) -> Iterable:
        """
        Sample from the sum node using the latent variable interpretation.
        """

        # sample the latent variable
        states = random.choices(list(range(len(self.children))), weights=self.weights, k=amount)

        # sample from the children
        result = []
        for index, child in enumerate(self.children):
            result.extend(child.sample(states.count(index)))
        return result

    def _conditional(self, event: EncodedEvent) -> Tuple[Optional[Self], float]:
        """
        Calculate the condition probability distribution using the latent variable interpretation and bayes theorem.

        :param event: The event to condition on
        :return:
        """

        # conditional weights of new sum unit
        conditional_weights = []

        # conditional children of new sum unit
        conditional_children = []

        # initialize probability
        probability = 0.

        for weight, child in zip(self.weights, self.children):
            conditional_child, conditional_probability = child._conditional(event)

            if conditional_probability == 0:
                continue

            conditional_probability = conditional_probability * weight
            probability += conditional_probability

            conditional_weights.append(conditional_probability)
            conditional_children.append(conditional_child)

        if probability == 0:
            return None, 0

        result = self.__class__(self.variables, conditional_weights)
        result.children = conditional_children
        return result.normalize(), probability

    def moment(self, order: VariableMap[Union[Integer, Continuous], int],
               center: VariableMap[Union[Integer, Continuous], float]) \
            -> VariableMap[Union[Integer, Continuous], float]:

        # create map for orders and centers
        order_of_self = self.filter_variable_map_by_self(order)
        center_of_self = self.filter_variable_map_by_self(center)

        # initialize result
        result = VariableMap({variable: 0 for variable in order_of_self})

        # for every weighted child
        for weight, child in zip(self.weights, self.children):

            # calculate the moment of the child
            child_moment = child.moment(order_of_self, center_of_self)

            # add up the linear combination of the child moments
            for variable, moment in child_moment.items():
                result[variable] += weight * moment

        return result


class DeterministicSumUnit(SmoothSumUnit):
    """
    Deterministic sum node used in a probabilistic circuit
    """

    def merge_modes_if_one_dimensional(self, modes: List[EncodedEvent]) -> List[EncodedEvent]:
        """
        Merge the modes in `modes` to one mode if the model is one dimensional.

        :param modes: The modes to merge.
        :return: The (possibly) merged modes.
        """
        if len(self.variables) > 1:
            return modes

        # merge modes
        mode = modes[0]

        for mode_ in modes[1:]:
            mode = mode | mode_

        return [mode]

    def _mode(self) -> Tuple[Iterable[EncodedEvent], float]:
        """
        Calculate the mode of the model.
        As there may exist multiple modes, this method returns an Iterable of modes and their likelihood.

        :return: The internal representation of the mode and the likelihood.
        """
        modes = []
        likelihoods = []

        # gather all modes from the children
        for weight, child in zip(self.weights, self.children):
            mode, likelihood = child._mode()
            modes.append(mode)
            likelihoods.append(weight * likelihood)

        # get the most likely result
        maximum_likelihood = max(likelihoods)

        result = []

        # gather all results that are maximum likely
        for mode, likelihood in zip(modes, likelihoods):
            if likelihood == maximum_likelihood:
                result.extend(mode)

        return self.merge_modes_if_one_dimensional(result), maximum_likelihood

    @staticmethod
    def from_sum_unit(unit: SmoothSumUnit) -> 'DeterministicSumUnit':
        """
        Downcast a sum unit to a deterministic sum unit.

        :param unit: The sum unit to downcast.
        """
        result = DeterministicSumUnit(variables=unit.variables, weights=unit.weights)
        result.children = unit.children
        return result


class ProductUnit(Unit):
    """
    Product node used in a probabilistic circuit
    """

    def __str__(self):
        return "(" + " * ".join([f"{str(child)}" for child in self.children]) + ")"

    def __repr__(self):
        return "*"


class DecomposableProductUnit(ProductUnit):
    """
    Decomposable product node used in a probabilistic circuit
    """

    def _likelihood(self, event: Iterable) -> float:
        result = 1.

        for child in self.children:
            indices = self.variable_indices_of_child(child)

            partial_event = [event[index] for index in indices]

            result = result * child._likelihood(partial_event)

        return result

    def _probability(self, event: EncodedEvent) -> float:
        result = 1.

        for child in self.children:
            # construct partial event for child
            result = result * child._probability(EncodedEvent({variable: event[variable] for variable in
                                                               self.variables}))

        return result

    def _mode(self) -> Tuple[Iterable[EncodedEvent], float]:
        """
        Calculate the mode of the model.
        As there may exist multiple modes, this method returns an Iterable of modes and their likelihood.

        :return: The internal representation of the mode and the likelihood.
        """
        modes = []
        resulting_likelihood = 1.

        # gather all modes from the children
        for child in self.children:
            mode, likelihood = child._mode()
            modes.append(mode)
            resulting_likelihood *= likelihood

        result = []

        # perform the cartesian product of all modes
        for mode_combination in itertools.product(*modes):

            # form the intersection of the modes inside one cartesian product mode
            mode = mode_combination[0]
            for mode_ in mode_combination[1:]:
                mode = mode | mode_

            result.append(mode)

        return result, resulting_likelihood

    def _conditional(self, event: EncodedEvent) -> Tuple[Self, float]:

        # initialize probability and new children
        probability = 1.
        conditional_children = []

        for child in self.children:

            # get conditional child and probability in pre-order
            conditional_child, conditional_probability = child._conditional(event)

            # if any is 0, the whole probability is 0
            if conditional_probability == 0:
                return None, 0

            # update probability and children
            probability *= conditional_probability
            conditional_children.append(conditional_child)

        # construct conditional product node
        conditional_self = self.__class__(self.variables)
        conditional_self.children = conditional_children

        return conditional_self, probability

    def sample(self, amount: int) -> List[List[Any]]:

        # list for the samples content in the same order as self.variables
        rearranged_sample = [[None] * len(self.variables)] * amount

        for child in self.children:
            sample_subset = child.sample(amount)

            for sample_index in range(amount):
                for child_variable_index, variable in enumerate(child.variables):

                    rearranged_sample[sample_index][self.variables.index(variable)] \
                        = sample_subset[sample_index][child_variable_index]

        return rearranged_sample

    def moment(self, order: VariableMap[Union[Integer, Continuous], int],
               center: VariableMap[Union[Integer, Continuous], float])\
            -> VariableMap[Union[Integer, Continuous], float]:

        # initialize result
        result = VariableMap()

        for child in self.children:

            # calculate the moment of the child
            child_moment = child.moment(order, center)

            result = VariableMap({**result, **child_moment})

        return result
