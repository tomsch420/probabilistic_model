from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous, Integer, Symbolic, Variable
from typing_extensions import Iterable

from .distributions import UnivariateContinuousLeaf, UnivariateDiscreteLeaf
from .probabilistic_circuit import ProductUnit, SumUnit, ProbabilisticCircuit
from ...utils import MissingDict
from ...distributions import UniformDistribution, SymbolicDistribution, IntegerDistribution, GaussianDistribution
import plotly.graph_objects as go


def uniform_measure_of_event(event: Event) -> ProbabilisticCircuit:
    """
    Create a uniform measure for the given event.

    :param event: The event
    :return: The circuit describing the uniform measure
    """

    # calculate the bounding box of the event
    bounding_box = event.bounding_box()

    # create a uniform measure for the bounding box
    uniform_model = uniform_measure_of_simple_event(bounding_box)
    # condition the uniform measure on the event
    uniform_model, _ = uniform_model.conditional(event)

    return uniform_model

def uniform_measure_of_simple_event(simple_event: SimpleEvent) -> ProbabilisticCircuit:
    """
    Create a uniform measure for the given simple event.
    :param simple_event: The simple event
    :return: The circuit describing the uniform measure over the simple event
    """

    # initialize the root of the circuit
    uniform_model = ProductUnit()
    for variable, assignment in simple_event.items():

        # handle different variables
        if isinstance(variable, Continuous):

            # create a uniform distribution for every interval in a continuous variables description
            distribution = SumUnit()
            for assignment_ in assignment:
                u = UniformDistribution(variable, assignment_)
                distribution.add_subcircuit(UnivariateContinuousLeaf(u), 1 / u.pdf_value())
            distribution.normalize()

        # create uniform distribution for symbolic variables
        elif isinstance(variable, Symbolic):
            distribution = SymbolicDistribution(variable, MissingDict(float, {value: 1 / len(assignment.simple_sets) for
                                                                              value in assignment}))
            distribution = UnivariateDiscreteLeaf(distribution)

        # create uniform distribution for integer variables
        elif isinstance(variable, Integer):
            distribution = IntegerDistribution(variable,
                                               MissingDict(float, {value.lower: 1 / len(assignment.simple_sets) for
                                                                   value in assignment}))
            distribution = UnivariateDiscreteLeaf(distribution)

        else:
            raise NotImplementedError

        # mount the distribution on the root
        uniform_model.add_subcircuit(distribution)

    return uniform_model.probabilistic_circuit


def fully_factorized(variables: Iterable[Variable], means: dict, variances: dict) -> ProbabilisticCircuit:
    """
    Create a fully factorized distribution over a set of variables.
    For symbolic variables, the distribution is uniform.
    For continuous variables, the distribution is normal.

    :param variables: The variables.
    :param means: The means of the normal distributions.
    :param variances: The variances of the normal distributions.
    :return: The circuit describing the fully factorized normal distribution
    """

    # initialize the root of the circuit
    root = ProductUnit()
    for variable in variables:

        # create a normal distribution for every continuous variable
        if isinstance(variable, Continuous):
            distribution = GaussianDistribution(variable, means[variable], variances[variable])
            distribution = UnivariateContinuousLeaf(distribution)

        # create uniform distribution for symbolic variables
        elif isinstance(variable, Symbolic):
            domain_elements = list(variable.domain.simple_sets)
            distribution = SymbolicDistribution(variable, MissingDict(float, {int(v): 1/len(domain_elements)
                                                                              for v in domain_elements}))
            distribution = UnivariateDiscreteLeaf(distribution)
        else:
            raise NotImplementedError(f"Variable type not supported: {variable}")
        root.add_subcircuit(distribution)

    return root.probabilistic_circuit