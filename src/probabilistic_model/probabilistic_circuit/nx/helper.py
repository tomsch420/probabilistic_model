from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous, Integer, Symbolic

from .probabilistic_circuit import ProductUnit, SumUnit, ProbabilisticCircuit
from ...utils import MissingDict
from .distributions import UniformDistribution, SymbolicDistribution, IntegerDistribution
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
                distribution.add_subcircuit(u, 1 / u.pdf_value())
            distribution.normalize()

        # create uniform distribution for symbolic variables
        elif isinstance(variable, Symbolic):
            distribution = SymbolicDistribution(variable, MissingDict(float, {value: 1 / len(assignment.simple_sets) for
                                                                              value in assignment}))

        # create uniform distribution for integer variables
        elif isinstance(variable, Integer):
            distribution = IntegerDistribution(variable,
                                               MissingDict(float, {value.lower: 1 / len(assignment.simple_sets) for
                                                                   value in assignment}))
        else:
            raise NotImplementedError

        # mount the distribution on the root
        uniform_model.add_subcircuit(distribution)

    return uniform_model.probabilistic_circuit