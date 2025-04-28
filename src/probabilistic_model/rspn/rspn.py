from dataclasses import dataclass
from typing import Type

from random_events.product_algebra import VariableMap
from random_events.variable import Variable
from typing_extensions import List
from dataclasses import Field

from ..probabilistic_circuit.nx.probabilistic_circuit import ProbabilisticCircuit, ProductUnit


@dataclass
class DistributionTemplate:
    model: Type
    template: ProbabilisticCircuit

    def ground(self, instances: List):
        """
        Create a distribution over the instances.
        :param instances: The instances to create the distribution over.
        :return:
        """

        result = ProbabilisticCircuit()
        root = ProductUnit(probabilistic_circuit=result)

        for instance in instances:
            grounded_template = self.template.__copy__()
            new_variables = VariableMap()
            for variable in grounded_template.variables:
                variable: Variable
                new_variable = variable.__class__(variable.name + str(id(instance)), variable.domain)
                new_variables[variable] = new_variable
            grounded_template.update_variables(new_variables)

            root.add_subcircuit(grounded_template.root)

        return result


def aggregate(field_name: str):
    def decorator(func):
        func._is_aggregate = True
        func._aggregate_field = field_name
        return func
    return decorator

def get_aggregated_methods(cls):
    return {
        name: method
        for name, method in cls.__dict__.items()
        if callable(method) and getattr(method, "_is_aggregate", False)
    }
