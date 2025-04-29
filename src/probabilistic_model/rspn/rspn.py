from __future__ import annotations
import enum
from dataclasses import dataclass, fields
from functools import cached_property

from ormatic.field_info import FieldInfo
from typing import Type, get_type_hints, TypeVar, Generic, Dict

import networkx as nx
from random_events.product_algebra import VariableMap
from random_events.variable import Variable
from typing_extensions import List
from dataclasses import Field

from ..probabilistic_circuit.nx.probabilistic_circuit import ProbabilisticCircuit, ProductUnit

# Define type variables for parent and child
P = TypeVar('P')
C = TypeVar('C')

@dataclass
class Relation(Generic[P, C]):
    """
    Represents a relation between a parent and a child of specified types.
    """

    parent: P
    child: C


leaf_types = (float, int, str, enum.Enum)

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

@dataclass
class ClassSPN:

    clazz: Type
    """
    The top level class.
    """

    attribute_templates: Dict[str, DistributionTemplate]
    relation_templates: Dict[str, DistributionTemplate]
    part_spns: Dict[str, ClassSPN]

    @cached_property
    def parsed_fields(self) -> List[FieldInfo]:
        return [FieldInfo(self.clazz, f) for f in fields(self.clazz)]

    @cached_property
    def attributes(self) -> List[FieldInfo]:
        """
        :return: All fields of self.clazz which are univariate attributes (float, int, enum, str, etc)
        """
        return [f for f in self.parsed_fields if issubclass(f.type, leaf_types)]

    @cached_property
    def unique_parts(self) -> List[FieldInfo]:
        return [f for f in self.parsed_fields if not issubclass(f.type, leaf_types) and f.container is None]

    @property
    def exchangeable_parts(self) -> List[FieldInfo]:
        return [f for f in self.parsed_fields if not issubclass(f.type, leaf_types) and f.container is not None and not issubclass(f.type, Relation)]

    @property
    def relations(self):
        return [f for f in self.parsed_fields if not issubclass(f.type, leaf_types) and f.container is not None and issubclass(f.type, Relation)]

    def ground(self, instance) -> ProbabilisticCircuit:
        pc = ProbabilisticCircuit()
        root = ProductUnit(probabilistic_circuit=pc)

        # 1. Ground univariate attributes
        for field in self.attributes:
            dist_template = self.attribute_templates[field.name]
            value = getattr(instance, field.name)
            # Each template is responsible for returning a circuit for one instance
            leaf_circuit = dist_template.ground([value])  # wrap in list for compatibility
            root.add_subcircuit(leaf_circuit.root)

        # 2. Ground relations (EDTs over sets)
        for field in self.relations:
            rel_template = self.relation_templates[field.name]
            relation_instances = getattr(instance, field.name)
            relation_circuit = rel_template.ground(relation_instances)
            root.add_subcircuit(relation_circuit.root)

        # 3. Recursively ground unique parts
        for field in self.unique_parts:
            child_instance = getattr(instance, field.name)
            part_spn = self.part_spns[field.name]
            child_circuit = part_spn.ground(child_instance)
            root.add_subcircuit(child_circuit.root)

        # 4. Recursively ground exchangeable parts
        for field in self.exchangeable_parts:
            child_instances = getattr(instance, field.name)
            part_spn = self.part_spns[field.name]
            for child in child_instances:
                child_circuit = part_spn.ground(child)
                root.add_subcircuit(child_circuit.root)

        return pc


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
