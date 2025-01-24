from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from inspect import isclass

import networkx as nx
import sqlalchemy
from sqlalchemy import Column, Table, select
from sqlalchemy.orm import Session, DeclarativeBase
from typing_extensions import Type, Iterable, Dict, Tuple, Any, Self, List

from .helper import fully_factorized
from .probabilistic_circuit import ProbabilisticCircuit


class EdgeType(str, Enum):
    ATTRIBUTE = "attribute"
    UNIQUE_PART = "unique part"
    EXCHANGEABLE_PART = "exchangeable part"
    RELATION_ARGUMENT_1 = "relates this"
    RELATION_ARGUMENT_2 = "to that"
    RELATION_FROM_TO = "from to"
    IS_RELATION_IN = "has relation"


class PartDecompositionBaseMixin(DeclarativeBase):
    """
    Mixin class for all tables in a database such that the part decomposition can be computed.
    For details on the definitions check https://www.scrofula.org/papers/nath-domingos15.pdf.
    """

    _aggregated_columns = dict()

    @classmethod
    def attributes(cls) -> Iterable[Column]:
        """
        In the article, this is referenced to as 'attributes'.

        :return: All columns that are atomic datatypes
        """
        return [column for column in cls.__table__.columns if column.primary_key is False
                and not column.foreign_keys]

    @classmethod
    def one_to_many_relationships(cls) -> Iterable[Type[Self]]:
        """
        In the article, this is referenced to as 'exchangeable parts'.

        :return: A list of tables that are related to the current table in a one-to-many relationship.
        """
        result = []
        for rel in cls.__mapper__.relationships:
            rel: sqlalchemy.orm.relationships.RelationshipProperty
            if rel.uselist:
                target_table = cls.get_class_of_table(rel.target)
                if issubclass(target_table, AssociationMixin):
                    continue
                for target_rel in target_table.__mapper__.relationships:
                    target_rel: sqlalchemy.orm.relationships.RelationshipProperty
                    if target_rel.target == cls.__table__:
                        if not target_rel.uselist:
                            result.append(cls.get_class_of_table(rel.target))
        return result

    @classmethod
    def one_to_one_relationships(cls) -> List[Type[Self]]:
        """
         In the article, this is referenced to as 'unique parts'.

        :return: A list of tables that are related to the current table in a one-to-one relationship.
        """
        result = []
        for rel in cls.__mapper__.relationships:

            rel: sqlalchemy.orm.relationships.RelationshipProperty

            if not rel.uselist:
                for target_rel in cls.get_class_of_table(rel.target).__mapper__.relationships:
                    target_rel: sqlalchemy.orm.relationships.RelationshipProperty
                    if target_rel.target == cls.__table__ and not target_rel.single_parent:
                        if not target_rel.uselist:
                            result.append(cls.get_class_of_table(rel.target))
        return result

    @classmethod
    def get_class_of_table(cls, table: Table) -> Type[Self]:
        """
        :param table: A table
        :return: The python class that corresponds to the table
        """

        for key, value in cls.registry._class_registry.items():
            if isclass(value):
                if value.__tablename__ == table.name:
                    return value

    def aggregation_statistics_for_relations(self) -> Dict[Type[PartDecompositionBaseMixin], Any]:
        """
        Calculate the aggregation statistics for all relations.
        """
        ...


class AssociationMixin(PartDecompositionBaseMixin):
    """
    Mixin class for all association tables that represent many-to-many relationships.
    These tables have to follow the association object pattern
    (https://docs.sqlalchemy.org/en/20/orm/basic_relationships.html#association-object)

    In the article, this is referenced to as 'relationship'.
    """
    __abstract__ = True

    @classmethod
    def associated_tables(cls):
        result = []
        for rel in cls.__mapper__.relationships:
            rel: sqlalchemy.orm.relationships.RelationshipProperty
            relation_member = cls.get_class_of_table(rel.target)
            result.append(relation_member)
        return result


@dataclass
class WrappedTable:
    """
    Wrapper Class for sqlalchemy tables.
    """
    table: Type[PartDecompositionBaseMixin]

    def __eq__(self, other):
        return hash(self.table) == hash(other.table)

    def __hash__(self):
        return hash(self.table)

    def __repr__(self):
        return self.table.__tablename__


class PartDecomposition(nx.DiGraph):
    """
    Representation of the Part Decomposition of a relational database.
    """

    base_table: Type[PartDecompositionBaseMixin]
    """
    The base class of all tables in the database.
    """

    def __init__(self, base_table: Type[PartDecompositionBaseMixin], **attr):
        super().__init__(**attr)
        self.base_table = base_table

    def all_wrapped_classes(self) -> Iterable[WrappedTable]:
        """
        :return: List of all classes (tables) in the database wrapped into the WrappedTable class.
        """
        return [WrappedTable(cls) for cls in self.base_table.registry._class_registry.values()
                if isclass(cls) and issubclass(cls, DeclarativeBase) and not issubclass(cls, AssociationMixin)]

    def all_wrapped_relations(self) -> Iterable[WrappedTable]:
        """
        :return: List of all association tables in the database wrapped into the WrappedTable class.
        """
        return [WrappedTable(cls) for cls in self.base_table.registry._class_registry.values()
                if isclass(cls) and issubclass(cls, AssociationMixin)]

    def edge_labels(self) -> Dict[Tuple[WrappedTable, WrappedTable], EdgeType]:
        """
        :return: Dictionary of mapping all edges to their labels.
        """
        return nx.get_edge_attributes(self, "label")

    def exchangeable_parts_edges(self):
        """
        :return: A list of all edges that describe exchangeable parts.
        """
        return [edge for edge, label in self.edge_labels().items() if label == EdgeType.EXCHANGEABLE_PART]

    def unique_parts_edges(self):
        return [edge for edge, label in self.edge_labels().items() if label == EdgeType.UNIQUE_PART]

    def attribute_edges(self):
        return [edge for edge, label in self.edge_labels().items() if label == EdgeType.ATTRIBUTE]

    def exchangeable_part_of(self, node: WrappedTable):
        return [edge[1] for edge in self.successors(node) if (node, edge) in self.exchangeable_parts_edges()]

    def unique_part_of(self, node: WrappedTable):
        return [edge[1] for edge in self.successors(node) if (node, edge) in self.unique_parts_edges()]

    def make_graph(self):
        for wrapped_table in self.all_wrapped_classes():
            self.add_node(wrapped_table)

            for attribute in wrapped_table.table.attributes():
                self.add_node(attribute)
                self.add_edge(wrapped_table, attribute, label=EdgeType.ATTRIBUTE)

            for part in wrapped_table.table.one_to_many_relationships():
                self.add_node(WrappedTable(part))
                self.add_edge(wrapped_table, WrappedTable(part), label=EdgeType.EXCHANGEABLE_PART)

            for other_table in wrapped_table.table.one_to_one_relationships():
                # self.add_edge(wrapped_table, WrappedTable(other_table), label="unique_part")
                self.add_edge(WrappedTable(other_table), wrapped_table, label=EdgeType.UNIQUE_PART)

        for wrapped_relation in self.all_wrapped_relations():
            self.add_node(wrapped_relation)

            for attribute in wrapped_relation.table.attributes():
                self.add_node(attribute)
                self.add_edge(wrapped_relation, attribute, label=EdgeType.ATTRIBUTE)

            association: AssociationMixin = wrapped_relation.table
            t1, t2 = association.associated_tables()
            t1, t2 = WrappedTable(t1), WrappedTable(t2)
            if t1 == t2:
                self.add_edge(wrapped_relation, t2, label=EdgeType.RELATION_FROM_TO)
            else:
                self.add_edge(wrapped_relation, t1, label=EdgeType.RELATION_ARGUMENT_1)
                self.add_edge(wrapped_relation, t2, label=EdgeType.RELATION_ARGUMENT_2)

            edge_labels = self.edge_labels()

            # find a class such that t1 and t2 are parts of it
            t1_incoming_edges = self.in_edges(t1)
            t1_parents = [edge[0] for edge in t1_incoming_edges if edge_labels[edge] in [EdgeType.EXCHANGEABLE_PART,
                                                                                         EdgeType.UNIQUE_PART]]
            t2_incoming_edges = self.in_edges(t2)
            t2_parents = [edge[0] for edge in t2_incoming_edges if edge_labels[edge] in [EdgeType.EXCHANGEABLE_PART,
                                                                                         EdgeType.UNIQUE_PART]]
            common_parents = list(set(t1_parents).intersection(t2_parents))
            assert len(common_parents) == 1, "I think that this must be 1"
            self.add_edge(common_parents[0], wrapped_relation, label=EdgeType.IS_RELATION_IN)

        return self

    def plot(self):
        roots = [node for node in self.nodes if len(list(self.predecessors(node))) == 0]

        pos = nx.bfs_layout(self, roots[0])
        edge_labels = {edge: label.value for edge, label in self.edge_labels().items()}
        nx.draw(self, pos=pos, with_labels=True)
        nx.draw_networkx_edge_labels(self, pos=pos, edge_labels=edge_labels)


class ExchangeableDistributionTemplate:
    """
    A distribution template that is exchangeable.
    Exchangeable means that it is permutation invariant, e.g. P(X, Y, Z) = P(Y, X, Z) = P(Y, Z, X) = ...
    """

    template_distribution: ProbabilisticCircuit

    def ground(self, variables):
        return fully_factorized(variables)


class RelationalProbabilisticCircuit(ProbabilisticCircuit):
    base_table: Type[DeclarativeBase]
    session: Session

    part_decomposition: PartDecomposition

    def __init__(self, base_table: Type[PartDecompositionBaseMixin], session: Session):
        super().__init__()
        self.base_table = base_table
        self.session = session
        self.part_decomposition = PartDecomposition(base_table).make_graph()

    def ground(self, session):
        ...

    def learn(self):
        tasks = deque()

    def gather_data(self, cls: Type[PartDecompositionBaseMixin]):
        node = WrappedTable(cls)
        result = self.session.scalars(select(cls)).all()

        attributes = cls.attributes()

        for element in result:
            for attribute in attributes:
                print(attribute.name, element.__dict__[attribute.name])
            print("-----------------")
