from __future__ import annotations

import unittest
from dataclasses import dataclass
from enum import Enum
from inspect import isclass

import networkx as nx

from matplotlib import pyplot as plt
from sqlalchemy import create_engine, ForeignKey, UniqueConstraint, Engine, inspect, Table, \
    MetaData, Column
from sqlalchemy.orm import MappedAsDataclass, DeclarativeBase, mapped_column, Mapped, Session, relationship
from typing_extensions import List, Type, Iterable, Dict, Tuple, Any

from probabilistic_model.probabilistic_circuit.nx.helper import fully_factorized
from probabilistic_model.probabilistic_model import ProbabilisticModel
import sqlalchemy.schema


class EdgeType(str, Enum):
    ATTRIBUTE = "attribute"
    UNIQUE_PART = "unique part"
    EXCHANGEABLE_PART = "exchangeable part"
    RELATION_ARGUMENT_1 = "relates this"
    RELATION_ARGUMENT_2 = "to that"
    RELATION_FROM_TO = "from to"
    IS_RELATION_IN = "has relation"

class Base(MappedAsDataclass, DeclarativeBase):

    @classmethod
    def attributes(cls) -> Iterable[Column]:
        """
        In the article, this is referenced to as 'attributes'.

        :return: All columns that are atomic datatypes
        """
        return [column for column in cls.__table__.columns if column.primary_key is False
                and not column.foreign_keys]

    @classmethod
    def one_to_many_relationships(cls) -> Iterable[Type[Base]]:
        """
        This is in the paper referenced to as 'exchangeable parts'.

        :return: A list of tables that are related to the current table in a one-to-many relationship.
        """
        result = []
        for rel in cls.__mapper__.relationships:
            rel: sqlalchemy.orm.relationships.RelationshipProperty
            if rel.uselist:
                target_table = cls.get_class_of_table(rel.target)
                if issubclass(target_table, Association):
                    continue
                for target_rel in target_table.__mapper__.relationships:
                    target_rel: sqlalchemy.orm.relationships.RelationshipProperty
                    if target_rel.target == cls.__table__:
                        if not target_rel.uselist:
                            result.append(cls.get_class_of_table(rel.target))
        return result

    @classmethod
    def one_to_one_relationships(cls) -> List[Type[Base]]:
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
    def get_class_of_table(cls, table: Table) -> Type[Base]:
        return cls.registry._class_registry[table.name]


class Association(Base):
    """
    Super class for all association table many-to-many relationships.
    This has to follow the association object pattern
    (https://docs.sqlalchemy.org/en/20/orm/basic_relationships.html#association-object)
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


class Government(Base):
    __tablename__ = "Government"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    name: Mapped[str]
    form: Mapped[str]

    nation_id: Mapped[int] = mapped_column(ForeignKey('Nation.id'), init=False)
    nation: Mapped[Nation] = relationship(back_populates="government", single_parent=True)

    supported_by: Mapped[List[Supports]] = relationship(back_populates="government", init=False)
    __table_args__ = (UniqueConstraint("nation_id"),)


class Person(Base):
    __tablename__ = "Person"
    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    name: Mapped[str]
    age: Mapped[int]

    nation_id: Mapped[int] = mapped_column(ForeignKey('Nation.id'), init=False)
    nation: Mapped[Nation] = relationship("Nation", foreign_keys=[nation_id], back_populates="persons")
    supports: Mapped[List[Supports]] = relationship(back_populates="person", init=False)


class Region(Base):
    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    name: Mapped[str]
    nations: Mapped[List[Nation]] = relationship("Nation", back_populates="region", init=False)
    __tablename__ = "Region"


class Nation(Base):
    __tablename__ = "Nation"
    id: Mapped[int] = mapped_column(init=False, primary_key=True)

    region_id: Mapped[int] = mapped_column(ForeignKey('Region.id'), init=False)
    region: Mapped[Region] = relationship("Region", foreign_keys=[region_id], back_populates="nations")

    high_gdp: Mapped[bool]  # this is an attribute
    government: Mapped[Government] = relationship(back_populates="nation", init=False)
    persons: Mapped[List[Person]] = relationship("Person", back_populates="nation", init=False)



class Supports(Association):
    __tablename__ = "Supports"
    person_id: Mapped[int] = mapped_column(ForeignKey('Person.id'), primary_key=True, init=False)
    government_id: Mapped[int] = mapped_column(ForeignKey('Government.id'), primary_key=True, init=False)

    person: Mapped[Person] = relationship(back_populates="supports")
    government: Mapped[Government] = relationship(back_populates="supported_by")


class Adjacent(Association):
    __tablename__ = "Adjacent"
    left_id: Mapped[int] = mapped_column(ForeignKey("Nation.id"), primary_key=True, init=False)
    right_id: Mapped[int] = mapped_column(ForeignKey("Nation.id"), primary_key=True, init=False)
    left: Mapped[Nation] = relationship(foreign_keys=[left_id])
    right: Mapped[Nation] = relationship(foreign_keys=[right_id])


class Conflict(Association):
    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    __tablename__ = "Conflict"
    left_id: Mapped[int] = mapped_column(ForeignKey("Nation.id"), primary_key=True, init=False)
    right_id: Mapped[int] = mapped_column(ForeignKey("Nation.id"), primary_key=True, init=False)
    left: Mapped[Nation] = relationship(foreign_keys=[left_id])
    right: Mapped[Nation] = relationship(foreign_keys=[right_id])



class RSPNClass:
    attributes: set
    unique_parts: set
    exchangeable_parts: set
    relations: set
    model: ProbabilisticModel

@dataclass
class WrappedTable:
    table: Type[Base]

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

    base_table: Type[Base]
    """
    The base class of all tables in the database.
    """

    def __init__(self, base_table: Type[Base], **attr):
        super().__init__(**attr)
        self.base_table = base_table

    def all_wrapped_classes(self) -> Iterable[WrappedTable]:
        yield from [WrappedTable(cls) for cls in self.base_table.registry._class_registry.values()
                    if isclass(cls) and issubclass(cls, Base) and not issubclass(cls, Association)]

    def all_wrapped_relations(self) -> Iterable[WrappedTable]:
        yield from [WrappedTable(cls) for cls in self.base_table.registry._class_registry.values()
                    if isclass(cls) and issubclass(cls, Association)]

    def edge_labels(self) -> Dict[Tuple[Any, Any], EdgeType]:
        return nx.get_edge_attributes(self, "label")

    def exchangeable_parts_edges(self):
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
            association: Association = wrapped_relation.table
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

    def ground(self, variables):
        return fully_factorized(variables)


class RelationalSPN:
    base_table: Type[Base]
    session: Session

    def __init__(self, base_table: Type[Base], session: Session):
        self.base_table = base_table
        self.session = session

    def learn(self):
        ...


class RSPNTestCase(unittest.TestCase):
    session: Session

    @classmethod
    def setUpClass(cls):
        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(engine)
        cls.session = Session(engine)

    def setUp(self):
        na = Region("North America")
        usa = Nation(na, True)
        usa_gov = Government("Trump", "Republic", usa)
        anna = Person("Anna", 20, usa)
        bob = Person("Bob", 30, usa)
        s1 = Supports(anna, usa_gov)

        self.session.add_all([anna, bob, na, usa, s1])
        self.session.commit()

    def test_get_persons_from_nation(self):
        nation = self.session.query(Nation).first()
        self.assertEqual(len(nation.persons), 2)

    def test_pd(self):
        pd = PartDecomposition(Base)
        pd.make_graph()
        pd.plot()
        plt.show()


if __name__ == '__main__':
    unittest.main()
