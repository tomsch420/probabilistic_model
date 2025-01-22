from __future__ import annotations
import unittest
from dataclasses import dataclass

import networkx as nx
from matplotlib import pyplot as plt
from random_events.set import SetElement
from random_events.utils import recursive_subclasses
from sqlalchemy import create_engine, select, ForeignKey, Column, Integer, UniqueConstraint, Engine, inspect, Table, \
    MetaData
from sqlalchemy.orm import MappedAsDataclass, DeclarativeBase, mapped_column, Mapped, Session, relationship
from typing_extensions import List, Iterable, Type

from probabilistic_model.probabilistic_circuit.nx.helper import fully_factorized
from probabilistic_model.probabilistic_model import ProbabilisticModel


def attributes_of_table(table: Table):
    return [column for column in table.columns if column.primary_key is False
            and not column.foreign_keys]

def exchangeable_parts_of_table(table: Table):
    for fk in table.foreign_keys:
        if fk.column.table == table:
            yield fk.column.table

def unique_parts_of_table(table: Table, engine: Engine):
    inspector = inspect(engine)
    table_name = table.name
    unique_constraints = inspector.get_unique_constraints(table_name)

    result = []
    for constraint in unique_constraints:
        constraint_columns = [table.c[col_name] for col_name in constraint['column_names']]
        result.append((constraint['name'], constraint_columns))

    return result

def relations_of_table(table: Table):
    ...

class Base(MappedAsDataclass, DeclarativeBase):

    @classmethod
    def attributes(cls):
        return [column for column in cls.__table__.columns if column.primary_key is False
                and not column.foreign_keys ]

    @classmethod
    def exchangeable_parts(cls):
        for table in cls.metadata.tables.values():
            for fk in table.foreign_keys:
                if fk.column.table == cls.__table__:
                    yield table

    @classmethod
    def unique_parts(cls, engine: Engine):
        inspector = inspect(engine)
        table_name = cls.__tablename__
        unique_constraints = inspector.get_unique_constraints(table_name)

        table = Table(table_name, MetaData(), autoload_with=engine)
        result = []
        for constraint in unique_constraints:
            constraint_columns = [table.c[col_name] for col_name in constraint['column_names']]
            result.append((constraint['name'], constraint_columns))

        return result
class Government(Base):
    __tablename__ = "Government"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    name: Mapped[str]
    form: Mapped[str]

    nation_id: Mapped[int] = mapped_column(ForeignKey('Nation.id'), init=False)
    nation: Mapped[Nation] = relationship(back_populates="government", single_parent=True)
    __table_args__ = (UniqueConstraint("nation_id"),)

class Person(Base):
    __tablename__ = "Person"
    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    name: Mapped[str]
    age: Mapped[int]
    nation_id: Mapped[int] = mapped_column(ForeignKey('Nation.id'), init=False)
    nation: Mapped[Nation] = relationship("Nation", foreign_keys=[nation_id])

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
    government: Mapped[Government] = relationship(back_populates="nation", init=False)
    high_gdp: Mapped[bool] # this is an attribute

class Adjacent(Base):
    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    __tablename__ = "Adjacent"
    nation_1_id: Mapped[int] = mapped_column(ForeignKey('Nation.id'), init=False)
    nation_2_id: Mapped[int] = mapped_column(ForeignKey('Nation.id'), init=False)
    nation_1: Mapped[Nation] = relationship("Nation", foreign_keys=[nation_1_id])
    nation_2: Mapped[Nation] = relationship("Nation", foreign_keys=[nation_2_id])

class Conflict(Base):
    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    __tablename__ = "Conflict"
    nation_1_id: Mapped[int] = mapped_column(ForeignKey('Nation.id'), init=False)
    nation_2_id: Mapped[int] = mapped_column(ForeignKey('Nation.id'), init=False)
    nation_1: Mapped[Nation] = relationship("Nation", foreign_keys=[nation_1_id])
    nation_2: Mapped[Nation] = relationship("Nation", foreign_keys=[nation_2_id])

class Supports(Base):
    __tablename__ = "Supports"
    person_id: Mapped[int] = mapped_column(ForeignKey('Person.id'), primary_key=True, init=False)
    person: Mapped[Person] = relationship("Person", foreign_keys=[person_id])

    nation_id: Mapped[int] = mapped_column(ForeignKey('Nation.id'), primary_key=True, init=False)
    nation: Mapped[Nation] = relationship("Nation", foreign_keys=[nation_id])
    value: Mapped[bool]

class RSPNClass:
    attributes: set
    unique_parts: set
    exchangeable_parts: set
    relations: set
    model: ProbabilisticModel


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

    def part_decomposition(self):
        part_decomposition = nx.DiGraph()
        for table in self.base_table.metadata.tables.values():
            part_decomposition.add_node(str(table))
            print(table)
            print(*exchangeable_parts_of_table(table))
            print("--")

            for attribute in attributes_of_table(table):
                part_decomposition.add_node(str(attribute))
                part_decomposition.add_edge(str(table), str(attribute), label="attribute")

            for part in exchangeable_parts_of_table(table):
                part_decomposition.add_node(str(part))
                part_decomposition.add_edge(str(table), str(part), label="exchangeable")
            #
            # for name, columns in table.unique_parts(self.session.get_bind()):
            #     part_decomposition.add_node(name)
            #     part_decomposition.add_edge(table, name, label="unique")
            #     for column in columns:
            #         part_decomposition.add_node(column)
            #         part_decomposition.add_edge(name, column, label="unique")


        return part_decomposition

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
        s1 = Supports(anna, usa, True)
        s2 = Supports(bob, usa, False)

        self.session.add_all([anna, bob, na, usa, s1, s2])
        self.session.commit()

    def test_learn(self):
        model = RelationalSPN(Base, self.session)
        # print(*(Region.exchangeable_parts()), sep="\n")
        # print(*(Government.unique_parts(self.session.get_bind())), sep="\n")

        pd = model.part_decomposition()
        pos = nx.spring_layout(pd)
        edge_labels = {edge: str(nx.get_edge_attributes(pd, "label")[edge]) for edge in pd.edges}
        nx.draw(pd, pos=pos, with_labels=True)
        nx.draw_networkx_edge_labels(pd, pos=pos, edge_labels=edge_labels)
        plt.show()

if __name__ == '__main__':
    unittest.main()
