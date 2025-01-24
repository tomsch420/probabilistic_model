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

from probabilistic_model.probabilistic_circuit.nx.relational_probabilistic_circuit import (PartDecompositionBaseMixin,
                                                                                           AssociationMixin,
                                                                                           PartDecomposition,
                                                                                           RelationalProbabilisticCircuit)

class Base(MappedAsDataclass, PartDecompositionBaseMixin):
    __abstract__ = True

class Association(MappedAsDataclass, AssociationMixin):
    __abstract__ = True

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

        mexico = Nation(na, False)
        mexica_gov = Government("Pablo Escobar", "Cartel", mexico)

        anna = Person("Anna", 20, usa)
        bob = Person("Bob", 30, usa)
        carlos = Person("Carlos", 45, mexico)
        adj = Adjacent(usa, mexico)
        cf = Conflict(usa, mexico)
        s1 = Supports(anna, usa_gov)

        self.session.add_all([anna, bob, na, usa, s1, mexica_gov, mexico, usa_gov, carlos, adj, cf])
        self.session.commit()

    def test_get_persons_from_nation(self):
        nation = self.session.query(Nation).first()
        self.assertEqual(len(nation.persons), 2)

    def test_pd(self):
        pd = PartDecomposition(Base)
        pd.make_graph()
        pd.plot()
        plt.show()

    def test_gather_data(self):
        model = RelationalProbabilisticCircuit(Base, self.session)
        model.gather_data(Nation)


if __name__ == '__main__':
    unittest.main()
