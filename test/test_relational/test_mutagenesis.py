from __future__ import annotations

import unittest
from functools import cached_property
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sqlalchemy import create_engine, select, ForeignKey
from sqlalchemy.orm import Mapped, MappedAsDataclass, mapped_column, relationship, Session

from probabilistic_model.probabilistic_circuit.nx.relational_probabilistic_circuit import PartDecompositionBaseMixin, \
    AssociationMixin, PartDecomposition, RelationalProbabilisticCircuit


class Base(MappedAsDataclass, PartDecompositionBaseMixin):
    """
    generated via sqlacodegen --generator dataclasses mysql+pymysql://guest:ctu-relational@relational.fel.cvut.cz:3306/mutagenesis
    """
    __abstract__ = True


class Association(Base, AssociationMixin):
    __abstract__ = True


class Atom(Base):
    __tablename__ = "atom"

    atom_id: Mapped[str] = mapped_column(init=False, primary_key=True)

    molecule_id: Mapped[str] = mapped_column(ForeignKey('molecule.molecule_id'), init=False)
    molecule: Mapped[Molecule] = relationship("Molecule", foreign_keys=[molecule_id], back_populates="atoms")
    element: Mapped[str]
    type: Mapped[int]
    charge: Mapped[float]


class Molecule(Base):
    __tablename__ = "molecule"

    _aggregated_columns = dict()
    molecule_id: Mapped[str] = mapped_column(init=False, primary_key=True)
    ind1: Mapped[int]
    inda: Mapped[int]
    logp: Mapped[float]
    lumo: Mapped[float]
    mutagenic: Mapped[str]
    atoms: Mapped[List[Atom]] = relationship("Atom", back_populates="molecule", init=False)

    @cached_property
    def mean_charge_of_atoms(self):
        return np.mean([atom.charge for atom in self.atoms])
    _aggregated_columns[atoms] = mean_charge_of_atoms


class Bond(Association):
    __tablename__ = "bond"

    atom1_id: Mapped[str] = mapped_column(ForeignKey('atom.atom_id'), init=False, primary_key=True)
    atom2_id: Mapped[str] = mapped_column(ForeignKey('atom.atom_id'), init=False, primary_key=True)

    # type: Mapped[int]
    atom1: Mapped[Atom] = relationship(foreign_keys=[atom1_id])
    atom2: Mapped[Atom] = relationship(foreign_keys=[atom2_id])



class MutagenesisTestCase(unittest.TestCase):
    session: Session

    @classmethod
    def setUpClass(cls):
        engine = create_engine("mysql+pymysql://guest:guest@localhost:3306/mutagenesis")
        cls.session = Session(engine)

    def test_data_getting(self):
        for cls in [Molecule, Atom, Bond]:
            r = self.session.scalars(select(cls)).all()
            self.assertGreater(len(r), 0)

    def test_aggregation_statistics(self):
        print(Molecule._aggregated_columns)
        print(Atom._aggregated_columns)
        exit()
        for m in self.session.scalars(select(Molecule)).all():
            print(m.mean_charge_of_atoms)
            exit()

    def test_pd(self):
        pd = PartDecomposition(Base).make_graph()
        pd.plot()
        plt.show()

    def test_rspn(self):
        model = RelationalProbabilisticCircuit(Base, self.session)
        model.gather_data(Atom)
