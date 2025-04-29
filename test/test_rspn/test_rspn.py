import unittest
from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from random_events.interval import closed, open_closed
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Symbolic, Continuous

from probabilistic_model.probabilistic_circuit.nx.helper import uniform_measure_of_simple_event, \
    uniform_measure_of_event
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import ProbabilisticCircuit, SumUnit
from probabilistic_model.rspn.rspn import *

from dataclasses import Field, field
from typing import Any, TypeVar, Generic


class Element(str, Enum):
    H = "H"
    C = "C"
    O = "O"

@dataclass
class Atom:
    element: Element
    color: float

@dataclass
class Bond(Relation[Atom, Atom]):
    ...

@dataclass
class Molecule:
    mutagenic: bool
    size: float
    atoms: List[Atom]
    bonds: List[Bond]


    @aggregate("atoms")
    def mean_atom_color(self) -> int:
        return np.mean([a.color for a in self.atoms])


class RSPNTestCase(unittest.TestCase):

    def test_class_spn(self):
        atom_spn = ClassSPN(Atom)
        self.assertEqual(len(atom_spn.attributes), 2)

        molecule_spn = ClassSPN(Molecule)
        self.assertEqual(len(molecule_spn.attributes), 2)
        self.assertEqual(len(molecule_spn.unique_parts), 0)
        self.assertEqual(len(molecule_spn.exchangeable_parts), 1)
        self.assertEqual(len(molecule_spn.relations), 1)


    def test_rspn(self):

        # data
        a1 = Atom(Element.H, 0.1)
        a2 = Atom(Element.C, 0.2)
        a3 = Atom(Element.O, 0.3)

        m1 = Molecule(True, 1.0, [a1, a2, a3], [Bond(a1, a2)])

        # variables
        element_variable = Symbolic("element", Set.from_iterable(Element))
        color_variable = Continuous("color")
        mutagenic_variable = Symbolic("mutagenic", Set.from_iterable({True, False}))
        size_variable = Continuous("size")
        atom_aggregate_variable = Continuous("mean_atom_color")

        atom_domain = SimpleEvent({element_variable: element_variable.domain, color_variable: closed(0, 1)})
        atom_template = uniform_measure_of_simple_event(atom_domain)

        atom_template = DistributionTemplate(model=Atom, template=atom_template)
        atom_grounded = atom_template.ground(m1.atoms)
        self.assertEqual(len(atom_grounded.variables), len(m1.atoms) * 2)

        molecule_domain = SimpleEvent({mutagenic_variable: True, size_variable: closed(0, 1), atom_aggregate_variable: closed(-np.inf, 0)}).as_composite_set().complement()
        molecule_lim = SimpleEvent({mutagenic_variable: mutagenic_variable.domain, size_variable: closed(0, 3),
                                    atom_aggregate_variable: closed(-2, 2)}).as_composite_set()
        molecule_domain &= molecule_lim
        p_molecule = uniform_measure_of_event(molecule_domain)



        for leaf in p_molecule.leaves:
            if leaf.variables[0] != atom_aggregate_variable:
                    continue
            parents = leaf.parents
            for parent in parents:
                kwargs = {}
                if isinstance(parent, SumUnit):
                    kwargs["log_weight"] = p_molecule.get_edge_data(parent, leaf, "log_weight")
                parent.add_subcircuit(atom_grounded.root, **kwargs)
            p_molecule.remove_node(leaf)

        self.assertTrue(p_molecule.is_decomposable())