import numpy as np
from N2G import drawio_diagram

from ...probabilistic_circuit.nx.probabilistic_circuit import ProbabilisticCircuit


class NoLabel(str):

    def __bool__(self):
        return True

    def strip(self, __chars = None):
        return self

class DrawIoExporter:
    """
    Export a probabilistic circuit to a drawio diagram.
    """

    model: ProbabilisticCircuit
    """
    The probabilistic circuit to export.
    """

    def __init__(self, model: ProbabilisticCircuit):
        self.model = model

    def export(self) -> drawio_diagram:
        diagram = drawio_diagram()
        diagram.add_diagram("Structure", width=1360, height=1864)

        for unit, (x, y) in self.model.unit_positions_for_structure_plot().items():
            diagram.add_node(id=str(hash(unit)), x_pos=x * 100, y_pos=y * 100, **unit.drawio_style)
            if not unit.is_leaf:
                diagram.current_root[-1].attrib["label"] = " "

        for source, target in self.model.unweighted_edges:
            diagram.add_link(str(hash(source)), str(hash(target)), style='endArrow=classic;html=1;rounded=0;')

        for source, target, weight in self.model.weighted_edges:
            diagram.add_link(str(hash(source)), str(hash(target)), label=f"{round(weight,2)}", style='endArrow=classic;html=1;rounded=0;')

        return diagram


