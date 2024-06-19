from N2G import drawio_diagram

from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit


class DrawIoExporter:

    model: ProbabilisticCircuit

    def __init__(self, model: ProbabilisticCircuit):
        self.model = model

    def export(self) -> drawio_diagram:
        diagram = drawio_diagram()
        diagram.add_diagram("Structure", width=1360, height=1864)
        for node in self.model.nodes:
            diagram.add_node(id=str(hash(node)), **node.draw_io_style())

        for source, target in self.model.unweighted_edges:
            diagram.add_link(str(hash(source)), str(hash(target)), style='endArrow=classic;html=1;rounded=0;')

        for source, target, weight in self.model.weighted_edges:
            diagram.add_link(str(hash(source)), str(hash(target)), label=f"{round(weight,2)}", style='endArrow=classic;html=1;rounded=0;')

        diagram.layout(algo="rt_circular")
        return diagram


