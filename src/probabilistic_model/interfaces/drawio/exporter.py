import numpy as np
from N2G import drawio_diagram

from ...probabilistic_circuit.nx.probabilistic_circuit import ProbabilisticCircuit


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

        # do a layer-wise BFS
        layers = self.model.layers

        # calculate the positions of the nodes
        maximum_layer_width = max([len(layer) for layer in layers])
        i = 1
        for depth, layer in enumerate(layers):
            i +=1
            number_of_nodes = len(layer)
            positions_in_layer = np.linspace(0, maximum_layer_width, number_of_nodes, endpoint=False)
            positions_in_layer += (maximum_layer_width - len(layer)) / (2 * len(layer))
            for position, node in zip(positions_in_layer, layer):
                diagram.add_node(id=str(hash(node)), x_pos=position*25, y_pos=i*25, **node.drawio_style)

        # for node in self.model.nodes:
        #     diagram.add_node(id=str(hash(node)), x_pos= 0, y_pos= 0, **node.draw_io_style())

        for source, target in self.model.unweighted_edges:
            diagram.add_link(str(hash(source)), str(hash(target)), style='endArrow=classic;html=1;rounded=0;')

        for source, target, weight in self.model.weighted_edges:
            diagram.add_link(str(hash(source)), str(hash(target)), label=f"{round(weight,2)}", style='endArrow=classic;html=1;rounded=0;')

        diagram.layout(algo="tree")
        return diagram


