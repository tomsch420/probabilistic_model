import graphviz
from anytree import PreOrderIter

from probabilistic_model.probabilistic_circuit.distributions import UnivariateDistribution
from probabilistic_model.probabilistic_circuit.units import Unit
import plotly.graph_objects as go
import tempfile


class GraphVizExporter:

    model: Unit

    def __init__(self, model: Unit):
        self.model = model

    def to_graphviz(self) -> graphviz.Digraph:
        dot = graphviz.Digraph(node_attr={'shape': 'plaintext'})

        for node in PreOrderIter(self.model, filter_=lambda n: not isinstance(n.parent, UnivariateDistribution)):
            node: Unit

            if isinstance(node, UnivariateDistribution):
                figure = go.Figure()
                figure.add_traces(node.plot())
                figure.update_layout(title=node.__class__.__name__, xaxis_title=node.variable.name, template="plotly")
                figure.write_image(f"{tempfile.gettempdir()}/plot{id(node)}.png")
                dot.node(str(id(node)), label="", image=f"{tempfile.gettempdir()}/plot{id(node)}.png", fontsize="30pt")
            else:
                dot.node(str(id(node)), node.representation, fontsize="30pt")
            if node.parent is not None:
                weight = node.get_weight_if_possible()
                if weight is None:
                    dot.edge(str(id(node.parent)), str(id(node)))
                else:
                    dot.node(str(id(weight)), label=str(weight))
                    dot.edge(str(id(weight)), str(id(node)))
                    dot.edge(str(id(node.parent)), str(id(weight)), dir="none")
        return dot
