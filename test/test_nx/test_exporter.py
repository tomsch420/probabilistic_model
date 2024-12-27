import tempfile
import unittest

from probabilistic_model.interfaces.drawio.exporter import DrawIoExporter
from random_events.interval import SimpleInterval
from random_events.variable import Continuous
from random_events.variable import Variable
from typing_extensions import List

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import SumUnit, ProductUnit, \
    ProbabilisticCircuit

class DrawIOExporterTestCase(unittest.TestCase):
    variables: List[Variable]

    x = Continuous("x")
    y = Continuous("y")
    sum1, sum2, sum3 = SumUnit(), SumUnit(), SumUnit()
    sum4, sum5 = SumUnit(), SumUnit()
    prod1, prod2 = ProductUnit(), ProductUnit()
    model = ProbabilisticCircuit()
    model.add_node(sum1)
    model.add_node(prod1)
    model.add_node(prod2)
    model.add_edge(sum1, prod1, weight=0.5)
    model.add_edge(sum1, prod2, weight=0.5)
    model.add_node(sum2)
    model.add_node(sum3)
    model.add_node(sum4)
    model.add_node(sum5)
    model.add_edge(prod1, sum2)
    model.add_edge(prod1, sum4)
    model.add_edge(prod2, sum3)
    model.add_edge(prod2, sum5)
    uni_x1, uni_x2 = UniformDistribution(x, SimpleInterval(0, 1)), UniformDistribution(x, SimpleInterval(1, 2))
    uni_y1, uni_y2 = UniformDistribution(y, SimpleInterval(0, 1)), UniformDistribution(y, SimpleInterval(1, 2))

    model.add_node(uni_y1)
    model.add_node(uni_x2)
    model.add_node(uni_y2)
    model.add_node(uni_x1)

    model.add_edge(sum2, uni_x1, weight=0.8)
    model.add_edge(sum2, uni_x2, weight=0.2)
    model.add_edge(sum3, uni_x1, weight=0.7)
    model.add_edge(sum3, uni_x2, weight=0.3)

    model.add_edge(sum4, uni_y1, weight=0.5)
    model.add_edge(sum4, uni_y2, weight=0.5)
    model.add_edge(sum5, uni_y1, weight=0.1)
    model.add_edge(sum5, uni_y2, weight=0.9)

    def test_export_to_drawio(self):
        diagram = DrawIoExporter(self.model.root.probabilistic_circuit).export()
        with tempfile.NamedTemporaryFile(suffix=".drawio", delete=True) as temp_file:
            temp_file_name = temp_file.name

        # Write the diagram data to the temporary file
        diagram.dump_file(temp_file_name)
        print(f"Diagram exported to temporary file: {temp_file_name}")


if __name__ == '__main__':
    unittest.main()
