import os.path
import tempfile
import unittest

from probabilistic_model.interfaces.drawio.exporter import DrawIoExporter
from random_events.interval import SimpleInterval
from random_events.variable import Continuous
from random_events.variable import Variable
from typing_extensions import List

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.nx.distributions import UnivariateContinuousLeaf
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import SumUnit, ProductUnit, \
    ProbabilisticCircuit

class DrawIOExporterTestCase(unittest.TestCase):
    variables: List[Variable]

    x = Continuous("x")
    y = Continuous("y")
    sum1, sum2, sum3 = SumUnit(), SumUnit(), SumUnit()
    sum4, sum5 = SumUnit(), SumUnit()
    prod1, prod2 = ProductUnit(), ProductUnit()

    sum1.add_subcircuit(prod1, 0.5)
    sum1.add_subcircuit(prod2, 0.5)
    prod1.add_subcircuit(sum2)
    prod1.add_subcircuit(sum4)
    prod2.add_subcircuit(sum3)
    prod2.add_subcircuit(sum5)

    d_x1 = UnivariateContinuousLeaf(UniformDistribution(x, SimpleInterval(0, 1)))
    d_x2 = UnivariateContinuousLeaf(UniformDistribution(x, SimpleInterval(2, 3)))
    d_y1 = UnivariateContinuousLeaf(UniformDistribution(y, SimpleInterval(0, 1)))
    d_y2 = UnivariateContinuousLeaf(UniformDistribution(y, SimpleInterval(3, 4)))

    sum2.add_subcircuit(d_x1, 0.8)
    sum2.add_subcircuit(d_x2, 0.2)
    sum3.add_subcircuit(d_x1, 0.7)
    sum3.add_subcircuit(d_x2, 0.3)

    sum4.add_subcircuit(d_y1, 0.5)
    sum4.add_subcircuit(d_y2, 0.5)
    sum5.add_subcircuit(d_y1, 0.1)
    sum5.add_subcircuit(d_y2, 0.9)

    model = sum1.probabilistic_circuit

    def test_export_to_drawio(self):
        diagram = DrawIoExporter(self.model.root.probabilistic_circuit).export()
        with tempfile.NamedTemporaryFile(suffix=".drawio", delete=False) as temp_file:
            temp_file_name = temp_file.name

        # temp_file_name = os.path.join(os.path.expanduser("~"), "Documents", "test.drawio")

        # Write the diagram data to the temporary file
        with open(temp_file_name, "w") as f:
            f.write(diagram.dump_xml())
        print(f"Diagram exported to temporary file: {temp_file_name}")


if __name__ == '__main__':
    unittest.main()
