
import unittest

from typing_extensions import List
import tempfile
from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from sklearn.datasets import load_breast_cancer
import pandas as pd
from random_events.variable import Variable

from probabilistic_model.probabilistic_circuit.exporter.draw_io_expoter import DrawIoExporter



class MyTestCase(unittest.TestCase):

    dataset: pd.DataFrame
    variables: List[Variable]
    # model: JPT

    from random_events.interval import closed, SimpleInterval
    from random_events.set import SetElement
    from random_events.variable import Continuous

    from probabilistic_model.distributions.uniform import UniformDistribution
    from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import SumUnit, ProductUnit, SimpleEvent, ShallowProbabilisticCircuit, ProbabilisticCircuit
    # from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import ProbabilisticCircuit, SumUnit, ProductUnit, SimpleEvent, ShallowProbabilisticCircuit
    from probabilistic_model.probabilistic_circuit.nx.distributions import UnivariateContinuousLeaf
    import plotly.graph_objects as go
    from probabilistic_model.monte_carlo_estimator import MonteCarloEstimator

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
    uni_x1, uni_x2  = UniformDistribution(x, SimpleInterval(0, 1)), UniformDistribution(x, SimpleInterval(1, 2))
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


    # @classmethod
    # def setUpClass(cls):
    #     data = load_breast_cancer(as_frame=True)
    #
    #     df = data.data
    #     target = data.target
    #     target[target == 1] = "malignant"
    #     target[target == 0] = "friendly"
    #     df["malignant"] = target
    #     cls.dataset = df
    #
    #     variables = infer_variables_from_dataframe(df, min_likelihood_improvement=1)
    #
    #     model = JPT(variables, min_samples_leaf=0.9)
    #     model.fit(df)
    #     cls.model = model

    def test_export_to_drawio(self):
        diagram = DrawIoExporter(self.model.root.probabilistic_circuit).export()
        with tempfile.NamedTemporaryFile(suffix=".drawio", delete=False) as temp_file:
            temp_file_name = temp_file.name

        # Write the diagram data to the temporary file
        diagram.dump_file(temp_file_name)
        print(f"Diagram exported to temporary file: {temp_file_name}")

if __name__ == '__main__':
    unittest.main()



