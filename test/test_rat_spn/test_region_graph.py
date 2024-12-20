import random
import unittest

from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from random_events.variable import Continuous
import pydot
from probabilistic_model.learning.region_graph.region_graph import *
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import ProbabilisticCircuit as JPC
from probabilistic_model.probabilistic_circuit.jax.gaussian_layer import GaussianLayer
import numpy as np

np.random.seed(420)
random.seed(420)

class RandomRegionGraphTestCase(unittest.TestCase):

    variables = SortedSet([Continuous(str(i)) for i in range(4)])

    region_graph = RegionGraph(variables, partitions=2, depth=1, repetitions=2)
    region_graph = region_graph.create_random_region_graph()

    def test_region_graph(self):
        self.assertEqual(len(self.region_graph.nodes()), len(self.region_graph.nodes()))

    def test_as_pc(self):
        model = self.region_graph.as_probabilistic_circuit(input_units=1, sum_units=1)
        model.plot_structure()
        plt.show()
        jax_model = JPC.from_nx(model)

    def test_as_jpc(self):
        model = self.region_graph.as_jax_pc(input_units=10, sum_units=5)
        print(model)
        nx_model = model.to_nx()
        nx_model.plot_structure()
        plt.show()


if __name__ == '__main__':
    unittest.main()
