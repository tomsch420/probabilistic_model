import unittest

from matplotlib import pyplot as plt
from random_events.interval import *
from random_events.variable import Integer, Continuous

from probabilistic_model.cirkit_interface import CirkitImporter
from probabilistic_model.probabilistic_circuit.nx.distributions import UnivariateContinuousLeaf, UnivariateDiscreteLeaf
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import LeafUnit
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.distributions.distributions import SymbolicDistribution, IntegerDistribution, \
    DiscreteDistribution
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import *
from probabilistic_model.utils import MissingDict

from cirkit.templates import data_modalities, utils

import random
import numpy as np
import torch

from cirkit.pipeline import compile



class CirkitConverterTestCase(unittest.TestCase):

    importer: CirkitImporter

    @classmethod
    def setUpClass(cls) -> None:
        # Set some seeds
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        variables = [Continuous(f"x{i}") for i in range(15)]

        symbolic_circuit = data_modalities.image_data(
            (1, 3, 5),  # The shape of MNIST image, i.e., (num_channels, image_height, image_width)
            region_graph='quad-graph',  # Select the structure of the circuit to follow the QuadGraph region graph
            input_layer='categorical',  # Use Categorical distributions for the pixel values (0-255) as input layers
            num_input_units=20,  # Each input layer consists of 64 Categorical input units
            sum_product_layer='cp',
            # Use CP sum-product layers, i.e., alternate dense layers with Hadamard product layers
            num_sum_units=30,  # Each dense sum layer consists of 64 sum units
            sum_weight_param=utils.Parameterization(
                activation='softmax',  # Parameterize the sum weights by using a softmax activation
                initialization='normal'  # Initialize the sum weights by sampling from a standard normal distribution
            )
        )

        # Set the torch device to use
        device = torch.device('cpu')
        circuit = compile(symbolic_circuit)
        cls.importer = CirkitImporter(circuit, variables)

    def test_to_nx(self):

        result = self.importer.to_nx()