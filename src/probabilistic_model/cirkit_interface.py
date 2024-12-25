import cirkit
from cirkit.backend.base import Circuit
from cirkit.backend.torch.circuits import TorchCircuit, TorchLayer
from cirkit.backend.torch.layers import TorchCategoricalLayer
from random_events.variable import Variable
from typing_extensions import List

from .probabilistic_circuit.nx.probabilistic_circuit import ProbabilisticCircuit


class CirkitImporter:

    cirkit_model: TorchCircuit
    variables: List[Variable]

    def __init__(self, cirkit_model: TorchCircuit,  variables: List[Variable]):
        self.cirkit_model = cirkit_model
        self.variables = variables


    def handle_layer(self, layer: TorchLayer):
        if isinstance(layer, TorchCategoricalLayer):
            for scope_index in layer.scope_idx:
                ...


    def to_nx(self,) -> ProbabilisticCircuit:
        """
        Convert the model to a networkx probabilistic circuit.

        :return: The model as a networkx probabilistic circuit.
        """
        nx_model = ProbabilisticCircuit()

        for level in self.cirkit_model.layerwise_topological_ordering():
            for layer in level:
                self.handle_layer(layer)




        return nx_model