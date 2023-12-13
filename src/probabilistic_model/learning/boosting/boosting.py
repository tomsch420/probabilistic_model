from typing import Iterable

import torch
import torch.nn as nn


from ...probabilistic_circuit.units import SmoothSumUnit, DecomposableProductUnit as PCDecomposableProductUnit
from ...probabilistic_circuit.distributions import ContinuousDistribution
from random_events.variables import Continuous, Variable


class NormalDistribution(ContinuousDistribution, torch.distributions.Normal):

    def __init__(self, variable: Continuous, loc: torch.Tensor, scale: torch.Tensor, parent=None):
        super().__init__(variable, parent=parent)
        torch.distributions.Normal.__init__(self, loc, scale)
        self.loc.requires_grad_(True)
        self.scale.requires_grad_(True)

    def log_likelihood(self, data: torch.Tensor) -> torch.Tensor:
        return super().log_prob(data)

    def reset_gradients(self):
        self.loc.grad = None
        self.scale.grad = None


class DecomposableProductUnit(PCDecomposableProductUnit):

    def __init__(self, variables: Iterable[Variable]):
        super().__init__(variables)

    def log_likelihood(self, data: torch.Tensor):
        log_likelihoods_of_children = torch.cat([child.log_likelihood(data) for child in self.children], dim=1)
        return torch.sum(log_likelihoods_of_children, dim=1)

    def reset_gradients(self):
        for child in self.children:
            child.reset_gradients()


class BoostedCircuit(SmoothSumUnit):

    _weights: torch.Tensor

    def __init__(self, variables, number_of_components: int = 1):
        super().__init__(variables, torch.tensor([], requires_grad=True).double())
        self.number_of_components = number_of_components

    @property
    def weights(self) -> torch.Tensor:
        return nn.Softmax(dim=0)(self._weights)

    @weights.setter
    def weights(self, value: torch.Tensor):
        self._weights = value

    def log_likelihood(self, data: torch.Tensor):
        log_likelihoods_of_children = torch.cat([child.log_likelihood(data).unsqueeze(1) for child in self.children],
                                                dim=1)
        likelihoods_of_children = torch.exp(log_likelihoods_of_children) * self.weights
        likelihoods = torch.sum(likelihoods_of_children, dim=1)
        return torch.log(likelihoods)

    def loss(self, data: torch.Tensor):
        return -torch.sum(self.log_likelihood(data))

    def create_model_from_weights_and_data(self, weights: torch.Tensor, data: torch.Tensor) -> DecomposableProductUnit:

        result = DecomposableProductUnit(self.variables)

        for index, variable in enumerate(self.variables):
            if not isinstance(variable, Continuous):
                raise ValueError("Boosting only supports continuous variables.")

            column = data[:, index]
            mean = torch.sum(weights * column) / torch.sum(weights)
            print(mean)
            std = torch.sqrt(torch.sum(weights * (column - mean) ** 2) / torch.sum(weights))
            print(std)
            distribution = NormalDistribution(variable, mean, std, parent=result)

        return result

    def reset_gradients(self):
        self.weights.grad = None
        for child in self.children:
            child.reset_gradients()

    def fit(self, data: torch.Tensor):
        data = data.requires_grad_(True)
        data.retain_grad()
        sample_weights = torch.full((len(data), ), 1/len(data)).double()
        model = self.create_model_from_weights_and_data(sample_weights, data)

        initial_weight = torch.sum(model.log_likelihood(data)).unsqueeze(0)
        model.parent = self
        self.weights = torch.tensor([initial_weight], requires_grad=True)

        for _ in range(self.number_of_components - 1):
            log_likelihoods = self.log_likelihood(data)
            log_likelihoods.retain_grad()
            loss = -torch.sum(log_likelihoods)
            loss.retain_grad()
            loss.backward(retain_graph=True)
            weights = data.grad
            normalized_weights = nn.Softmax(dim=0)(weights)
            # print(normalized_weights)
            # print(normalized_weights)
            model = self.create_model_from_weights_and_data(normalized_weights, data)

            weight_of_new_mixture_component = torch.sum(model.log_likelihood(data)).unsqueeze(0)
            print("weight", weight_of_new_mixture_component)
            model.parent = self
            print("sle.fweights", self.weights)
            self.weights = torch.cat((self.weights, torch.tensor(weight_of_new_mixture_component, requires_grad=True)))

            self.reset_gradients()
            data.grad = None
