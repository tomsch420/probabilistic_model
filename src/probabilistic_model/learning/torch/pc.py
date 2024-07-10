from abc import abstractmethod, ABC

import torch
import torch.nn as nn
from random_events.variable import Continuous, Variable
from torch.nn import ModuleList
from typing_extensions import List


class Layer(nn.Module):
    """
    Abstract class for Layers of a layered circuit
    """

    @property
    def number_of_nodes(self) -> int:
        """
        The number of nodes in the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log-likelihood of the distribution.

        .. note::
            The shape of the log likelihood depends on the number of samples and nodes.
            The shape of the result is (#samples, #nodes).
            The first dimension indexes the samples, the second the nodes.
        """
        raise NotImplementedError


class InnerLayer(Layer, ABC):
    """
    Class for inner layers
    """

    child_layers: List[Layer]
    """
    The child layers of this layer.
    """


class InputLayer(Layer, ABC):
    """
    Abstract base class for input units.
    Input layers should contain only one type of distribution such that the vectorization of the log likelihood
    calculation works without bottleneck statements like if/else or loops.
    """

    variable: Variable
    """
    The variable of the distributions.
    """

class UniformLayer(InputLayer):
    """
    A layer that represents a uniform distribution over a single variable.
    """

    variable: Continuous
    """
    The index of the variable that this layer represents.
    """

    lower: torch.Tensor
    """
    The lower bounds of the uniform distributions as a tensor of shape (self.number_of_nodes, ).
    The lower bounds are treated as open intervals.
    """

    upper: torch.Tensor
    """
    The upper bound of the uniform distributions as a tensor of shape (self.number_of_nodes, ).
    The upper bounds are treated as open intervals.
    """

    def __init__(self, variable: Continuous, lower: torch.Tensor, upper: torch.Tensor):
        """
        Initialize the uniform layer.

        :param variable: The variable that this layer represents.
        :param lower: The lower bounds of the uniform distributions.
        :param upper: The upper bounds of the uniform distributions.
        """
        super().__init__()
        self.variable = variable
        assert lower.shape == upper.shape, "The shapes of the lower and upper bounds must match."
        self.lower = lower
        self.upper = upper

    @property
    def number_of_nodes(self) -> int:
        """
        The number of nodes in the layer.
        """
        return len(self.lower)

    def log_pdf_value(self) -> torch.Tensor:
        """
        Calculate the log-density of the uniform distribution.
        """
        return -torch.log(self.upper - self.lower)

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log-likelihood of the uniform distribution.

        :param x: The input tensor of shape (n, 1).
        :return: The log-likelihood of the uniform distribution.
        """
        return torch.where((x > self.lower) & (x < self.upper), self.log_pdf_value(), -torch.inf)


class SumLayer(InnerLayer):
    """
    A layer that represents the sum of multiple other layers.
    """

    log_weights: List[torch.Tensor]
    """
    The logarithmic weights of each edge.
    The list consists of tensor that are interpreted as weights for each child layer.
    
    The first dimension of each tensor must match the number of nodes of this layer and hence has to be 
    constant.
    The second dimension of each tensor must match the number of nodes of the  respective child layer.

    The weights are normalized per row.
    """

    def __init__(self, child_layers: List[nn.Module], log_weights: List[torch.Tensor]):
        """
        Initialize the sum layer.

        :param child_layers: The child layers of the sum layer.
        :param log_weights: The logarithmic weights of each edge.
        """
        super().__init__()
        self.child_layers = ModuleList(child_layers)
        self.log_weights = log_weights

    def validate(self):
        """
        Check that the weight shapes are valid.
        """
        for log_weights in self.log_weights:
            assert log_weights.shape[0] == self.number_of_nodes, "The number of nodes must match the number of weights."

        for log_weights, child_layer in self.log_weighted_child_layers:
            assert log_weights.shape[1] == child_layer.number_of_nodes, \
                "The number of nodes must match the number of weights."

    @property
    def log_weighted_child_layers(self):
        yield from zip(self.log_weights, self.child_layers)

    @property
    def concatenated_weights(self) -> torch.Tensor:
        return torch.cat(self.log_weights, dim=1)

    @property
    def log_normalization_constants(self) -> torch.Tensor:
        """
        :return: The normalization constants for every node in this sum layer.
        """
        return torch.logsumexp(self.concatenated_weights, dim=1)

    @property
    def number_of_nodes(self) -> int:
        """
        The number of nodes in the layer.
        """
        return self.log_weights[0].shape[0]

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(len(x), self.number_of_nodes)

        for log_weights, child_layer in self.log_weighted_child_layers:

            # get the log likelihoods of the child nodes
            ll = child_layer.log_likelihood(x)
            # assert ll.shape == (len(x), child_layer.number_of_nodes)

            # expand the log likelihood of the child nodes to the number of nodes in this layer, i.e.
            # (#x, #child_nodes, #nodes)
            ll = ll.unsqueeze(-1).repeat(1, 1, self.number_of_nodes)
            # assert ll.shape == (len(x), child_layer.number_of_nodes, self.number_of_nodes)

            # weight the log likelihood of the child nodes by the weight for each node of this layer
            ll = torch.exp(ll + log_weights.T).sum(dim=1)

            # sum the child layer result
            result += ll

        return torch.log(result) - self.log_normalization_constants.repeat(len(x), 1)

