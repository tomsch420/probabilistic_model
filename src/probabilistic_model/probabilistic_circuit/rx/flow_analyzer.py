from collections import defaultdict
import numpy as np
import tqdm
from .probabilistic_circuit import ProbabilisticCircuit, Unit, SumUnit, ProductUnit

class NodeLikelihoods:
    """
    Registry for node likelihood values.
    """
    
    def __init__(self):
        self._likelihoods = defaultdict(float)
    
    def set(self, node: Unit, likelihood: float):
        """
        Set the likelihood value for a node.
        """
        self._likelihoods[node] = likelihood
    
    def get(self, node: Unit, default: float = 0.0) -> float:
        """
        Get the likelihood value for a node.
        """
        return self._likelihoods.get(node, default)
    
    def has(self, node: Unit) -> bool:
        """
        Check if a node has a likelihood value.
        """
        return node in self._likelihoods

class NodeFlows:
    """
    Registry for node flow values.
    """
    
    def __init__(self, root_node: Unit):
        self._flows = {root_node: 1.0}
    
    def set(self, node: Unit, flow: float):
        """
        Set the flow value for a node.
        """
        self._flows[node] = flow
    
    def get(self, node: Unit, default: float = 0.0) -> float:
        """
        Get the flow value for a node.
        """
        return self._flows.get(node, default)
    
    def __getitem__(self, node: Unit) -> float:
        """
        Allow dictionary-style access.
        """
        return self._flows[node]
    
    def __setitem__(self, node: Unit, flow: float):
        """
        Allow dictionary-style assignment.
        """
        self._flows[node] = flow

class EdgeFlows:
    """
    Registry for edge flow values.
    """
    
    def __init__(self):
        self._flows = defaultdict(float)
    
    def add(self, parent: Unit, child: Unit, flow: float):
        """
        Add flow value to an edge.
        """
        self._flows[(parent, child)] += flow
    
    def get(self, parent: Unit, child: Unit, default: float = 0.0) -> float:
        """
        Get the flow value for an edge.
        """
        return self._flows.get((parent, child), default)

class CircuitFlowAnalyzer:
    """
    A class for analyzing information flow through probabilistic circuits.
    This class computes edge flows by performing forward and backward passes through the circuit.
    """
    
    def __init__(self, circuit: ProbabilisticCircuit):
        """
        Initialize the flow analyzer with a probabilistic circuit.
        
        :param circuit: The probabilistic circuit to analyze.
        """
        self.circuit = circuit
        self.flow_likelihoods = NodeLikelihoods()
    
    def _forward_pass(self, x: np.ndarray):
        """
        Perform forward pass through the circuit.
        
        :param x: Input sample(s).
        """
        self.circuit.log_likelihood(x)
        
        # Convert log-likelihoods to likelihoods for flow computation
        for layer in reversed(self.circuit.layers):
            for node in layer:
                if node.result_of_current_query is not None:
                    if isinstance(node.result_of_current_query, np.ndarray):
                        self.flow_likelihoods.set(node, np.exp(node.result_of_current_query[0]))
                    else:
                        self.flow_likelihoods.set(node, np.exp(node.result_of_current_query))
    
    def _backward_pass(self) -> NodeFlows:
        """
        Perform backward pass to compute node flows.
        
        :return: Node flow values.
        """        
        node_flows = NodeFlows(self.circuit.root)
        
        for layer in self.circuit.layers:
            for node in layer:
                if node.index == self.circuit.root.index:
                    continue
                    
                node_flows.set(node, 0.0)
                parents = self.circuit.predecessors(node)
                
                for parent in parents:
                    if isinstance(parent, SumUnit):
                        node_flows[node] += node_flows[parent]
                    elif isinstance(parent, ProductUnit):
                        if self.flow_likelihoods.get(parent) > 0:
                            contribution = self.flow_likelihoods.get(node, 0) / self.flow_likelihoods.get(parent, 1)
                            node_flows[node] += contribution * node_flows[parent]

        return node_flows

    def _accumulate_edge_flows(self, edge_flows: EdgeFlows, node_flows: NodeFlows):
        """
        Accumulate edge flows based on node flows and circuit structure.
        
        :param edge_flows: Storage to accumulate edge flows.
        :param node_flows: Node flows from backward pass.
        """        
        for layer in self.circuit.layers:
            for node in layer:
                if isinstance(node, SumUnit):
                    for child in node.subcircuits:
                        if self.flow_likelihoods.get(node, 0) > 0:
                            # Get weight from the edge between parent and child
                            weight = self.circuit.graph.get_edge_data(node.index, child.index)
                            if weight is None:
                                weight = 0.0

                            if self.flow_likelihoods.has(child):
                                edge_flow = (np.exp(weight) * self.flow_likelihoods.get(child, 0) / self.flow_likelihoods.get(node, 1) * node_flows[node])
                                edge_flows.add(node, child, edge_flow)

    def compute_flows(self, dataset: np.ndarray) -> EdgeFlows:
        """
        Compute the flow of information through the circuit for a given dataset.
        
        :param dataset: The input dataset.
        :return: Edge flow values.
        """
        edge_flows = EdgeFlows()

        for x in tqdm.tqdm(dataset, desc="Computing circuit flows"):
            # Ensure x is 2D (reshape single sample to batch of size 1)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            self._forward_pass(x)
            node_flows = self._backward_pass()
            self._accumulate_edge_flows(edge_flows, node_flows)

        return edge_flows
    
    def prune(self, dataset: np.ndarray, pruning_percentage: float) -> ProbabilisticCircuit:
        """
        Prune the circuit based on the computed edge flows and a pruning percentage.

        :param dataset: The input dataset.
        :param pruning_percentage: The percentage of edges to prune.
        :return: The pruned probabilistic circuit.
        """
        edge_flows = self.compute_flows(dataset)
        edge_list = []
        for layer in self.circuit.layers:
            for node in layer:
                if isinstance(node, SumUnit):
                    for child in node.subcircuits:
                        edge_list.append((node, child, edge_flows.get(node, child, 0.0)))

        edge_list.sort(key=lambda x: x[2])
        num_edges_to_prune = int(pruning_percentage * len(edge_list))
        edges_to_prune = edge_list[:num_edges_to_prune]

        pruned_root = self.circuit.root
        for parent, child, _ in edges_to_prune:
            self.circuit.remove_edge(parent, child)

        self.circuit.remove_unreachable_nodes(pruned_root)
        self.circuit.normalize()
        return self.circuit
