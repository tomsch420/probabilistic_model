######################
Probabilistic Circuits
######################

Probabilistic Circuits (PCs) define a framework for tractable, probabilistic inference.
You can read about PCs in detail here :cite:p:`choi2020probabilistic`.

This package provides the following datastructures for PCs:

-  :class:`probabilistic_model.probabilistic_circuit.probabilistic_circuit.DecomposableProductUnit`
-  :class:`probabilistic_model.probabilistic_circuit.probabilistic_circuit.SmoothSumUnit`
-  :class:`probabilistic_model.probabilistic_circuit.probabilistic_circuit.DeterministicSumUnit`


Datastructures in the Module
============================
This section will run through the datastructures implemented for PCs that are necessary to fully understand
manipulation and functionality of circuits.


Probabilistic Circuit
************************************************************************************************

.. autoapi-inheritance-diagram:: probabilistic_model.probabilistic_circuit.probabilistic_circuit.ProbabilisticCircuit
    :parts: 1

The most important aspect of the
:class:`probabilistic_model.probabilistic_circuit.probabilistic_circuit.ProbabilisticCircuit` class, is the inheritance
from `networkx.DiGraph <https://networkx.org/documentation/stable/reference/classes/digraph.html>`_ .
A PC in this sense is a directed acyclic graph (DAG) where the nodes are computational units
or distributions. The edges represent how the nodes are combined to form the general model.

The nodes that can be added to a PC have to inherit from
:class:`probabilistic_model.probabilistic_circuit.probabilistic_circuit.ProbabilisticCircuitMixin` such that they
correctly work with inference algorithms.

Additionally, PCs inherit from probabilistic models. Since undirected cycles inside a circuit are possible, it is also
possible that some sub-circuit has to be evaluated multiple times. Inference methods from this class cache the results
at every node, such that results are not calculated multiple times. The caches are cleared after each inference run.
See the :class:`probabilistic_model.probabilistic_circuit.probabilistic_circuit.graph_inference_caching_wrapper` and
:class:`probabilistic_model.probabilistic_circuit.probabilistic_circuit.cache_inference_result` decorators for more
information.


Probabilistic Circuit Mixin
***************************************************************************************************

.. autoapi-inheritance-diagram:: probabilistic_model.probabilistic_circuit.probabilistic_circuit.ProbabilisticCircuitMixin
    :parts: 1


This class serves as a `mixin class <https://en.wikipedia.org/wiki/Mixin>`_ for components that can be used in a
:class:`probabilistic_model.probabilistic_circuit.probabilistic_circuit.ProbabilisticCircuit`

Nodes inside a PC have to inherit from
:class:`probabilistic_model.probabilistic_circuit.probabilistic_circuit.ProbabilisticCircuitMixin` such that they work
as intended with PCs. Besides being an abstract specialization of a Probabilistic Model, it is important that the
hash method of such a component refers to the objects id. NetworkX uses the hashes of objects as so to speak pointers
in their graphs. Since PCs can certainly contain components that could be seen as equal but yet have to exist multiple
times, there is, up to my knowledge, no better way of defining the hash.


Decomposable Product Unit
*************************

.. autoapi-inheritance-diagram:: probabilistic_model.probabilistic_circuit.probabilistic_circuit.DecomposableProductUnit
    :parts: 1

:class:`probabilistic_model.probabilistic_circuit.probabilistic_circuit.DecomposableProductUnit` represent, as the name
suggests, a decomposable product unit.
Edges that have instances of this class as a source must not be weighted. Besides that, there is nothing special about
them.

Smooth and Deterministic Sum Units
***************

.. autoapi-inheritance-diagram:: probabilistic_model.probabilistic_circuit.probabilistic_circuit.DeterministicSumUnit
    :parts: 1

:class:`probabilistic_model.probabilistic_circuit.probabilistic_circuit.SmoothSumUnit` and
:class:`probabilistic_model.probabilistic_circuit.probabilistic_circuit.DeterministicSumUnit` represent smooth and
deterministic summation operations just as described in the theory behind it. Edges that have these as source, must
be weighted.

A notable addition to circuits as described by  :cite:p:`choi2020probabilistic` is the
:meth:`probabilistic_model.probabilistic_circuit.probabilistic_circuit.SmoothSumUnit.mount_with_interaction_terms`
method.

