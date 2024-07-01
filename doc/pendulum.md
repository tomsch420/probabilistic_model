---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# The Pendulum between Machine Learning and Knowledge Graphs

In March 2024 [Frank van Harmelen](https://www.cs.vu.nl/~frankh/) visited the [Institute for Artificial Intelligence](https://ai.uni-bremen.de/) 
(It is where I work).

Frank talked about the duality of research in artificial intelligence using the metaphor of a pendulum. 
The pendulum swings between the extremes of having purely machine learning enabled AI and 
purely knowledge graph driven AI.
A full swing takes a couple of years.
Frank argued that there needs to be a middle ground where machine learning and knowledge graphs come together.

The implementation of probabilistic models in this package is capable of doing so. 
Knowledge graphs generate sets that describe possible assignments that match the constraints and instance knowledge of ontologies. 
Probability distributions describe the likelihoods of every possible solution. 
Combining these in an efficient way provides a framework that is capable of putting the pendulum to rest.

The integration of ontological reasoning into probabilistic models is currently researched by 
[Dr. Mihai Pomarlan](https://ai.uni-bremen.de/team/mihai_pomarlan) and 
[me](https://ai.uni-bremen.de/team/tom_schierenbeck).

The integration of neural techniques into probabilistic models is an ongoing research and implementation by me.
However, {cite}`martires2024probabilistic` wonderfully explains a very similar approach.
