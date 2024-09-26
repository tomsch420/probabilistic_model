# samples = pd.read_sql(fpa.query_for_database(), engine)
# samples
#
# variables = infer_variables_from_dataframe(samples, scale_continuous_types=False, min_samples_per_quantile=5)
# model = JPT(variables, min_samples_leaf=100)
# model.fit(samples)
# model = model.probabilistic_circuit
import os
from os import uname
from tkinter import Event

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tqdm
from numpy.core.defchararray import title

import probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit as pc
import probabilistic_model.Monte_Carlo_Estimator as mc
from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe

samples = pd.read_csv("data.df")
samples.drop(columns=["Unnamed: 0"], inplace=True)
variables1 = infer_variables_from_dataframe(samples, scale_continuous_types=False, min_samples_per_quantile=800)
#print(variables1)
model1 = JPT(variables1, min_samples_leaf=0.01)
model1.fit(samples)
model1 = model1.probabilistic_circuit
variables = infer_variables_from_dataframe(samples, scale_continuous_types=False, min_samples_per_quantile=600)
model2 = JPT(variables, min_samples_leaf=0.1)
model2.fit(samples)
model2 = model2.probabilistic_circuit
#print("done")
model1.replace_discrete_distribution_with_deterministic_sum()
model2.replace_discrete_distribution_with_deterministic_sum()

shallow1 = pc.ShallowProbabilisticCircuit.from_probabilistic_circuit(model1)
shallow2 = pc.ShallowProbabilisticCircuit.from_probabilistic_circuit(model2)

# shallow1.root.plot_structure()
# shallow2.root.plot_structure()
#print(shallow1, shallow2)
#print("shallow done")
# print(shallow1.events_of_higher_density_sum(shallow2))
# print(shallow1.events_of_higher_density_cool(shallow2))
# go.Figure(shallow1.plot()).show()
# go.Figure(shallow2.plot()).show()
# avm = shallow1.area_validation_metric(shallow2)
#l1 = shallow1.l1_swag(shallow2)
l1 = 0.6461379536351356
# mc_l1 = mc.MonteCarloEstimator(shallow1, sample_size=100000).l1(shallow2)
# mc_uni = mc.MonteCarloEstimator(shallow1, sample_size=100000).l1_metric_but_with_uniform_measure(shallow2)
# print("l1: {}, mc_l1: {}, mc_uni: {}".format(l1, mc_l1, mc_uni))
mc_l1_list = []
mc_avm_list = []
size_li = []
mc = mc.MonteCarloEstimator(shallow1, sample_size=10)
for size in tqdm.trange(1000, 10000,500):
    mc.set_sample_size(size)
    mc_l1 = mc.l1(model2)
    mc_avm = mc.area_validation_metric2(model2)

    mc_l1_list.append(mc_l1)
    mc_avm_list.append(mc_avm)
    size_li.append(size)
    mc_true= (mc_l1 + mc_avm)/2
    print("sampel: {} l1: {}, mc_l1: {}, mc_avm: {}, mc_true {}".format(size, l1, mc_l1, mc_avm, mc_true))
# traces = model.plot()
# go.Figure(traces).show()
print("avg:")
print(f"mc_l1: {np.mean(mc_l1_list)}, mc_uni: {np.mean(mc_avm_list)}, true_avg: {np.mean([(a+b)/2 for a, b in zip(mc_l1_list, mc_avm_list)])}")
plot = go.Figure()
plot.add_trace(go.Scatter(x=size_li, y=mc_l1_list))
plot.add_trace(go.Scatter(x=size_li, y=mc_avm_list))
plot.add_trace(go.Scatter(x=size_li, y=[l1]*len(size_li)))
plot.show()