# samples = pd.read_sql(fpa.query_for_database(), engine)
# samples
#
# variables = infer_variables_from_dataframe(samples, scale_continuous_types=False, min_samples_per_quantile=5)
# model = JPT(variables, min_samples_leaf=100)
# model.fit(samples)
# model = model.probabilistic_circuit
import os
from os import uname
import pandas as pd
import plotly.graph_objects as go

import probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit as pc
import probabilistic_model.Monte_Carlo_Estimator as mc
from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe

samples = pd.read_csv("data.df")
samples.drop(columns=["Unnamed: 0"], inplace=True)
variables1 = infer_variables_from_dataframe(samples, scale_continuous_types=False, min_samples_per_quantile=500)
#print(variables1)
model1 = JPT(variables1, min_samples_leaf=0.1)
model1.fit(samples)
model1 = model1.probabilistic_circuit
variables = infer_variables_from_dataframe(samples, scale_continuous_types=False, min_samples_per_quantile=600)
model2 = JPT(variables, min_samples_leaf=0.1)
model2.fit(samples)
model2 = model2.probabilistic_circuit
#print("done")

shallow1 = pc.ShallowProbabilisticCircuit.from_probabilistic_circuit(model1)
shallow2 = pc.ShallowProbabilisticCircuit.from_probabilistic_circuit(model2)
# shallow1.root.plot_structure()
# shallow2.root.plot_structure()
#print(shallow1, shallow2)
#print("shallow done")
avm = shallow1.area_validation_metric(shallow2)
# print("avm: {}".format(avm))
mc_avm = mc.MonteCarloEstimator(model1, sample_size=100000).area_validation_metric2(model2)
#print(mc_avm)
print(f"shallwo_avm: {avm}, mc_avm: {mc_avm}, diff: {avm - mc_avm}")
# traces = model.plot()
# go.Figure(traces).show()
