# samples = pd.read_sql(fpa.query_for_database(), engine)
# samples
#
# variables = infer_variables_from_dataframe(samples, scale_continuous_types=False, min_samples_per_quantile=5)
# model = JPT(variables, min_samples_leaf=100)
# model.fit(samples)
# model = model.probabilistic_circuit
import json
import os
import time
from os import uname
from tkinter import Event

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tqdm
from numpy.core.defchararray import title
from random_events.variable import Continuous, Symbolic

import probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit as pc
import probabilistic_model.Monte_Carlo_Estimator as mc
from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import ProbabilisticCircuit
from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.utils import MissingDict



# samples = pd.read_csv("data.df")
# samples.drop(columns=["Unnamed: 0"], inplace=True)
# variables1 = infer_variables_from_dataframe(samples, scale_continuous_types=False, min_samples_per_quantile=800)
# #print(variables1)
# model1 = JPT(variables1, min_samples_leaf=0.01)
# model1.fit(samples)
# model1 = model1.probabilistic_circuit
# variables = infer_variables_from_dataframe(samples, scale_continuous_types=False, min_samples_per_quantile=600)
# model2 = JPT(variables, min_samples_leaf=0.1)
# model2.fit(samples)
# model2 = model2.probabilistic_circuit
#
import sklearn.datasets

data = sklearn.datasets.load_iris(as_frame=True)
df = data.data

# target = data.target.astype(str)
# target[target == "1"] = "malignant"
# target[target == "0"] = "friendly"



# df["malignant"] = target
variables1 = infer_variables_from_dataframe(df, scale_continuous_types=False, min_samples_per_quantile=100)
variables2 = infer_variables_from_dataframe(df, scale_continuous_types=False, min_samples_per_quantile=20)
print(variables1)
model1 = JPT(variables1, min_samples_leaf=0.05)
model1.fit(df)
model1 = model1.probabilistic_circuit
model2 = JPT(variables2, min_samples_leaf=0.1)
model2.fit(df)
model2 = model2.probabilistic_circuit
print("M1: ", model1)
print("M2: ", model2)
# s = time.time()
# model1.replace_discrete_distribution_with_deterministic_sum()
# model2.replace_discrete_distribution_with_deterministic_sum()
# e = time.time()
# print("replace: ", e-s)
# s = time.time()
# shallow1 = pc.ShallowProbabilisticCircuit.from_probabilistic_circuit(model1)
# shallow2 = pc.ShallowProbabilisticCircuit.from_probabilistic_circuit(model2)
# e = time.time()
# print("shallow: ", e-s)
# # print(model1)
# # print(model2)
mc_l1_list = []
# mc_uniform_model_li = []
size_li = []
mc = mc.MonteCarloEstimator(model1, sample_size=10)
for size in tqdm.trange(20000, 30000,1000):
    mc.set_sample_size(size)
    mc_l1, time_l1 = mc.l1(model2)
    mc_l1_list.append(mc_l1)
    size_li.append(size)
print(mc_l1_list)
plot = go.Figure()
plot.add_trace(go.Scatter(x=size_li, y=mc_l1_list, name="Monte Carlo L1"))
plot.update_layout(xaxis_title="Sample Size", yaxis_title="Digits", title="iris")

plot.show()


exit()

#print("done")
# print("model1")
# with open("model1_support.json", "w") as f:
#     json_sting = json.dumps(model1.support.to_json(), indent=4)
#     f.write(json_sting)
#     f.close()
# print("model2")
# with open("model2_support.json", "w") as f:
#     json_sting = json.dumps(model2.support.to_json(), indent=4)
#     f.write(json_sting)
#     f.close()
# with open("model_union_support.json", "w") as f:
#     start_time = time.time()
#     a = model1.support | model2.support
#     end_time = time.time()
#     print(f"Union Calc Time: {end_time - start_time}")
#     json_sting = json.dumps(a.to_json(), indent=4)
#     f.write(json_sting)
#     f.close()
# exit()

example_data = {}
model_data = {"model1": {"data":model1.to_json(), "string": model1.__str__()}, "model2": {"data":model2.to_json(), "string": model2.__str__()}}
example_data.update(model_data)


shallow_dic = {"replace_time": 0, "shallowing_time": 0}
s = time.time()
model1.replace_discrete_distribution_with_deterministic_sum()
model2.replace_discrete_distribution_with_deterministic_sum()
e = time.time()
shallow_dic.update({"replace_time": e-s})
s = time.time()
shallow1 = pc.ShallowProbabilisticCircuit.from_probabilistic_circuit(model1)
shallow2 = pc.ShallowProbabilisticCircuit.from_probabilistic_circuit(model2)
e = time.time()
shallow_dic.update({"shallowing_time": e-s})
example_data.update(shallow_dic)

#
print("Calculating union of support")
support_dic = {}
s = time.time()
union_of_supports = model1.support | model2.support
e = time.time()
support_dic.update({"union_time": e-s, "union_value": union_of_supports.to_json()})
print("creating uniform model")
s = time.time()
uniform_model = mc.uniform_measure_of_event(union_of_supports)
e = time.time()
support_dic.update({"uniform_time": e-s, "uniform_value": uniform_model.to_json()})
support_model_time = support_dic.get("uniform_time") + support_dic.get("union_time")
example_data.update(support_dic)

# with open("uniform_model.json", "w") as f:
#     json_sting = json.dumps(uniform_model.to_json(), indent=4)
#     f.write(json_sting)
#     f.close()


# uniform_model = ProbabilisticCircuit.from_json...
# uniform_model = ProbabilisticCircuit()
# with open("uniform_model.json", "r") as f:
#     uniform_model = ProbabilisticCircuit.from_json(json.load(f))
#
# print(uniform_model)


# shallow1.root.plot_structure()
# shallow2.root.plot_structure()
#print(shallow1, shallow2)
#print("shallow done")
# print(shallow1.events_of_higher_density_sum(shallow2))
# print(shallow1.events_of_higher_density_cool(shallow2))
# go.Figure(shallow1.plot()).show()
# go.Figure(shallow2.plot()).show()
# avm = shallow1.area_validation_metric(shallow2)
s  = time.time()
l1 = shallow1.l1(shallow2)
e = time.time()
example_data.update({"l1_time": e-s, "l1_value": l1})
# print("-"*80)
#l1 = 0.8431124873874959
# mc_l1 = mc.MonteCarloEstimator(shallow1, sample_size=100000).l1(shallow2)
# mc_uni = mc.MonteCarloEstimator(shallow1, sample_size=100000).l1_metric_but_with_uniform_measure(shallow2)
# print("l1: {}, mc_l1: {}, mc_uni: {}".format(l1, mc_l1, mc_uni))
mc_l1_list = []
mc_uniform_model_li = []
mc_l1_list_time = []
mc_uniform_model_li_time = []
size_li = []
time_l1_li = []
time_uni_li = []
mc = mc.MonteCarloEstimator(shallow1, sample_size=10)
for size in tqdm.trange(100, 300000,100):
    mc.set_sample_size(size)
    s = time.time()
    mc_l1, time_l1 = mc.l1(model2)
    e = time.time()
    mc_l1_list_time.append(e-s)
    s = time.time()
    mc_uniform_model, time_uni = l1_metric_but_with_uniform_measure(model1, model2, size)
    e = time.time()
    mc_uniform_model_li_time.append(e-s)
    mc_l1_list.append(mc_l1)
    mc_uniform_model_li.append(mc_uniform_model)
    size_li.append(size)
    time_l1_li.append(time_l1)
    time_uni_li.append(time_uni)
    #print("sampel: {} l1: {}, mc_l1: {}, mc_unif: {}".format(size, l1, mc_l1, mc_uniform_model))
# traces = model.plot()
# go.Figure(traces).show()
monte_carlo_dic = {"mc_l1": mc_l1_list, "mc_uni": mc_uniform_model_li}
example_data.update(monte_carlo_dic)
print("avg:")
print(f"mc_l1: {np.mean(mc_l1_list)}, mc_uni: {np.mean(mc_uniform_model_li)}")
plot = go.Figure()
plot.add_trace(go.Scatter(x=size_li, y=mc_l1_list, name="Monte Carlo L1"))
plot.add_trace(go.Scatter(x=size_li, y=mc_uniform_model_li, name="Monte Carlo with Uniform"))
plot.add_trace(go.Scatter(x=size_li, y=[l1]*len(size_li), name="L1 distance"))
plot.update_layout(xaxis_title="Sample Size", yaxis_title="Distance", title="Monte Carlo estimation")
plot.write_html("mc_value.html")
plot.show()

plot_time = go.Figure()
plot_time.add_trace(go.Scatter(x=size_li, y=mc_l1_list_time, name="Monte Carlo L1"))
plot_time.add_trace(go.Scatter(x=size_li, y=mc_uniform_model_li_time, name="Monte Carlo with Uniform"))
plot_time.add_trace(go.Scatter(x=size_li, y=[example_data.get("l1_time")]*len(size_li), name="L1 distance"))
plot_time.write_html("mc_time.html")
plot_time.update_layout(xaxis_title="Sample Size", yaxis_title="Seconds", title="")
plot_time.show()

plot_time2 = go.Figure()
plot_time2.add_trace(go.Scatter(x=size_li, y=mc_l1_list_time, name="Monte Carlo L1"))
plot_time2.add_trace(go.Scatter(x=size_li, y=mc_uniform_model_li_time, name="Monte Carlo with Uniform"))
plot_time2.add_trace(go.Scatter(x=size_li, y=list(map(lambda x : x + support_model_time,mc_uniform_model_li_time)), name="Monte Carlo with Uniform+ union"))
plot_time2.add_trace(go.Scatter(x=size_li, y=[example_data.get("l1_time")]*len(size_li), name="L1 distance"))
plot_time2.update_layout(xaxis_title="Sample Size", yaxis_title="Seconds")
plot_time2.write_html("mc_time2.html")
plot_time2.show()

plot_time_split = go.Figure()
plot_time_split.add_trace(go.Scatter(x=size_li, y=mc_l1_list_time, name="Monte Carlo L1"))
plot_time_split.add_trace(go.Scatter(x=size_li, y=mc_uniform_model_li_time, name="Monte Carlo with Uniform"))
plot_time_split.update_layout(xaxis_title="Sample Size", yaxis_title="Seconds")
plot_time_split.write_html("mc_only_time.html")
plot_time_split.show()


with open("example_data.json", "w") as f:
    json_sting = json.dumps(example_data, indent=4)
    f.write(json_sting)
    f.close()



