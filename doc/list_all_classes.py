import sys
from random_events.utils import recursive_subclasses, get_full_class_name
from probabilistic_model.probabilistic_model import *
from probabilistic_model.constants import *
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import *


def list_all_classes():
    for cls in recursive_subclasses(ProbabilisticModel):
        print(get_full_class_name(cls))


list_all_classes()
