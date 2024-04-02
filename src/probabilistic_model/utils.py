from typing_extensions import Dict, Any, Self, TYPE_CHECKING, Type
from random_events.utils import get_full_class_name, recursive_subclasses
import types


if TYPE_CHECKING:
    from .distributions.distributions import UnivariateDistribution


def type_converter(abstract_type: Type, package: types.ModuleType):
    """
    Convert a type to a different type from a target sub-package that inherits from this type.

    :param abstract_type: The type to convert
    :param package: The sub-package to search in for that type

    :return: The converted type
    """
    for subclass in recursive_subclasses(abstract_type):
        if subclass.__module__.startswith(package.__name__):
            return subclass

    raise ValueError("Could not find type {} in package {}".format(abstract_type, package))


class SubclassJSONSerializer:
    """
    Class for automatic (de)serialization of subclasses.
    Classes that inherit from this class can be serialized and deserialized automatically by calling this classes
    'from_json' method.
    """

    def to_json(self) -> Dict[str, Any]:
        return {"type": get_full_class_name(self.__class__)}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create a variable from a json dict.
        This method is called from the from_json method after the correct subclass is determined and should be
        overwritten by the respective subclass.

        :param data: The json dict
        :return: The deserialized object
        """
        raise NotImplementedError()

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create the correct instanceof the subclass from a json dict.

        :param data: The json dict
        :return: The correct instance of the subclass
        """
        for subclass in recursive_subclasses(SubclassJSONSerializer):
            if get_full_class_name(subclass) == data["type"]:
                return subclass._from_json(data)

        raise ValueError("Unknown type {}".format(data["type"]))
