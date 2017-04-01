"""
Abstract base class for objects that are configurable using
a parameters dictionary.
"""

import abc
import copy
from pydoc import locate
from abc import ABCMeta, abstractmethod, abstractproperty
import yaml
import tensorflow as tf


def _create_from_dict(dict_, default_module, *args, **kwargs):
    """Creates a configurable class from a dictionary. The dictionary must have
    "class" and "params" properties. The class can be either fully qualified, or
    it is looked up in the modules passed via `default_module`.
    """
    class_ = locate(dict_["class"]) or getattr(default_module, dict_["class"])
    params = {}
    if "params" in dict_:
        params = dict_["params"]
    instance = class_(params, *args, **kwargs)
    return instance


def _maybe_load_yaml(item):
    """Parses `item` only if it is a string. If `item` is a dictionary
    it is returned as-is.
    """
    if isinstance(item, str):
        return yaml.load(item)
    elif isinstance(item, dict):
        return item
    else:
        raise ValueError("Got {}, expected YAML string or dict", type(item))


def _deep_merge_dict(dict_x, dict_y, path=None):
    """Recursively merges dict_y into dict_x.
    """
    if path is None: path = []
    for key in dict_y:
        if key in dict_x:
            if isinstance(dict_x[key], dict) and isinstance(dict_y[key], dict):
                _deep_merge_dict(dict_x[key], dict_y[key], path + [str(key)])
            elif dict_x[key] != dict_y[key]:
                dict_x[key] = dict_y[key]
        else:
            dict_x[key] = dict_y[key]
    return dict_x


def _parse_params(params, default_params):
    """Parses parameter values to the types defined by the default parameters.
    Default parameters are used for missing values.
    """
    # Cast parameters to correct types
    if params is None:
        params = {}
    result = copy.deepcopy(default_params)
    for key, value in params.items():
        # If param is unknown, drop it to stay compatible with past versions
        if key not in default_params:
            raise ValueError("%s is not a valid model parameter" % key)
        # Param is a dictionary
        if isinstance(value, dict):
            default_dict = default_params[key]
            if not isinstance(default_dict, dict):
                raise ValueError("%s should not be a dictionary", key)
            if default_dict:
                value = _parse_params(value, default_dict)
        if value is None:
            continue
        if default_params[key] is None:
            result[key] = value
        else:
            result[key] = type(default_params[key])(value)
    return result


class Configurable(object):
    """Interface for all classes that are configurable
    via a parameters dictionary.

    Args:
    params: A dictionary of parameters.
    """

    def __init__(self, params):
        self._params = _parse_params(params, self.default_params())
        self._print_params()

    def _print_params(self):
        """Logs parameter values"""
        classname = self.__class__.__name__
        tf.logging.info("\n%s", yaml.dump({classname: self._params}))

    @property
    def params(self):
        """Returns a dictionary of parsed parameters."""
        return self._params

    @staticmethod
    @abstractmethod
    def default_params():
        """Returns a dictionary of default parameters. The default parameters
        are used to define the expected type of passed parameters. Missing
        parameter values are replaced with the defaults returned by this method.
        """
        raise NotImplementedError
