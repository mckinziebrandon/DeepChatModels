
import yaml
import copy
import tensorflow as tf
from abc import abstractmethod


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


class Component(object):
    """
    Args:
        params: A dictionary of parameters.
    """

    def __init__(self, params):
        self._params = _parse_params(params, self.default_params())
        self._print_params()

    def _print_params(self):
        classname = self.__class__.__name__
        tf.logging.info("\n%s", yaml.dump({classname: self._params}))

    @property
    def params(self):
        return self._params

    @staticmethod
    @abstractmethod
    def default_params():
        raise NotImplementedError



class GraphComponent(object):

    def __init__(self, name):

        self.name = name
        self._template = tf.make_template(name, self._build, create_scope_now_=True)
        self.__doc__ = self._build.__doc__
        self.__call__.__func__.__doc__ = self._build.__doc__

    def _build(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self._template(*args, **kwargs)

    def variable_scope(self):
        return tf.variable_scope(self._template.variable_scope)
