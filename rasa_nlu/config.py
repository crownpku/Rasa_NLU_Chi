from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

import six
import yaml
from builtins import object
# Describes where to search for the config file if no location is specified
from typing import Text

from rasa_nlu import utils
from rasa_nlu.utils import json_to_string

DEFAULT_CONFIG_LOCATION = "config.yml"

DEFAULT_CONFIG = {
    "language": "en",
    "pipeline": [],
    "data": None,
}


logger = logging.getLogger(__name__)


class InvalidConfigError(ValueError):
    """Raised if an invalid configuration is encountered."""

    def __init__(self, message):
        # type: (Text) -> None
        super(InvalidConfigError, self).__init__(message)


def load(filename=None, **kwargs):
    if filename is None and os.path.isfile(DEFAULT_CONFIG_LOCATION):
        filename = DEFAULT_CONFIG_LOCATION

    if filename is not None:
        try:
            file_config = utils.read_yaml_file(filename)
        except yaml.parser.ParserError as e:
            raise InvalidConfigError("Failed to read configuration file "
                                     "'{}'. Error: {}".format(filename, e))

        if kwargs:
            file_config.update(kwargs)
        return RasaNLUModelConfig(file_config)
    else:
        return RasaNLUModelConfig(kwargs)


def override_defaults(defaults, custom):
    cfg = defaults or {}
    if custom:
        cfg.update(custom)
    return cfg


def make_path_absolute(path):
    if path and not os.path.isabs(path):
        return os.path.join(os.getcwd(), path)
    else:
        return path


class RasaNLUModelConfig(object):
    DEFAULT_PROJECT_NAME = "default"

    def __init__(self, configuration_values=None):

        if not configuration_values:
            configuration_values = {}

        self.override(DEFAULT_CONFIG)
        self.override(configuration_values)

        if isinstance(self.__dict__['pipeline'], six.string_types):
            from rasa_nlu import registry

            template_name = self.__dict__['pipeline']
            pipeline = registry.pipeline_template(template_name)

            if pipeline:
                # replaces the template with the actual components
                self.__dict__['pipeline'] = pipeline
            else:
                known_templates = ", ".join(
                        registry.registered_pipeline_templates.keys())

                raise InvalidConfigError("No pipeline specified and unknown "
                                         "pipeline template '{}' passed. Known "
                                         "pipeline templates: {}"
                                         "".format(template_name,
                                                   known_templates))

        for key, value in self.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __getstate__(self):
        return self.as_dict()

    def __setstate__(self, state):
        self.override(state)

    def items(self):
        return list(self.__dict__.items())

    def as_dict(self):
        return dict(list(self.items()))

    def view(self):
        return json_to_string(self.__dict__, indent=4)

    def for_component(self, name, defaults=None):
        for c in self.pipeline:
            if c.get("name") == name:
                return override_defaults(defaults, c)
        else:
            return defaults or {}

    @property
    def component_names(self):
        if self.pipeline:
            return [c.get("name") for c in self.pipeline]
        else:
            return []

    def set_component_attr(self, name, **kwargs):
        for c in self.pipeline:
            if c.get("name") == name:
                c.update(kwargs)
        else:
            logger.warn("Tried to set configuration value for component '{}' "
                        "which is not part of the pipeline.".format(name))

    def override(self, config):
        if config:
            self.__dict__.update(config)
