from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest

from rasa_nlu import registry
from rasa_nlu.components import find_unavailable_packages
from rasa_nlu.model import Metadata


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_no_components_with_same_name(component_class):
    """The name of the components need to be unique as they will
    be referenced by name when defining processing pipelines."""

    names = [cls.name for cls in registry.component_classes]
    assert names.count(component_class.name) == 1, \
        "There is more than one component named {}".format(component_class.name)


@pytest.mark.parametrize("pipeline_template",
                         registry.registered_pipeline_templates)
def test_all_components_in_model_templates_exist(pipeline_template):
    """We provide a couple of ready to use pipelines, this test ensures
    all components referenced by name in the
    pipeline definitions are available."""

    components = registry.registered_pipeline_templates[pipeline_template]
    for component in components:
        assert component in registry.registered_components, \
            "Model template contains unknown component."


@pytest.mark.parametrize("component_class", registry.component_classes)
def test_all_arguments_can_be_satisfied(component_class):
    """Check that `train` method parameters can be filled
    filled from the context. Similar to `pipeline_init` test."""

    # All available context arguments that will ever be generated during train
    # it might still happen, that in a certain pipeline
    # configuration arguments can not be satisfied!
    provided_properties = {provided
                           for c in registry.component_classes
                           for provided in c.provides}

    for req in component_class.requires:
        assert req in provided_properties, \
            "No component provides required property."


def test_find_unavailable_packages():
    unavailable = find_unavailable_packages(["my_made_up_package_name", "io",
                                             "foo_bar", "foo_bar"])
    assert unavailable == {"my_made_up_package_name", "foo_bar"}


def test_builder_create_unknown(component_builder, default_config):
    with pytest.raises(Exception) as excinfo:
        component_builder.create_component("my_made_up_componment",
                                           default_config)
    assert "Unknown component name" in str(excinfo.value)


def test_builder_create_by_module_path(component_builder, default_config):
    from rasa_nlu.featurizers.regex_featurizer import RegexFeaturizer

    path = "rasa_nlu.featurizers.regex_featurizer.RegexFeaturizer"
    component = component_builder.create_component(path, default_config)
    assert type(component) == RegexFeaturizer


def test_builder_load_unknown(component_builder):
    with pytest.raises(Exception) as excinfo:
        component_builder.load_component("my_made_up_componment", "",
                                         Metadata({}, None))
    assert "Unknown component name" in str(excinfo.value)
