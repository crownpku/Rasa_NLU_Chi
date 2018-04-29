"""This is a somewhat delicate package. It contains all registered components
and preconfigured templates.

Hence, it imports all of the components. To avoid cycles, no component should
import this in module scope."""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import typing
from rasa_nlu import utils
from typing import Any
from typing import Optional
from typing import Text
from typing import Type

from rasa_nlu.classifiers.keyword_intent_classifier import \
    KeywordIntentClassifier
from rasa_nlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
from rasa_nlu.classifiers.sklearn_intent_classifier import \
    SklearnIntentClassifier
from rasa_nlu.classifiers.embedding_intent_classifier import \
    EmbeddingIntentClassifier
from rasa_nlu.extractors.duckling_extractor import DucklingExtractor
from rasa_nlu.extractors.duckling_http_extractor import DucklingHTTPExtractor
from rasa_nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa_nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from rasa_nlu.featurizers.mitie_featurizer import MitieFeaturizer
from rasa_nlu.featurizers.ngram_featurizer import NGramFeaturizer
from rasa_nlu.featurizers.regex_featurizer import RegexFeaturizer
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
from rasa_nlu.featurizers.count_vectors_featurizer import \
    CountVectorsFeaturizer
from rasa_nlu.model import Metadata
from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa_nlu.tokenizers.jieba_tokenizer import JiebaTokenizer
from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa_nlu.utils.mitie_utils import MitieNLP
from rasa_nlu.utils.spacy_utils import SpacyNLP

if typing.TYPE_CHECKING:
    from rasa_nlu.components import Component
    from rasa_nlu.config import RasaNLUModelConfig, RasaNLUModelConfig

# Classes of all known components. If a new component should be added,
# its class name should be listed here.
component_classes = [
    SpacyNLP, MitieNLP,
    SpacyEntityExtractor, MitieEntityExtractor, DucklingExtractor,
    CRFEntityExtractor, DucklingHTTPExtractor,
    EntitySynonymMapper,
    SpacyFeaturizer, MitieFeaturizer, NGramFeaturizer, RegexFeaturizer,
    CountVectorsFeaturizer,
    MitieTokenizer, SpacyTokenizer, WhitespaceTokenizer, JiebaTokenizer,
    SklearnIntentClassifier, MitieIntentClassifier, KeywordIntentClassifier,
    EmbeddingIntentClassifier
]

# Mapping from a components name to its class to allow name based lookup.
registered_components = {c.name: c for c in component_classes}

# To simplify usage, there are a couple of model templates, that already add
# necessary components in the right order. They also implement
# the preexisting `backends`.
registered_pipeline_templates = {
    "spacy_sklearn": [
        "nlp_spacy",
        "tokenizer_spacy",
        "intent_featurizer_spacy",
        "intent_entity_featurizer_regex",
        "ner_crf",
        "ner_synonyms",
        "intent_classifier_sklearn",
    ],
    "keyword": [
        "intent_classifier_keyword",
    ],
    "tensorflow_embedding": [
        "intent_featurizer_count_vectors",
        "intent_classifier_tensorflow_embedding"
    ]
}


def pipeline_template(s):
    components = registered_pipeline_templates.get(s)

    if components:
        # converts the list of components in the configuration
        # format expected (one json object per component)
        return [{"name": c} for c in components]

    else:
        return None


def get_component_class(component_name):
    # type: (Text) -> Optional[Type[Component]]
    """Resolve component name to a registered components class."""

    if component_name not in registered_components:
        try:
            return utils.class_from_module_path(component_name)
        except Exception:
            raise Exception(
                    "Failed to find component class for '{}'. Unknown "
                    "component name. Check your configured pipeline and make "
                    "sure the mentioned component is not misspelled. If you "
                    "are creating your own component, make sure it is either "
                    "listed as part of the `component_classes` in "
                    "`rasa_nlu.registry.py` or is a proper name of a class "
                    "in a module.".format(component_name))
    return registered_components[component_name]


def load_component_by_name(component_name,  # type: Text
                           model_dir,  # type: Text
                           metadata,  # type: Metadata
                           cached_component,  # type: Optional[Component]
                           **kwargs  # type: **Any
                           ):
    # type: (...) -> Optional[Component]
    """Resolves a component and calls it's load method to init it based on a
    previously persisted model."""

    component_clz = get_component_class(component_name)
    return component_clz.load(model_dir, metadata, cached_component, **kwargs)


def create_component_by_name(component_name, config):
    # type: (Text, RasaNLUModelConfig) -> Optional[Component]
    """Resolves a component and calls it's create method to init it based on a
    previously persisted model."""

    component_clz = get_component_class(component_name)
    return component_clz.create(config)
