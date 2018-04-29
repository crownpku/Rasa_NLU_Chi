from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import pytest

from rasa_nlu import training_data, config
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData


@pytest.mark.parametrize("sentence, expected", [
    ("hey how are you today", [-0.19649599, 0.32493639,
                               -0.37408298, -0.10622784, 0.062756])
])
def test_spacy_featurizer(sentence, expected, spacy_nlp):
    from rasa_nlu.featurizers import spacy_featurizer
    doc = spacy_nlp(sentence)
    vecs = spacy_featurizer.features_for_doc(doc)
    assert np.allclose(doc.vector[:5], expected, atol=1e-5)
    assert np.allclose(vecs, doc.vector, atol=1e-5)


def test_mitie_featurizer(mitie_feature_extractor, default_config):
    from rasa_nlu.featurizers.mitie_featurizer import MitieFeaturizer

    ftr = MitieFeaturizer.create(config.load("sample_configs/config_mitie.yml"))
    sentence = "Hey how are you today"
    tokens = MitieTokenizer().tokenize(sentence)
    vecs = ftr.features_for_tokens(tokens, mitie_feature_extractor)
    expected = np.array([0., -4.4551446, 0.26073121, -1.46632245, -1.84205751])
    assert np.allclose(vecs[:5], expected, atol=1e-5)


def test_ngram_featurizer(spacy_nlp):
    from rasa_nlu.featurizers.ngram_featurizer import NGramFeaturizer
    ftr = NGramFeaturizer({"max_number_of_ngrams": 10})

    # ensures that during random sampling of the ngram CV we don't end up
    # with a one-class-split
    repetition_factor = 5

    greet = {"intent": "greet", "text_features": [0.5]}
    goodbye = {"intent": "goodbye", "text_features": [0.5]}
    labeled_sentences = [
                            Message("heyheyheyhey", greet),
                            Message("howdyheyhowdy", greet),
                            Message("heyhey howdyheyhowdy", greet),
                            Message("howdyheyhowdy heyhey", greet),
                            Message("astalavistasista", goodbye),
                            Message("astalavistasista sistala", goodbye),
                            Message("sistala astalavistasista", goodbye),
                        ] * repetition_factor

    for m in labeled_sentences:
        m.set("spacy_doc", spacy_nlp(m.text))

    ftr.min_intent_examples_for_ngram_classification = 2
    ftr.train_on_sentences(labeled_sentences)
    assert len(ftr.all_ngrams) > 0
    assert ftr.best_num_ngrams > 0


@pytest.mark.parametrize("sentence, expected, labeled_tokens", [
    ("hey how are you today", [0., 1.], [0]),
    ("hey 123 how are you", [1., 1.], [0, 1]),
    ("blah balh random eh", [0., 0.], []),
    ("looks really like 123 today", [1., 0.], [3]),
])
def test_regex_featurizer(sentence, expected, labeled_tokens, spacy_nlp):
    from rasa_nlu.featurizers.regex_featurizer import RegexFeaturizer
    patterns = [
        {"pattern": '[0-9]+', "name": "number", "usage": "intent"},
        {"pattern": '\\bhey*', "name": "hello", "usage": "intent"}
    ]
    ftr = RegexFeaturizer(known_patterns=patterns)

    # adds tokens to the message
    tokenizer = SpacyTokenizer()
    message = Message(sentence)
    message.set("spacy_doc", spacy_nlp(sentence))
    tokenizer.process(message)

    result = ftr.features_for_patterns(message)
    assert np.allclose(result, expected, atol=1e-10)

    # the tokenizer should have added tokens
    assert len(message.get("tokens", [])) > 0
    for i, token in enumerate(message.get("tokens")):
        if i in labeled_tokens:
            assert token.get("pattern") in [0, 1]
        else:
            # if the token is not part of a regex the pattern should not be set
            assert token.get("pattern") is None


def test_spacy_featurizer_casing(spacy_nlp):
    from rasa_nlu.featurizers import spacy_featurizer

    # if this starts failing for the default model, we should think about
    # removing the lower casing the spacy nlp component does when it
    # retrieves vectors. For compressed spacy models (e.g. models
    # ending in _sm) this test will most likely fail.

    td = training_data.load_data('data/examples/rasa/demo-rasa.json')
    for e in td.intent_examples:
        doc = spacy_nlp(e.text)
        doc_capitalized = spacy_nlp(e.text.capitalize())

        vecs = spacy_featurizer.features_for_doc(doc)
        vecs_capitalized = spacy_featurizer.features_for_doc(doc_capitalized)

        assert np.allclose(vecs, vecs_capitalized, atol=1e-5), \
            "Vectors are unequal for texts '{}' and '{}'".format(
                    e.text, e.text.capitalize())


@pytest.mark.parametrize("sentence, expected", [
    ("hello hello hello hello hello ", [5]),
    ("hello goodbye hello", [1, 2]),
    ("a b c d e f", [1, 1, 1, 1, 1, 1]),
    ("a 1 2", [2, 1])
])
def test_count_vector_featurizer(sentence, expected):
    from rasa_nlu.featurizers.count_vectors_featurizer import \
        CountVectorsFeaturizer

    ftr = CountVectorsFeaturizer({"token_pattern": r'(?u)\b\w+\b'})
    message = Message(sentence)
    message.set("intent", "bla")
    data = TrainingData([message])

    ftr.train(data)
    ftr.process(message)

    assert np.all(message.get("text_features")[0] == expected)
