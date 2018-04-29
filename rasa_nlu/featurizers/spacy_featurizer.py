from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import typing
from typing import Any

from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc


def ndim(spacy_nlp):
    """Number of features used to represent a document / sentence."""
    # type: Language -> int
    return spacy_nlp.vocab.vectors_length


def features_for_doc(doc):
    """Feature vector for a single document / sentence."""
    # type: Doc -> np.ndarray
    return doc.vector


class SpacyFeaturizer(Featurizer):
    name = "intent_featurizer_spacy"

    provides = ["text_features"]

    requires = ["spacy_doc"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None

        for example in training_data.intent_examples:
            self._set_spacy_features(example)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        self._set_spacy_features(message)

    def _set_spacy_features(self, message):
        """Adds the spacy word vectors to the messages text features."""

        fs = features_for_doc(message.get("spacy_doc"))
        features = self._combine_with_existing_text_features(message, fs)
        message.set("text_features", features)
