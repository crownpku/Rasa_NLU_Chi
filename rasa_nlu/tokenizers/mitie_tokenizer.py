from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import str
import re

from typing import Any
from typing import Dict
from typing import List
from typing import Text
from typing import Tuple

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Token
from rasa_nlu.tokenizers import Tokenizer
from rasa_nlu.components import Component
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData


class MitieTokenizer(Tokenizer, Component):
    name = "tokenizer_mitie"

    provides = ["tokens"]

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["mitie"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def _token_from_offset(self, text, offset, encoded_sentence):
        return Token(text.decode('utf-8'),
                     self._byte_to_char_offset(encoded_sentence, offset))

    def tokenize(self, text):
        # type: (Text) -> List[Token]
        import mitie

        encoded_sentence = text.encode('utf-8')
        tokenized = mitie.tokenize_with_offsets(encoded_sentence)
        tokens = [self._token_from_offset(token, offset, encoded_sentence)
                  for token, offset in tokenized]
        return tokens

    @staticmethod
    def _byte_to_char_offset(text, byte_offset):
        return len(text[:byte_offset].decode('utf-8'))
