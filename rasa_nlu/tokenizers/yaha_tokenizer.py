# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:54:35 2017

@author: user
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.components import Component
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

import sys
from yaha import Cuttor

reload(sys)
sys.setdefaultencoding('utf-8')

class YahaTokenizer(Tokenizer, Component):
    
    
    name = "tokenizer_yaha"

    provides = ["tokens"]
    
    cuttor = Cuttor()
    
    def __init__(self):
        pass
       

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["yaha"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None
        if config['language'] != 'zh':
            raise Exception("tokenizer_yaha is only used for Chinese. Check your configure json file.")
            
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Text) -> List[Token]
        tokenized = self.cuttor.tokenize(text.decode('utf-8'), search=True)
        tokens = [Token(word, start) for (word, start, end) in tokenized]

        return tokens
