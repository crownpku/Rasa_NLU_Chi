from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.components import Component
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData


class JiebaTokenizer(Tokenizer, Component):
    
    
    name = "tokenizer_jieba"

    provides = ["tokens"]

    language_list = ["zh"]

    def __init__(self,
                 component_config=None,  # type: Dict[Text, Any]
                 tokenizer=None
                 ):
        # type: (...) -> None
        
        super(JiebaTokenizer, self).__init__(component_config)

        self.tokenizer = tokenizer


    @classmethod
    def create(cls, cfg):
        # type: (RasaNLUModelConfig) -> JiebaTokenizer
        import glob        
        import jieba as tokenizer
        component_conf = cfg.for_component(cls.name, cls.defaults)
        jieba_defaultdict = glob.glob("{}".format(component_conf.get("default_dict")))
        jieba_userdicts = glob.glob("{}/*".format(component_conf.get("user_dicts")))
        tokenizer = cls.set_defaultdict(tokenizer, jieba_defaultdict)
        tokenizer = cls.set_userdicts(tokenizer, jieba_userdicts)
        
        return JiebaTokenizer(component_conf, tokenizer)

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        import glob        
        import jieba as tokenizer
        component_meta = model_metadata.for_component(cls.name)
        jieba_defaultdict = glob.glob("{}".format(component_meta.get("default_dict")))
        jieba_userdicts = glob.glob("{}/*".format(component_meta.get("user_dicts")))
        tokenizer = cls.set_defaultdict(tokenizer, jieba_defaultdict)
        tokenizer = cls.set_userdicts(tokenizer, jieba_userdicts)

        return JiebaTokenizer(component_meta, tokenizer)

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["jieba"]


    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
            
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))


    def tokenize(self, text):
        # type: (Text) -> List[Token]
        tokenized = self.tokenizer.tokenize(text)
        tokens = [Token(word, start) for (word, start, end) in tokenized]

        return tokens


    @staticmethod
    def set_defaultdict(tokenizer, jieba_defaultdict):
        if len(jieba_defaultdict) == 0:
            print("No Jieba Default Dictionary found")
        elif len(jieba_defaultdict) == 1:
            print("Setting Jieba Default Dictionary at " + str(jieba_defaultdict[0]))
            tokenizer.set_dictionary(jieba_defaultdict[0])
        else:
            print("The number of Jieba Default Dictionaries has to be one only")
        return tokenizer


    @staticmethod
    def set_userdicts(tokenizer, jieba_userdicts):
        if len(jieba_userdicts) > 0:
            for jieba_userdict in jieba_userdicts:
                print("Loading Jieba User Dictionary at " + str(jieba_userdict))
                tokenizer.load_userdict(jieba_userdict)
        else:
            print("No Jieba User Dictionary found")
        return tokenizer

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        return {
            "user_dicts": self.component_config.get("user_dicts"),
            "default_dict": self.component_config.get("default_dict")
        }
