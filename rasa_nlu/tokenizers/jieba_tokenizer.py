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

import os
import glob
import shutil

DEFAULT_DICT_FILE_NAME = "jieba_default_dict"
USER_DICTS_FOLDER_NAME = "jieba_user_dicts/"
USER_DICT_FILE_NAME = USER_DICTS_FOLDER_NAME + "user_dict.txt"

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

        import jieba as tokenizer

        component_conf = cfg.for_component(cls.name, cls.defaults)
        tokenizer = cls.init_jieba(tokenizer, component_conf)
                        
        return cls(component_conf, tokenizer)

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> JiebaTokenizer
                
        import jieba as tokenizer

        component_meta = model_metadata.for_component(cls.name)

        if component_meta.get("default_dict"):
            path_default_dict = os.path.join(model_dir, component_meta.get("default_dict"))
            component_meta["default_dict"] = path_default_dict
        if component_meta.get("user_dicts"):
            path_user_dicts = os.path.join(model_dir, component_meta.get("user_dicts"))
            component_meta["user_dicts"] = path_user_dicts
        tokenizer = cls.init_jieba(tokenizer, component_meta)

        return cls(component_meta, tokenizer)

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


    @classmethod
    def init_jieba(cls, tokenizer, dict_config):
        
        if dict_config.get("default_dict"):
            if os.path.isfile(dict_config.get("default_dict")):
                path_default_dict = glob.glob("{}".format(dict_config.get("default_dict")))
                tokenizer = cls.set_default_dict(tokenizer, path_default_dict[0])
            else:
                print("Because the path of Jieba Default Dictionary has to be a file, not a directory, \
                       so Jieba Default Dictionary hasn't been switched.")
        else:
            print("No Jieba Default Dictionary found")

        if dict_config.get("user_dicts"):
            if os.path.isdir(dict_config.get("user_dicts")):
                parse_pattern = "{}/*"
            else:
                parse_pattern = "{}"

            path_user_dicts = glob.glob(parse_pattern.format(dict_config.get("user_dicts")))    
            tokenizer = cls.set_user_dicts(tokenizer, path_user_dicts)
        else:
            print("No Jieba User Dictionary found")

        return tokenizer


    @staticmethod
    def set_default_dict(tokenizer, path_default_dict):
        print("Setting Jieba Default Dictionary at " + str(path_default_dict))
        tokenizer.set_dictionary(path_default_dict)
        
        return tokenizer


    @staticmethod
    def set_user_dicts(tokenizer, path_user_dicts):
        if len(path_user_dicts) > 0:
            for path_user_dict in path_user_dicts:
                print("Loading Jieba User Dictionary at " + str(path_user_dict))
                tokenizer.load_userdict(path_user_dict)
        else:
            print("No Jieba User Dictionary found")

        return tokenizer


    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        return_dict = {}

        if self.component_config.get("default_dict"):
            des_path_default_dict = os.path.join(model_dir, DEFAULT_DICT_FILE_NAME)
            if os.path.isfile(self.component_config.get("default_dict")):
                shutil.copy2(self.component_config.get("default_dict"), des_path_default_dict)
                return_dict.update({"default_dict": DEFAULT_DICT_FILE_NAME})
        
        if self.component_config.get("user_dicts"):
            des_path_user_dicts = os.path.join(model_dir, USER_DICTS_FOLDER_NAME)
            os.mkdir(des_path_user_dicts)
            if os.path.isdir(self.component_config.get("user_dicts")):
                parse_pattern = "{}/*"
                path_user_dicts = glob.glob(parse_pattern.format(self.component_config.get("user_dicts")))
                for path_user_dict in path_user_dicts:
                    shutil.copy2(path_user_dict, des_path_user_dicts)
                return_dict.update({"user_dicts":  USER_DICTS_FOLDER_NAME})
            else:
                des_path_user_dict = os.path.join(model_dir, USER_DICT_FILE_NAME)
                shutil.copy2(self.component_config.get("user_dicts"), des_path_user_dict)
                return_dict.update({"user_dicts":  USER_DICT_FILE_NAME})
        
        return return_dict
