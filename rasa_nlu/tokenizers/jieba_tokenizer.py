import re

from typing import Any
from typing import Dict
from typing import List
from typing import Text
from typing import Tuple

from rasa_nlu.tokenizers import Tokenizer
from rasa_nlu.components import Component

class JiebaTokenizer(Tokenizer, Component):
    name = "tokenizer_jieba"

    context_provides = {
        "process": ["tokens"],
    }

    def __init__(self):
        pass

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["jieba"]

    def tokenize(self, text):
        # type: (Text) -> List[Text]
        import jieba

        return [w.decode('utf-8') for w in jieba.lcut(text.encode('utf-8'))]

    def process(self, text):
        # type: (Text) -> Dict[Text, Any]

        return {
            "tokens": self.tokenize(text)
        }

    def tokenize_with_offsets(self, text):
        # type: (Text) -> Tuple[List[Text], List[int]]
        import jieba

        _text = text.encode('utf-8')
        offsets = []
        offset = 0
        tokens = [w.decode('utf-8') for w in jieba.lcut(_text)]
        for tok in tokens:
            m = re.search(re.escape(tok), text[offset:], re.UNICODE)
            if m is None:
                message = "Invalid MITIE offset. Token '{}' in message '{}'.".format(str(tok),
                                                                                     str(text.encode('utf-8')))
                raise ValueError(message)
            offsets.append(offset + m.start())
            offset += m.end()
        return tokens, offsets
