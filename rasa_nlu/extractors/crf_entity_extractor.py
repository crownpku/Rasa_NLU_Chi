from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

import typing
from builtins import str
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import sklearn_crfsuite

CRF_MODEL_FILE_NAME = "crf_model.pkl"


class CRFEntityExtractor(EntityExtractor):
    name = "ner_crf"

    provides = ["entities"]

    requires = ["spacy_doc", "tokens"]

    defaults = {
        # BILOU_flag determines whether to use BILOU tagging or not.
        # More rigorous however requires more examples per entity
        # rule of thumb: use only if more than 100 egs. per entity
        "BILOU_flag": True,

        # crf_features is [before, word, after] array with before, word,
        # after holding keys about which
        # features to use for each word, for example, 'title' in
        # array before will have the feature
        # "is the preceding word in title case?"
        "features": [
            ["low", "title", "upper", "pos", "pos2"],
            ["bias", "low", "word3", "word2", "upper",
             "title", "digit", "pos", "pos2", "pattern"],
            ["low", "title", "upper", "pos", "pos2"]],

        # The maximum number of iterations for optimization algorithms.
        "max_iterations": 50,

        # weight of theL1 regularization
        "L1_c": 1,

        # weight of the L2 regularization
        "L2_c": 1e-3
    }

    function_dict = {
        'low': lambda doc: doc[0].lower(),
        'title': lambda doc: doc[0].istitle(),
        'word3': lambda doc: doc[0][-3:],
        'word2': lambda doc: doc[0][-2:],
        'word1': lambda doc: doc[0][-1:],
        'pos': lambda doc: doc[1],
        'pos2': lambda doc: doc[1][:2],
        'bias': lambda doc: 'bias',
        'upper': lambda doc: doc[0].isupper(),
        'digit': lambda doc: doc[0].isdigit(),
        'pattern': lambda doc: str(doc[3]) if doc[3] is not None else 'N/A',
    }

    def __init__(self, component_config=None, ent_tagger=None):
        # type: (sklearn_crfsuite.CRF, Dict[Text, Any]) -> None

        super(CRFEntityExtractor, self).__init__(component_config)

        self.ent_tagger = ent_tagger

        self._validate_configuration()

    def _validate_configuration(self):
        if len(self.component_config.get("features", [])) % 2 != 1:
            raise ValueError("Need an odd number of crf feature "
                             "lists to have a center word.")

    @classmethod
    def required_packages(cls):
        return ["sklearn_crfsuite", "sklearn", "spacy"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig) -> None

        self.component_config = config.for_component(self.name, self.defaults)

        self._validate_configuration()

        # checks whether there is at least one
        # example with an entity annotation
        if training_data.entity_examples:

            # filter out pre-trained entity examples
            filtered_entity_examples = self.filter_trainable_entities(
                    training_data.training_examples)

            # convert the dataset into features
            # this will train on ALL examples, even the ones
            # without annotations
            dataset = self._create_dataset(filtered_entity_examples)

            self._train_model(dataset)

    def _create_dataset(self, examples):
        # type: (List[Message]) -> List[List[Tuple[Text, Text, Text, Text]]]
        dataset = []
        for example in examples:
            entity_offsets = self._convert_example(example)
            dataset.append(self._from_json_to_crf(example, entity_offsets))
        return dataset

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.extract_entities(message))
        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)

    @staticmethod
    def _convert_example(example):
        # type: (Message) -> List[Tuple[int, int, Text]]

        def convert_entity(entity):
            return entity["start"], entity["end"], entity["entity"]

        return [convert_entity(ent) for ent in example.get("entities", [])]

    def extract_entities(self, message):
        # type: (Message) -> List[Dict[Text, Any]]
        """Take a sentence and return entities in json format"""

        if self.ent_tagger is not None:
            text_data = self._from_text_to_crf(message)
            features = self._sentence_to_features(text_data)
            ents = self.ent_tagger.predict_marginals_single(features)
            return self._from_crf_to_json(message, ents)
        else:
            return []

    def most_likely_entity(self, idx, entities):
        if len(entities) > idx:
            entity_probs = entities[idx]
        else:
            entity_probs = None
        if entity_probs:
            label = max(entity_probs,
                        key=lambda key: entity_probs[key])
            if self.component_config["BILOU_flag"]:
                # if we are using bilou flags, we will combine the prob
                # of the B, I, L and U tags for an entity (so if we have a
                # score of 60% for `B-address` and 40% and 30%
                # for `I-address`, we will return 70%)
                return label, sum([v
                                   for k, v in entity_probs.items()
                                   if k[2:] == label[2:]])
            else:
                return label, entity_probs[label]
        else:
            return "", 0.0

    @staticmethod
    def _create_entity_dict(sentence_doc, start, end, entity, confidence):
        return {
            'start': sentence_doc[start].idx,
            'end': sentence_doc[start:end + 1].end_char,
            'value': sentence_doc[start:end + 1].text,
            'entity': entity,
            'confidence': confidence
        }

    @staticmethod
    def _entity_from_label(label):
        return label[2:]

    @staticmethod
    def _bilou_from_label(label):
        if len(label) >= 2 and label[1] == "-":
            return label[0].upper()
        return None

    def _find_bilou_end(self, word_idx, entities):
        ent_word_idx = word_idx + 1
        finished = False

        # get information about the first word, tagged with `B-...`
        label, confidence = self.most_likely_entity(word_idx, entities)
        entity_label = self._entity_from_label(label)

        while not finished:
            label, label_confidence = self.most_likely_entity(
                    ent_word_idx, entities)

            confidence = min(confidence, label_confidence)

            if label[2:] != entity_label:
                # words are not tagged the same entity class
                logger.debug("Inconsistent BILOU tagging found, B- tag, L- "
                             "tag pair encloses multiple entity classes.i.e. "
                             "[B-a, I-b, L-a] instead of [B-a, I-a, L-a].\n"
                             "Assuming B- class is correct.")

            if label.startswith('L-'):
                # end of the entity
                finished = True
            elif label.startswith('I-'):
                # middle part of the entity
                ent_word_idx += 1
            else:
                # entity not closed by an L- tag
                finished = True
                ent_word_idx -= 1
                logger.debug("Inconsistent BILOU tagging found, B- tag not "
                             "closed by L- tag, i.e [B-a, I-a, O] instead of "
                             "[B-a, L-a, O].\nAssuming last tag is L-")
        return ent_word_idx, confidence

    def _handle_bilou_label(self, word_idx, entities):
        label, confidence = self.most_likely_entity(word_idx, entities)
        entity_label = self._entity_from_label(label)

        if self._bilou_from_label(label) == "U":
            return word_idx, confidence, entity_label

        elif self._bilou_from_label(label) == "B":
            # start of multi word-entity need to represent whole extent
            ent_word_idx, confidence = self._find_bilou_end(
                    word_idx, entities)
            return ent_word_idx, confidence, entity_label

        else:
            return None, None, None

    def _from_crf_to_json(self, message, entities):
        # type: (Message, List[Any]) -> List[Dict[Text, Any]]

        sentence_doc = message.get("spacy_doc")

        if len(sentence_doc) != len(entities):
            raise Exception('Inconsistency in amount of tokens '
                            'between crfsuite and spacy')

        if self.component_config["BILOU_flag"]:
            return self._convert_bilou_tagging_to_entity_result(
                    sentence_doc, entities)
        else:
            # not using BILOU tagging scheme, multi-word entities are split.
            return self._convert_simple_tagging_to_entity_result(
                    sentence_doc, entities)

    def _convert_bilou_tagging_to_entity_result(self, sentence_doc, entities):
        # using the BILOU tagging scheme
        json_ents = []
        word_idx = 0
        while word_idx < len(sentence_doc):
            end_idx, confidence, entity_label = self._handle_bilou_label(
                    word_idx, entities)

            if end_idx is not None:
                ent = self._create_entity_dict(sentence_doc,
                                               word_idx,
                                               end_idx,
                                               entity_label,
                                               confidence)
                json_ents.append(ent)
                word_idx = end_idx + 1
            else:
                word_idx += 1
        return json_ents

    def _convert_simple_tagging_to_entity_result(self, sentence_doc, entities):
        json_ents = []

        for word_idx in range(len(sentence_doc)):
            entity_label, confidence = self.most_likely_entity(
                    word_idx, entities)
            word = sentence_doc[word_idx]
            if entity_label != 'O':
                ent = {'start': word.idx,
                       'end': word.idx + len(word),
                       'value': word.text,
                       'entity': entity_label,
                       'confidence': confidence}
                json_ents.append(ent)

        return json_ents

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[CRFEntityExtractor]
             **kwargs  # type: **Any
             ):
        # type: (...) -> CRFEntityExtractor
        from sklearn.externals import joblib

        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("classifier_file", CRF_MODEL_FILE_NAME)
        model_file = os.path.join(model_dir, file_name)

        if os.path.exists(model_file):
            ent_tagger = joblib.load(model_file)
            return CRFEntityExtractor(meta, ent_tagger)
        else:
            return CRFEntityExtractor(meta)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""

        from sklearn.externals import joblib

        if self.ent_tagger:
            model_file_name = os.path.join(model_dir, CRF_MODEL_FILE_NAME)

            joblib.dump(self.ent_tagger, model_file_name)

        return {"classifier_file": CRF_MODEL_FILE_NAME}

    def _sentence_to_features(self, sentence):
        # type: (List[Tuple[Text, Text, Text, Text]]) -> List[Dict[Text, Any]]
        """Convert a word into discrete features in self.crf_features,
        including word before and word after."""

        configured_features = self.component_config["features"]
        sentence_features = []
        for word_idx in range(len(sentence)):
            # word before(-1), current word(0), next word(+1)
            feature_span = len(configured_features)
            half_span = feature_span // 2
            feature_range = range(- half_span, half_span + 1)
            prefixes = [str(i) for i in feature_range]
            word_features = {}
            for f_i in feature_range:
                if word_idx + f_i >= len(sentence):
                    word_features['EOS'] = True
                    # End Of Sentence
                elif word_idx + f_i < 0:
                    word_features['BOS'] = True
                    # Beginning Of Sentence
                else:
                    word = sentence[word_idx + f_i]
                    f_i_from_zero = f_i + half_span
                    prefix = prefixes[f_i_from_zero]
                    features = configured_features[f_i_from_zero]
                    for feature in features:
                        # append each feature to a feature vector
                        value = self.function_dict[feature](word)
                        word_features[prefix + ":" + feature] = value
            sentence_features.append(word_features)
        return sentence_features

    @staticmethod
    def _sentence_to_labels(sentence):
        # type: (List[Tuple[Text, Text, Text, Text]]) -> List[Text]

        return [label for _, _, label, _ in sentence]

    def _from_json_to_crf(self,
                          message,  # type: Message
                          entity_offsets  # type: List[Tuple[int, int, Text]]
                          ):
        # type: (...) -> List[Tuple[Text, Text, Text, Text]]
        """Convert json examples to format of underlying crfsuite."""
        from spacy.gold import GoldParse

        doc = message.get("spacy_doc")
        gold = GoldParse(doc, entities=entity_offsets)
        ents = [l[5] for l in gold.orig_annot]
        if '-' in ents:
            logger.warn("Misaligned entity annotation in sentence '{}'. "
                        "Make sure the start and end values of the "
                        "annotated training examples end at token "
                        "boundaries (e.g. don't include trailing "
                        "whitespaces).".format(doc.text))
        if not self.component_config["BILOU_flag"]:
            for i, label in enumerate(ents):
                if self._bilou_from_label(label) in {"B", "I", "U", "L"}:
                    # removes BILOU prefix from label
                    ents[i] = self._entity_from_label(label)

        return self._from_text_to_crf(message, ents)

    @staticmethod
    def __pattern_of_token(message, i):
        if message.get("tokens"):
            return message.get("tokens")[i].get("pattern")
        else:
            return None

    @staticmethod
    def __tag_of_token(token):
        import spacy
        if spacy.about.__version__ > "2" and token._.has("tag"):
            return token._.get("tag")
        else:
            return token.tag_

    def _from_text_to_crf(self, message, entities=None):
        # type: (Message, List[Text]) -> List[Tuple[Text, Text, Text, Text]]
        """Takes a sentence and switches it to crfsuite format."""

        crf_format = []
        for i, token in enumerate(message.get("spacy_doc")):
            pattern = self.__pattern_of_token(message, i)
            entity = entities[i] if entities else "N/A"
            tag = self.__tag_of_token(token)
            crf_format.append((token.text, tag, entity, pattern))
        return crf_format

    def _train_model(self, df_train):
        # type: (List[List[Tuple[Text, Text, Text, Text]]]) -> None
        """Train the crf tagger based on the training data."""
        import sklearn_crfsuite

        X_train = [self._sentence_to_features(sent) for sent in df_train]
        y_train = [self._sentence_to_labels(sent) for sent in df_train]
        self.ent_tagger = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                # coefficient for L1 penalty
                c1=self.component_config["L1_c"],
                # coefficient for L2 penalty
                c2=self.component_config["L2_c"],
                # stop earlier
                max_iterations=self.component_config["max_iterations"],
                # include transitions that are possible, but not observed
                all_possible_transitions=True
        )
        self.ent_tagger.fit(X_train, y_train)
