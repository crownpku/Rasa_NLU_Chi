from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import time
import warnings
from collections import Counter
from string import punctuation

import numpy as np
import typing
from builtins import map
from builtins import range
from typing import Any, Dict, List, Optional, Text

from rasa_nlu import utils
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_nlu.model import Metadata

NGRAM_MODEL_FILE_NAME = "ngram_featurizer.pkl"


class NGramFeaturizer(Featurizer):
    name = "intent_featurizer_ngrams"

    provides = ["text_features"]

    requires = ["spacy_doc"]

    defaults = {
        # defines the maximum number of ngrams to collect and add
        # to the featurization of a sentence
        "max_number_of_ngrams": 10,

        # the minimal length in characters of an ngram to be eligible
        "ngram_min_length": 3,

        # the maximal length in characters of an ngram to be eligible
        "ngram_max_length": 17,

        # the minimal number of times an ngram needs to occur in the
        # training data to be considered as a feature
        "ngram_min_occurrences": 5,

        # during cross validation (used to detect which ngrams are most
        # valuable) every intent with fever examples than this config
        # value will be excluded
        "min_intent_examples": 4,
    }

    def __init__(self, component_config=None):
        super(NGramFeaturizer, self).__init__(component_config)

        self.best_num_ngrams = None
        self.all_ngrams = None

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["spacy", "sklearn", "cloudpickle"]

    def train(self, training_data, cfg, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        start = time.time()
        self.train_on_sentences(training_data.intent_examples)
        logger.debug("Ngram collection took {} seconds"
                     "".format(time.time() - start))

        for example in training_data.training_examples:
            updated = self._text_features_with_ngrams(example,
                                                      self.best_num_ngrams)
            example.set("text_features", updated)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        updated = self._text_features_with_ngrams(message, self.best_num_ngrams)
        message.set("text_features", updated)

    def _text_features_with_ngrams(self, message, max_ngrams):

        ngrams_to_use = self._ngrams_to_use(max_ngrams)

        if ngrams_to_use is not None:
            extras = np.array(self._ngrams_in_sentence(message, ngrams_to_use))
            return self._combine_with_existing_text_features(message, extras)
        else:
            return message.get("text_features")

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[NGramFeaturizer]
             **kwargs  # type: **Any
             ):
        # type: (...) -> NGramFeaturizer

        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("featurizer_file", NGRAM_MODEL_FILE_NAME)
        featurizer_file = os.path.join(model_dir, file_name)

        if os.path.exists(featurizer_file):
            return utils.pycloud_unpickle(featurizer_file)
        else:
            return NGramFeaturizer(meta)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this model into the passed directory."""

        featurizer_file = os.path.join(model_dir, NGRAM_MODEL_FILE_NAME)
        utils.pycloud_pickle(featurizer_file, self)
        return {"featurizer_file": NGRAM_MODEL_FILE_NAME}

    def train_on_sentences(self, examples):
        labels = [e.get("intent") for e in examples]
        self.all_ngrams = self._get_best_ngrams(examples, labels)
        self.best_num_ngrams = self._cross_validation(examples, labels)

    def _ngrams_to_use(self, num_ngrams):
        if num_ngrams == 0 or self.all_ngrams is None:
            return []
        elif num_ngrams is not None:
            return self.all_ngrams[:num_ngrams]
        else:
            return self.all_ngrams

    def _get_best_ngrams(self, examples, labels):
        """Return an ordered list of the best character ngrams."""

        oov_strings = self._remove_in_vocab_words(examples)
        ngrams = self._generate_all_ngrams(
                oov_strings, self.component_config["ngram_min_length"])
        return self._sort_applicable_ngrams(ngrams, examples, labels)

    def _remove_in_vocab_words(self, examples):
        """Automatically removes words with digits in them, that may be a
        hyperlink or that _are_ in vocabulary for the nlp."""

        new_sents = []
        for example in examples:
            new_sents.append(self._remove_in_vocab_words_from_sentence(example))
        return new_sents

    @staticmethod
    def _is_ngram_worthy(token):
        """Decide if we should use this token for ngram counting.

        Excludes every word with digits in them, hyperlinks or
        an assigned word vector."""
        return (not token.has_vector and not token.like_url
                and not token.like_num and not token.like_email
                and not token.is_punct)

    def _remove_in_vocab_words_from_sentence(self, example):
        """Filter for words that do not have a word vector."""

        cleaned_tokens = [token
                          for token in example.get("spacy_doc")
                          if self._is_ngram_worthy(token)]

        # keep only out-of-vocab 'non_word' words
        non_words = ' '.join([t.text for t in cleaned_tokens])

        # remove digits and extra spaces
        non_words = ''.join([letter
                             for letter in non_words
                             if not letter.isdigit()])
        non_words = ' '.join([word
                              for word in non_words.split(' ')
                              if word != ''])

        # add cleaned sentence to list of these sentences
        return non_words

    def _intents_with_enough_examples(self, labels, examples):
        """Filter examples where we do not have a min number of examples."""

        min_intent_examples = self.component_config["min_intent_examples"]
        usable_labels = []

        for label in np.unique(labels):
            lab_sents = np.array(examples)[np.array(labels) == label]
            if len(lab_sents) < min_intent_examples:
                continue
            usable_labels.append(label)

        return usable_labels

    def _rank_ngrams_using_cv(self, examples, labels, list_of_ngrams):
        from sklearn import linear_model

        X = np.array(self._ngrams_in_sentences(examples, list_of_ngrams))
        y = self.encode_labels(labels)

        clf = linear_model.RandomizedLogisticRegression(C=1)
        clf.fit(X, y)

        # sort the ngrams according to the classification score
        scores = clf.scores_
        sorted_idxs = sorted(enumerate(scores), key=lambda x: -1 * x[1])
        sorted_ngrams = [list_of_ngrams[i[0]] for i in sorted_idxs]

        return sorted_ngrams

    def _sort_applicable_ngrams(self, ngrams_list, examples, labels):
        """Given an intent classification problem and a list of ngrams,

        creates ordered list of most useful ngrams."""

        if not ngrams_list:
            return []

        # make sure we have enough labeled instances for cv
        usable_labels = self._intents_with_enough_examples(labels, examples)

        mask = [label in usable_labels for label in labels]
        if any(mask) and len(usable_labels) >= 2:
            try:
                examples = np.array(examples)[mask]
                labels = np.array(labels)[mask]

                return self._rank_ngrams_using_cv(
                        examples, labels, ngrams_list)
            except ValueError as e:
                if "needs samples of at least 2 classes" in str(e):
                    # we got unlucky during the random
                    # sampling :( and selected a slice that
                    # only contains one class
                    return []
                else:
                    raise e
        else:
            # there is no example we can use for the cross validation
            return []

    def _ngrams_in_sentences(self, examples, ngrams):
        """Given a set of sentences, returns a feature vector for each sentence.

        The first $k$ elements are from the `intent_features`,
        the rest are {1,0} elements denoting whether an ngram is in sentence."""

        all_vectors = []
        for example in examples:
            presence_vector = self._ngrams_in_sentence(example, ngrams)
            all_vectors.append(presence_vector)
        return all_vectors

    def _ngrams_in_sentence(self, example, ngrams):
        """Given a set of sentences, return a vector indicating ngram presence.

        The vector will return 1 entries if the corresponding ngram is
        present in the sentence and 0 if it is not."""

        cleaned_sentence = self._remove_in_vocab_words_from_sentence(example)
        presence_vector = np.zeros(len(ngrams))
        idx_array = [idx
                     for idx in range(len(ngrams))
                     if ngrams[idx] in cleaned_sentence]
        presence_vector[idx_array] = 1
        return presence_vector

    def _generate_all_ngrams(self, list_of_strings, ngram_min_length):
        """Takes a list of strings and generates all character ngrams.

        Generated ngrams are at least 3 characters (and at most 17),
        occur at least 5 times and occur independently of longer
        superset ngrams at least once."""

        features = {}
        counters = {ngram_min_length - 1: Counter()}
        max_length = self.component_config["ngram_max_length"]

        for n in range(ngram_min_length, max_length):
            candidates = []
            features[n] = []
            counters[n] = Counter()

            # generate all possible n length ngrams
            for text in list_of_strings:
                text = text.replace(punctuation, ' ')
                for word in text.lower().split(' '):
                    cands = [word[i:i + n] for i in range(len(word) - n)]
                    for cand in cands:
                        counters[n][cand] += 1
                        if cand not in candidates:
                            candidates.append(cand)

            min_count = self.component_config["ngram_min_occurrences"]
            # iterate over these candidates picking only the applicable ones
            for can in candidates:
                if counters[n][can] >= min_count:
                    features[n].append(can)
                    begin = can[:-1]
                    end = can[1:]
                    if n >= ngram_min_length:
                        if (counters[n - 1][begin] == counters[n][can]
                                and begin in features[n - 1]):
                            features[n - 1].remove(begin)
                        if (counters[n - 1][end] == counters[n][can]
                                and end in features[n - 1]):
                            features[n - 1].remove(end)

        return [item for sublist in list(features.values()) for item in sublist]

    @staticmethod
    def _collect_features(examples):
        if examples:
            collected_features = [e.get("text_features")
                                  for e in examples
                                  if e.get("text_features") is not None]
        else:
            collected_features = []

        if collected_features:
            return np.stack(collected_features)
        else:
            return None

    def _append_ngram_features(self, examples, existing_features, max_ngrams):
        ngrams_to_use = self._ngrams_to_use(max_ngrams)
        extras = np.array(self._ngrams_in_sentences(examples,
                                                    ngrams_to_use))
        if existing_features is not None:
            return np.hstack((existing_features, extras))
        else:
            return extras

    @staticmethod
    def _num_cv_splits(y):
        return min(10, np.min(np.bincount(y))) if y.size > 0 else 0

    @staticmethod
    def encode_labels(labels):
        from sklearn import preprocessing

        intent_encoder = preprocessing.LabelEncoder()
        intent_encoder.fit(labels)
        return intent_encoder.transform(labels)

    def _score_ngram_selection(self, examples, y, existing_text_features,
                               cv_splits, max_ngrams):
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression

        if existing_text_features is None:
            return 0.0

        clf = LogisticRegression(class_weight='balanced')

        no_ngrams_X = self._append_ngram_features(
                examples, existing_text_features, max_ngrams)
        return np.mean(cross_val_score(clf, no_ngrams_X, y, cv=cv_splits))

    @staticmethod
    def _generate_test_points(max_ngrams):
        """Generate a list of increasing numbers.

        They are used to take the best n ngrams and evaluate them. This n
        is varied to find the best number of ngrams to use. This function
        defines the number of ngrams that get tested."""

        possible_ngrams = np.linspace(0, max_ngrams, 8)
        return np.unique(list(map(int, np.floor(possible_ngrams))))

    def _cross_validation(self, examples, labels):
        """Choose the best number of ngrams to include in bow.

        Given an intent classification problem and a set of ordered ngrams
        (ordered in terms of importance by pick_applicable_ngrams) we
        choose the best number of ngrams to include in our bow vecs
        by cross validation."""

        max_ngrams = self.component_config["max_number_of_ngrams"]

        if not self.all_ngrams:
            logger.debug("Found no ngrams. Using existing features.")
            return 0

        existing_text_features = self._collect_features(examples)

        y = self.encode_labels(labels)
        cv_splits = self._num_cv_splits(y)

        if cv_splits >= 3:
            logger.debug("Started ngram cross-validation to find b"
                         "est number of ngrams to use...")

            scores = []
            num_ngrams = self._generate_test_points(max_ngrams)
            for n in num_ngrams:
                score = self._score_ngram_selection(examples, y,
                                                    existing_text_features,
                                                    cv_splits,
                                                    max_ngrams=n)
                scores.append(score)
                logger.debug("Evaluating usage of {} ngrams. "
                             "Score: {}".format(n, score))

            n_top = num_ngrams[np.argmax(scores)]
            logger.info("Best score with {} ngrams: "
                        "{}".format(n_top, np.max(scores)))
            return n_top
        else:
            warnings.warn("Can't cross-validate ngram featurizer. "
                          "There aren't enough examples per intent "
                          "(at least 3)")
            return max_ngrams
