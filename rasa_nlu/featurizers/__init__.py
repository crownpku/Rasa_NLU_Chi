from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from rasa_nlu.components import Component


class Featurizer(Component):

    @staticmethod
    def _combine_with_existing_text_features(message,
                                             additional_features):
        if message.get("text_features") is not None:
            return np.hstack((message.get("text_features"),
                              additional_features))
        else:
            return additional_features
