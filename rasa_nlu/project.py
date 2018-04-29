from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import glob

import os
import logging

from builtins import object
from threading import Lock

from rasa_nlu import utils
from typing import Text, List

from rasa_nlu.classifiers.keyword_intent_classifier import \
    KeywordIntentClassifier
from rasa_nlu.model import Metadata, Interpreter

logger = logging.getLogger(__name__)

MODEL_NAME_PREFIX = "model_"

FALLBACK_MODEL_NAME = "fallback"


class Project(object):
    def __init__(self,
                 component_builder=None,
                 project=None,
                 project_dir=None,
                 remote_storage=None):
        self._component_builder = component_builder
        self._models = {}
        self.status = 0
        self._reader_lock = Lock()
        self._loader_lock = Lock()
        self._writer_lock = Lock()
        self._readers_count = 0
        self._path = None
        self._project = project
        self.remote_storage = remote_storage

        if project and project_dir:
            self._path = os.path.join(project_dir, project)
        self._search_for_models()

    def _begin_read(self):
        # Readers-writer lock basic double mutex implementation
        self._reader_lock.acquire()
        self._readers_count += 1
        if self._readers_count == 1:
            self._writer_lock.acquire()
        self._reader_lock.release()

    def _end_read(self):
        self._reader_lock.acquire()
        self._readers_count -= 1
        if self._readers_count == 0:
            self._writer_lock.release()
        self._reader_lock.release()

    def _load_local_model(self, requested_model_name=None):
        if requested_model_name is None:  # user want latest model
            # NOTE: for better parse performance, currently although
            # user may want latest model by set requested_model_name
            # explicitly to None, we are not refresh model list
            # from local and cloud which is pretty slow.
            # User can specific requested_model_name to the latest model name,
            # then model will be cached, this is a kind of workaround to
            # refresh latest project model.
            # BTW if refresh function is wanted, maybe add implement code to
            # `_latest_project_model()` is a good choice.

            logger.debug("No model specified. Using default")
            return self._latest_project_model()

        elif requested_model_name in self._models:  # model exists in cache
            return requested_model_name

        return None  # local model loading failed!

    def _dynamic_load_model(self, requested_model_name=None):
        # type: (Text) -> Text

        # first try load from local cache
        local_model = self._load_local_model(requested_model_name)
        if local_model:
            return local_model

        # now model not exists in model list cache
        # refresh model list from local and cloud

        # NOTE: if a malicious user sent lots of requests
        # with not existing model will cause performance issue.
        # because get anything from cloud is a time-consuming task
        self._search_for_models()

        # retry after re-fresh model cache
        local_model = self._load_local_model(requested_model_name)
        if local_model:
            return local_model

        # still not found user specified model
        logger.warn("Invalid model requested. Using default")
        return self._latest_project_model()

    def parse(self, text, time=None, requested_model_name=None):
        self._begin_read()

        model_name = self._dynamic_load_model(requested_model_name)

        self._loader_lock.acquire()
        try:
            if not self._models.get(model_name):
                interpreter = self._interpreter_for_model(model_name)
                self._models[model_name] = interpreter
        finally:
            self._loader_lock.release()

        response = self._models[model_name].parse(text, time)

        self._end_read()

        return response, model_name

    def load_model(self):
        self._begin_read()
        status = False
        model_name = self._dynamic_load_model()
        logger.debug('Loading model %s', model_name)

        self._loader_lock.acquire()
        try:
            if not self._models.get(model_name):
                interpreter = self._interpreter_for_model(model_name)
                self._models[model_name] = interpreter
                status = True
        finally:
            self._loader_lock.release()

        self._end_read()

        return status

    def update(self, model_name):
        self._writer_lock.acquire()
        self._models[model_name] = None
        self._writer_lock.release()
        self.status = 0

    def unload(self, model_name):
        self._writer_lock.acquire()
        try:
            del self._models[model_name]
            self._models[model_name] = None
            return model_name
        finally:
            self._writer_lock.release()

    def _latest_project_model(self):
        """Retrieves the latest trained model for an project"""

        models = {model[len(MODEL_NAME_PREFIX):]: model
                  for model in self._models.keys()
                  if model.startswith(MODEL_NAME_PREFIX)}
        if models:
            time_list = [datetime.datetime.strptime(time, '%Y%m%d-%H%M%S')
                         for time, model in models.items()]
            return models[max(time_list).strftime('%Y%m%d-%H%M%S')]
        else:
            return FALLBACK_MODEL_NAME

    def _fallback_model(self):
        meta = Metadata({"pipeline": [{
            "name": "intent_classifier_keyword",
            "class": utils.module_path_from_object(KeywordIntentClassifier())
        }]}, "")
        return Interpreter.create(meta, self._component_builder)

    def _search_for_models(self):
        model_names = (self._list_models_in_dir(self._path) +
                       self._list_models_in_cloud())
        if not model_names:
            if FALLBACK_MODEL_NAME not in self._models:
                self._models[FALLBACK_MODEL_NAME] = self._fallback_model()
        else:
            for model in set(model_names):
                if model not in self._models:
                    self._models[model] = None

    def _interpreter_for_model(self, model_name):
        metadata = self._read_model_metadata(model_name)
        return Interpreter.create(metadata, self._component_builder)

    def _read_model_metadata(self, model_name):
        if model_name is None:
            data = Project._default_model_metadata()
            return Metadata(data, model_name)
        else:
            if not os.path.isabs(model_name) and self._path:
                path = os.path.join(self._path, model_name)
            else:
                path = model_name

            # download model from cloud storage if needed and possible
            if not os.path.isdir(path):
                self._load_model_from_cloud(model_name, path)

            return Metadata.load(path)

    def as_dict(self):
        return {'status': 'training' if self.status else 'ready',
                'available_models': list(self._models.keys()),
                'loaded_models': self._list_loaded_models()}

    def _list_loaded_models(self):
        models = []
        for model, interpreter in self._models.items():
            if interpreter is not None:
                models.append(model)
        return models

    def _list_models_in_cloud(self):
        # type: () -> List[Text]

        try:
            from rasa_nlu.persistor import get_persistor
            p = get_persistor(self.remote_storage)
            if p is not None:
                return p.list_models(self._project)
            else:
                return []
        except Exception as e:
            logger.warn("Failed to list models of project {}. "
                        "{}".format(self._project, e))
            return []

    def _load_model_from_cloud(self, model_name, target_path):
        try:
            from rasa_nlu.persistor import get_persistor
            p = get_persistor(self.remote_storage)
            if p is not None:
                p.retrieve(model_name, self._project, target_path)
            else:
                raise RuntimeError("Unable to initialize persistor")
        except Exception as e:
            logger.warn("Using default interpreter, couldn't fetch "
                        "model: {}".format(e))
            raise  # re-raise this exception because nothing we can do now

    @staticmethod
    def _default_model_metadata():
        return {
            "language": None,
        }

    @staticmethod
    def _list_models_in_dir(path):
        if not path or not os.path.isdir(path):
            return []
        else:
            return [os.path.relpath(model, path)
                    for model in utils.list_subdirectories(path)]
