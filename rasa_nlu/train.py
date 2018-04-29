from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging

import typing
from typing import Optional, Any
from typing import Text
from typing import Tuple

from rasa_nlu import utils, config
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Trainer
from rasa_nlu.training_data import load_data
from rasa_nlu.training_data.loading import load_data_from_url

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_nlu.persistor import Persistor


def create_argument_parser():
    parser = argparse.ArgumentParser(
            description='train a custom language parser')

    parser.add_argument('-o', '--path',
                        default=None,
                        help="Path where model files will be saved")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-d', '--data',
                       default=None,
                       help="Location of the training data. For JSON and "
                            "markdown data, this can either be a single file "
                            "or a directory containing multiple training "
                            "data files.")

    group.add_argument('-u', '--url',
                       default=None,
                       help="URL from which to retrieve training data.")

    parser.add_argument('-c', '--config',
                        required=True,
                        help="Rasa NLU configuration file")

    parser.add_argument('-t', '--num_threads',
                        default=1,
                        type=int,
                        help="Number of threads to use during model training")

    parser.add_argument('--project',
                        default=None,
                        help="Project this model belongs to.")

    parser.add_argument('--fixed_model_name',
                        help="If present, a model will always be persisted "
                             "in the specified directory instead of creating "
                             "a folder like 'model_20171020-160213'")

    parser.add_argument('--storage',
                        help='Set the remote location where models are stored. '
                             'E.g. on AWS. If nothing is configured, the '
                             'server will only serve the models that are '
                             'on disk in the configured `path`.')

    utils.add_logging_option_arguments(parser)
    return parser


class TrainingException(Exception):
    """Exception wrapping lower level exceptions that may happen while training

      Attributes:
          failed_target_project -- name of the failed project
          message -- explanation of why the request is invalid
      """

    def __init__(self, failed_target_project=None, exception=None):
        self.failed_target_project = failed_target_project
        if exception:
            self.message = exception.args[0]

    def __str__(self):
        return self.message


def create_persistor(persistor):
    # type: (Optional[Text]) -> Optional[Persistor]
    """Create a remote persistor to store the model if configured."""

    if persistor is not None:
        from rasa_nlu.persistor import get_persistor
        return get_persistor(persistor)
    else:
        return None


def do_train_in_worker(config,  # type: RasaNLUModelConfig
                       data,  # type: Text
                       path,  # type: Text
                       project=None,  # type: Optional[Text]
                       fixed_model_name=None,  # type: Optional[Text]
                       storage=None,  # type: Text
                       component_builder=None
                       # type: Optional[ComponentBuilder]
                       ):
    # type: (...) -> Text
    """Loads the trainer and the data and runs the training in a worker."""

    try:
        _, _, persisted_path = do_train(config, data, path, project,
                                        fixed_model_name, storage,
                                        component_builder)
        return persisted_path
    except Exception as e:
        logger.exception("Failed to train project '{}'.".format(project))
        raise TrainingException(project, e)


def do_train(cfg,  # type: RasaNLUModelConfig
             data,  # type: Text
             path=None,  # type: Optional[Text]
             project=None,  # type: Optional[Text]
             fixed_model_name=None,  # type: Optional[Text]
             storage=None,  # type: Optional[Text]
             component_builder=None,  # type: Optional[ComponentBuilder]
             url=None,  # type: Optional[Text]
             **kwargs  # type: Any
             ):
    # type: (...) -> Tuple[Trainer, Interpreter, Text]
    """Loads the trainer and the data and runs the training of the model."""

    # Ensure we are training a model that we can save in the end
    # WARN: there is still a race condition if a model with the same name is
    # trained in another subprocess
    trainer = Trainer(cfg, component_builder)
    persistor = create_persistor(storage)
    if url is not None:
        training_data = load_data_from_url(url, cfg.language)
    else:
        training_data = load_data(data, cfg.language)
    interpreter = trainer.train(training_data, **kwargs)

    if path:
        persisted_path = trainer.persist(path,
                                         persistor,
                                         project,
                                         fixed_model_name)
    else:
        persisted_path = None

    return trainer, interpreter, persisted_path


if __name__ == '__main__':
    cmdline_args = create_argument_parser().parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    do_train(config.load(cmdline_args.config),
             cmdline_args.data,
             cmdline_args.path,
             cmdline_args.project,
             cmdline_args.fixed_model_name,
             cmdline_args.storage,
             url=cmdline_args.url,
             num_threads=cmdline_args.num_threads)
    logger.info("Finished training")
