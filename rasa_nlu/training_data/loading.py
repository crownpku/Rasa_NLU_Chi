from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging

import requests
from typing import Text, Optional

from rasa_nlu import utils
from rasa_nlu.training_data import TrainingData
from rasa_nlu.training_data.formats import (
    MarkdownReader, WitReader, LuisReader,
    RasaReader, DialogflowReader)
from rasa_nlu.training_data.formats import markdown
from rasa_nlu.training_data.formats.dialogflow import (
    DIALOGFLOW_AGENT, DIALOGFLOW_PACKAGE, DIALOGFLOW_INTENT,
    DIALOGFLOW_ENTITIES, DIALOGFLOW_ENTITY_ENTRIES, DIALOGFLOW_INTENT_EXAMPLES)

logger = logging.getLogger(__name__)

# Different supported file formats and their identifier
WIT = "wit"
LUIS = "luis"
RASA = "rasa_nlu"
UNK = "unk"
MARKDOWN = "md"
DIALOGFLOW_RELEVANT = {DIALOGFLOW_ENTITIES, DIALOGFLOW_INTENT}

_markdown_section_markers = ["## {}:".format(s)
                             for s in markdown.available_sections]
_json_format_heuristics = {
    WIT: lambda js, fn: "data" in js and isinstance(js.get("data"), list),
    LUIS: lambda js, fn: "luis_schema_version" in js,
    RASA: lambda js, fn: "rasa_nlu_data" in js,
    DIALOGFLOW_AGENT: lambda js, fn: "supportedLanguages" in js,
    DIALOGFLOW_PACKAGE: lambda js, fn: "version" in js and len(js) == 1,
    DIALOGFLOW_INTENT: lambda js, fn: "responses" in js,
    DIALOGFLOW_ENTITIES: lambda js, fn: "isEnum" in js,
    DIALOGFLOW_INTENT_EXAMPLES: lambda js, fn: "_usersays_" in fn,
    DIALOGFLOW_ENTITY_ENTRIES: lambda js, fn: "_entries_" in fn
}


def load_data(resource_name, language='en'):
    # type: (Text, Optional[Text]) -> TrainingData
    """Load training data from disk.

    Merges them if loaded from disk and multiple files are found."""

    files = utils.list_files(resource_name)
    data_sets = [_load(f, language) for f in files]
    data_sets = [ds for ds in data_sets if ds]
    if len(data_sets) == 0:
        return TrainingData()
    elif len(data_sets) == 1:
        return data_sets[0]
    else:
        return data_sets[0].merge(*data_sets[1:])


def load_data_from_url(url, language='en'):
    # type: (Text, Optional[Text]) -> TrainingData
    """Load training data from a URL."""

    if not utils.is_url(url):
        raise requests.exceptions.InvalidURL(url)
    try:
        response = requests.get(url)
        response.raise_for_status()
        temp_data_file = utils.create_temporary_file(response.content)
        return _load(temp_data_file, language)
    except Exception as e:
        logger.warning("Could not retrieve training data "
                       "from URL:\n{}".format(e))


def _reader_factory(fformat):
    """Generates the appropriate reader class based on the file format."""
    reader = None
    if fformat == LUIS:
        reader = LuisReader()
    elif fformat == WIT:
        reader = WitReader()
    elif fformat in DIALOGFLOW_RELEVANT:
        reader = DialogflowReader()
    elif fformat == RASA:
        reader = RasaReader()
    elif fformat == MARKDOWN:
        reader = MarkdownReader()
    return reader


def _load(filename, language='en'):
    """Loads a single training data file from disk."""

    fformat = _guess_format(filename)
    if fformat == UNK:
        raise ValueError("Unknown data format for file {}".format(filename))

    logger.info("Training data format of {} is {}".format(filename, fformat))
    reader = _reader_factory(fformat)

    if reader:
        return reader.read(filename, language=language, fformat=fformat)
    else:
        return None


def _guess_format(filename):
    # type: (Text) -> Text
    """Applies heuristics to guess the data format of a file."""
    guess = UNK
    content = utils.read_file(filename)
    try:
        js = json.loads(content)
    except ValueError:
        if any([marker in content for marker in _markdown_section_markers]):
            guess = MARKDOWN
    else:
        for fformat, format_heuristic in _json_format_heuristics.items():
            if format_heuristic(js, filename):
                guess = fformat
                break

    return guess
