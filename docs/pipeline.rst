.. _section_pipeline:

Processing Pipeline
===================
The process of incoming messages is split into different components. These components are executed one after another
in a so called processing pipeline. There are components for entity extraction, for intent classification,
pre-processing and there will be many more in the future.

Each component processes the input and creates an output. The ouput can be used by any component that comes after
this component in the pipeline. There are components which only produce information that is used by other components
in the pipeline and there are other components that produce ``Output`` attributes which will be returned after
the processing has finished. For example, for the sentence ``"I am looking for Chinese food"`` the output

.. code-block:: json

    {
        "text": "I am looking for Chinese food",
        "entities": [
            {"start": 8, "end": 15, "value": "chinese", "entity": "cuisine", "extractor": "ner_crf", "confidence": 0.864}
        ],
        "intent": {"confidence": 0.6485910906220309, "name": "restaurant_search"},
        "intent_ranking": [
            {"confidence": 0.6485910906220309, "name": "restaurant_search"},
            {"confidence": 0.1416153159565678, "name": "affirm"}
        ]
    }

is created as a combination of the results of the different components in the pre-configured pipeline ``spacy_sklearn``.
For example, the ``entities`` attribute is created by the ``ner_crf`` component.

Pre-configured Pipelines
------------------------
To ease the burden of coming up with your own processing pipelines, we provide a couple of ready to use templates
which can be used by setting the ``pipeline`` configuration value to the name of the template you want to use.
Here is a list of the **existing templates**:

spacy_sklearn
~~~~~~~~~~~~~

To use spacy as a template:

.. literalinclude:: ../sample_configs/config_spacy.yml
    :language: yaml

See :ref:`section_languages` for possible values for ``language``. To use
the components and configure them separately:

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "nlp_spacy"
    - name: "tokenizer_spacy"
    - name: "intent_entity_featurizer_regex"
    - name: "intent_featurizer_spacy"
    - name: "ner_crf"
    - name: "ner_synonyms"
    - name: "intent_classifier_sklearn"

mitie
~~~~~

There is no pipeline template, as you need to configure the location
of mities featurizer. To use the components and configure them separately:

.. literalinclude:: ../sample_configs/config_mitie.yml
    :language: yaml

mitie_sklearn
~~~~~~~~~~~~~

There is no pipeline template, as you need to configure the location
of mities featurizer. To use the components and configure them separately:

.. literalinclude:: ../sample_configs/config_mitie_sklearn.yml
    :language: yaml

keyword
~~~~~~~

to use it as a template:

.. code-block:: yaml

    language: "en"

    pipeline: "keyword"

to use the components and configure them separately:

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "intent_classifier_keyword"


tensorflow_embedding
~~~~~~~~~~~~~~~~~~~~

to use it as a template:

.. code-block:: yaml

    language: "en"

    pipeline: "tensorflow_embedding"

The tensorflow pipeline supports any language that can be tokenized. The
current tokenizer implementation relies on words being separated by spaces,
so any languages that adheres to that can be trained with this pipeline.

If you want to split intents into multiple labels, e.g. for predicting multiple intents or for modeling hierarchical intent structure, use these flags:

    - ``intent_tokenization_flag`` if ``true`` the algorithm will split the intent labels into tokens and use bag-of-words representations for them;
    - ``intent_split_symbol`` sets the delimiter string to split the intent labels. Default ``_``


Here's an example configuration:

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "intent_featurizer_count_vectors"
    - name: "intent_classifier_tensorflow_embedding"
      intent_tokenization_flag: true
      intent_split_symbol: "_"



Custom pipelines
~~~~~~~~~~~~~~~~

Creating your own pipelines is possible by directly passing the names of the ~
components to Rasa NLU in the ``pipeline`` configuration variable, e.g.

.. code-block:: yaml

    pipeline:
    - name: "nlp_spacy"
    - name: "ner_crf"
    - name: "ner_synonyms"

This creates a pipeline that only does entity recognition, but no
intent classification. Hence, the output will not contain any
useful intents.

Built-in Components
-------------------

Short explanation of every components and it's attributes. If you are looking for more details, you should have
a look at the corresponding source code for the component. ``Output`` describes, what each component adds to the final
output result of processing a message. If no output is present, the component is most likely a preprocessor for another
component.

.. _nlp_mitie:

nlp_mitie
~~~~~~~~~

:Short: MITIE initializer
:Outputs: nothing
:Description:
    Initializes mitie structures. Every mitie component relies on this, hence this should be put at the beginning
    of every pipeline that uses any mitie components.
:Configuration:
    The MITIE library needs a language model file, that **must** be specified in
    the configuration:

    .. code-block:: yaml

        pipeline:
        - name: "nlp_mitie"
          # language model to load
          model: "data/total_word_feature_extractor.dat"

    For more information where to get that file from, head over to
    :ref:`section_backends`.

nlp_spacy
~~~~~~~~~

:Short: spacy language initializer
:Outputs: nothing
:Description:
    Initializes spacy structures. Every spacy component relies on this, hence this should be put at the beginning
    of every pipeline that uses any spacy components.
:Configuration:
    Language model, default will use the configured language.
    If the spacy model to be used has a name that is different from the language tag (``"en"``, ``"de"``, etc.),
    the model name can be specified using this configuration variable. The name will be passed to ``spacy.load(name)``.

    .. code-block:: yaml

        pipeline:
        - name: "nlp_spacy"
          # language model to load
          model: "en_core_web_md"

          # when retrieving word vectors, this will decide if the casing
          # of the word is relevant. E.g. `hello` and `Hello` will
          # retrieve the same vector, if set to `false`. For some
          # applications and models it makes sense to differentiate
          # between these two words, therefore setting this to `true`.
          case_sensitive: false


intent_featurizer_mitie
~~~~~~~~~~~~~~~~~~~~~~~

:Short: MITIE intent featurizer
:Outputs: nothing, used as an input to intent classifiers that need intent features (e.g. ``intent_classifier_sklearn``)
:Description:
    Creates feature for intent classification using the MITIE featurizer.

    .. note::

        NOT used by the ``intent_classifier_mitie`` component. Currently, only ``intent_classifier_sklearn`` is able
        to use precomputed features.

:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "intent_featurizer_mitie"



intent_featurizer_spacy
~~~~~~~~~~~~~~~~~~~~~~~

:Short: spacy intent featurizer
:Outputs: nothing, used as an input to intent classifiers that need intent features (e.g. ``intent_classifier_sklearn``)
:Description:
    Creates feature for intent classification using the spacy featurizer.

intent_featurizer_ngrams
~~~~~~~~~~~~~~~~~~~~~~~~

:Short: Appends char-ngram features to feature vector
:Outputs: nothing, appends its features to an existing feature vector generated by another intent featurizer
:Description:
    This featurizer appends character ngram features to a feature vector. During training the component looks for the
    most common character sequences (e.g. ``app`` or ``ing``). The added features represent a boolean flag if the
    character sequence is present in the word sequence or not.

    .. note:: There needs to be another intent featurizer previous to this one in the pipeline!

:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "intent_featurizer_ngrams"
          # Maximum number of ngrams to use when augmenting
          # feature vectors with character ngrams
          max_number_of_ngrams: 10

intent_featurizer_count_vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Short: Creates bag-of-words representation of intent features
:Outputs: nothing, used as an input to intent classifiers that need bag-of-words representation of intent features (e.g. ``intent_classifier_tensorflow_embedding``)
:Description:
    Creates bag-of-words representation of intent features using
    `sklearn's CountVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_. All tokens which consist only of digits (e.g. 123 and 99 but not a123d) will be assigned to the same feature.

:Configuration:
    See `sklearn's CountVectorizer docs <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_
    for detailed description of the configuration parameters

    .. code-block:: yaml

        pipeline:
        - name: "intent_featurizer_count_vectors"
          # the parameters are taken from
          # sklearn's CountVectorizer
          # regular expression for tokens
          "token_pattern": r'(?u)\b\w\w+\b'
          # remove accents during the preprocessing step
          "strip_accents": None  # {'ascii', 'unicode', None}
          # list of stop words
          "stop_words": None  # string {'english'}, list, or None (default)
          # min document frequency of a word to add to vocabulary
          # float - the parameter represents a proportion of documents
          # integer - absolute counts
          "min_df": 1  # float in range [0.0, 1.0] or int
          # max document frequency of a word to add to vocabulary
          # float - the parameter represents a proportion of documents
          # integer - absolute counts
          "max_df": 1.0  # float in range [0.0, 1.0] or int
          # set ngram range
          "min_ngram": 1
          "max_ngram": 1
          # limit vocabulary size
          "max_features": None

intent_classifier_keyword
~~~~~~~~~~~~~~~~~~~~~~~~~

:Short: Simple keyword matching intent classifier.
:Outputs: ``intent``
:Output-Example:

    .. code-block:: json

        {
            "intent": {"name": "greet", "confidence": 0.98343}
        }

:Description:
    This classifier is mostly used as a placeholder. It is able to recognize `hello` and
    `goodbye` intents by searching for these keywords in the passed messages.

intent_classifier_mitie
~~~~~~~~~~~~~~~~~~~~~~~

:Short: MITIE intent classifier (using a `text categorizer <https://github.com/mit-nlp/MITIE/blob/master/examples/python/text_categorizer_pure_model.py>`_)
:Outputs: ``intent``
:Output-Example:

    .. code-block:: json

        {
            "intent": {"name": "greet", "confidence": 0.98343}
        }

:Description:
    This classifier uses MITIE to perform intent classification. The underlying classifier
    is using a multi class linear SVM with a sparse linear kernel (see `mitie trainer code <https://github.com/mit-nlp/MITIE/blob/master/mitielib/src/text_categorizer_trainer.cpp#L222>`_).

:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "intent_classifier_mitie"

intent_classifier_sklearn
~~~~~~~~~~~~~~~~~~~~~~~~~

:Short: sklearn intent classifier
:Outputs: ``intent`` and ``intent_ranking``
:Output-Example:

    .. code-block:: json

        {
            "intent": {"name": "greet", "confidence": 0.78343},
            "intent_ranking": [
                {
                    "confidence": 0.1485910906220309,
                    "name": "goodbye"
                },
                {
                    "confidence": 0.08161531595656784,
                    "name": "restaurant_search"
                }
            ]
        }

:Description:
    The sklearn intent classifier trains a linear SVM which gets optimized using a grid search. In addition
    to other classifiers it also provides rankings of the labels that did not "win". The spacy intent classifier
    needs to be preceded by a featurizer in the pipeline. This featurizer creates the features used for the classification.

:Configuration:
    During the training of the SVM a hyperparameter search is run to
    find the best parameter set. In the config, you can specify the parameters
    that will get tried

    .. code-block:: yaml

        pipeline:
        - name: "intent_classifier_sklearn"
          # Specifies the list of regularization values to
          # cross-validate over for C-SVM.
          # This is used with the ``kernel`` hyperparameter in GridSearchCV.
          C: [1, 2, 5, 10, 20, 100]
          # Specifies the kernel to use with C-SVM.
          # This is used with the ``C`` hyperparameter in GridSearchCV.
          kernels: ["linear"]

intent_classifier_tensorflow_embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Short: Embedding intent classifier
:Outputs: ``intent`` and ``intent_ranking``
:Output-Example:

    .. code-block:: json

        {
            "intent": {"name": "greet", "confidence": 0.8343},
            "intent_ranking": [
                {
                    "confidence": 0.385910906220309,
                    "name": "goodbye"
                },
                {
                    "confidence": 0.28161531595656784,
                    "name": "restaurant_search"
                }
            ]
        }

:Description:
    The embedding intent classifier embeds user inputs and intent labels into the same space. Supervised embeddings are
    trained by maximizing similarity between them. This algorithm is based on
    the starspace idea from: `<https://arxiv.org/abs/1709.03856>`_. However, in this implementation
    the ``mu`` parameter is treated differently and additional hidden layers are added together with dropout.
    This algorithm also provides similarity rankings of the labels that did not "win".

    The embedding intent classifier needs to be preceded by a featurizer in the pipeline.
    This featurizer creates the features used for the embeddings.
    It is recommended to use ``intent_featurizer_count_vectors`` that can be optionally preceded
    by ``nlp_spacy`` and ``tokenizer_spacy``.

:Configuration:
    If you want to split intents into multiple labels, e.g. for predicting multiple intents or for
    modeling hierarchical intent structure, use these flags:

    - tokenization of intent labels:
        - ``intent_tokenization_flag`` if ``true`` the algorithm will split the intent labels into tokens and use bag-of-words representations for them;
        - ``intent_split_symbol`` sets the delimiter string to split the intent labels. Default ``_``


    The algorithm also has hyperparameters to control:
        - neural network's architecture:
            - ``num_hidden_layers_a`` and ``hidden_layer_size_a`` set the number of hidden layers and their sizes before embedding layer for user inputs;
            - ``num_hidden_layers_b`` and ``hidden_layer_size_b`` set the number of hidden layers and their sizes before embedding layer for intent labels;
        - training:
            - ``batch_size`` sets the number of training examples in one forward/backward pass, the higher the batch size, the more memory space you'll need;
            - ``epochs`` sets the number of times the algorithm will see training data, where ``one epoch`` = one forward pass and one backward pass of all the training examples;
        - embedding:
            - ``embed_dim`` sets the dimension of embedding space;
            - ``mu_pos`` controls how similar the algorithm should try to make embedding vectors for correct intent labels;
            - ``mu_neg`` controls maximum negative similarity for incorrect intents;
            - ``similarity_type`` sets the type of the similarity, it should be either ``cosine`` or ``inner``;
            - ``num_neg`` sets the number of incorrect intent labels, the algorithm will minimize their similarity to the user input during training;
            - ``use_max_sim_neg`` if ``true`` the algorithm only minimizes maximum similarity over incorrect intent labels;
        - regularization:
            - ``C2`` sets the scale of L2 regularization
            - ``C_emb`` sets the scale of how important is to minimize the maximum similarity between embeddings of different intent labels;
            - ``droprate`` sets the dropout rate, it should be between ``0`` and ``1``, e.g. ``droprate=0.1`` would drop out ``10%`` of input units;

    .. note:: For ``cosine`` similarity ``mu_pos`` and ``mu_neg`` should be between ``-1`` and ``1``.

    In the config, you can specify these parameters:

    .. code-block:: yaml

        pipeline:
        - name: "intent_classifier_tensorflow_embedding"
          # nn architecture
          "num_hidden_layers_a": 2
          "hidden_layer_size_a": [256, 128]
          "num_hidden_layers_b": 0
          "hidden_layer_size_b": []
          "batch_size": 32
          "epochs": 300
          # embedding parameters
          "embed_dim": 10
          "mu_pos": 0.8  # should be 0.0 < ... < 1.0 for 'cosine'
          "mu_neg": -0.4  # should be -1.0 < ... < 1.0 for 'cosine'
          "similarity_type": "cosine"  # string 'cosine' or 'inner'
          "num_neg": 10
          "use_max_sim_neg": true  # flag which loss function to use
          # regularization
          "C2": 0.002
          "C_emb": 0.8
          "droprate": 0.2
          # flag if to tokenize intents
          "intent_tokenization_flag": false
          "intent_split_symbol": "_"

    .. note:: Parameter ``mu_neg`` is set to a negative value to mimic the original
              starspace algorithm in the case ``mu_neg = mu_pos`` and ``use_max_sim_neg = False``.
              See `starspace paper <https://arxiv.org/abs/1709.03856>`_ for details.

intent_entity_featurizer_regex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Short: regex feature creation to support intent and entity classification
:Outputs: ``text_features`` and ``tokens.pattern``
:Description:
    During training, the regex intent featurizer creates a list of `regular expressions` defined in the training data format.
    If an expression is found in the input, a feature will be set, that will later be fed into intent classifier / entity
    extractor to simplify classification (assuming the classifier has learned during the training phase, that this set
    feature indicates a certain intent). Regex features for entity extraction are currently only supported by the
    ``ner_crf`` component!

tokenizer_whitespace
~~~~~~~~~~~~~~~~~~~~

:Short: Tokenizer using whitespaces as a separator
:Outputs: nothing
:Description:
    Creates a token for every whitespace separated character sequence. Can be used to define tokens for the MITIE entity
    extractor.
                                                                   
tokenizer_jieba
~~~~~~~~~~~~~~~~~~~~

:Short: Tokenizer using Jieba for Chinese language
:Outputs: nothing
:Description:
    Creates tokens using the Jieba tokenizer specifically for Chinese
    language. For language other than Chinese, Jieba will work as
    ``tokenizer_whitespace``. Can be used to define tokens for the
    MITIE entity extractor. Make sure to install Jieba, ``pip install jieba``.
:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "tokenizer_jieba"

tokenizer_mitie
~~~~~~~~~~~~~~~

:Short: Tokenizer using MITIE
:Outputs: nothing
:Description:
    Creates tokens using the MITIE tokenizer. Can be used to define
    tokens for the MITIE entity extractor.
:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "tokenizer_mitie"

tokenizer_spacy
~~~~~~~~~~~~~~~

:Short: Tokenizer using spacy
:Outputs: nothing
:Description:
    Creates tokens using the spacy tokenizer. Can be used to define
    tokens for the MITIE entity extractor.


ner_mitie
~~~~~~~~~

:Short: MITIE entity extraction (using a `mitie ner trainer <https://github.com/mit-nlp/MITIE/blob/master/mitielib/src/ner_trainer.cpp>`_)
:Outputs: appends ``entities``
:Output-Example:

    .. code-block:: json

        {
            "entities": [{"value": "New York City",
                          "start": 20,
                          "end": 33,
                          "confidence": null,
                          "entity": "city",
                          "extractor": "ner_mitie"}]
        }

:Description:
    This uses the MITIE entitiy extraction to find entities in a message. The underlying classifier
    is using a multi class linear SVM with a sparse linear kernel and custom features.
    The MITIE component does not provide entity confidence values.
:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "ner_mitie"

ner_spacy
~~~~~~~~~

:Short: spacy entity extraction
:Outputs: appends ``entities``
:Output-Example:

    .. code-block:: json

        {
            "entities": [{"value": "New York City",
                          "start": 20,
                          "end": 33,
                          "entity": "city",
                          "confidence": null,
                          "extractor": "ner_spacy"}]
        }

:Description:
    Using spacy this component predicts the entities of a message. spacy uses a statistical BILUO transition model.
    As of now, this component can only use the spacy builtin entity extraction models and can not be retrained.
    This extractor does not provide any confidence scores.

ner_synonyms
~~~~~~~~~~~~

:Short: Maps synonymous entity values to the same value.
:Outputs: modifies existing entities that previous entity extraction components found

:Description:
    If the training data contains defined synonyms (by using the ``value`` attribute on the entity examples).
    this component will make sure that detected entity values will be mapped to the same value. For example,
    if your training data contains the following examples:

    .. code-block:: json

        [{
          "text": "I moved to New York City",
          "intent": "inform_relocation",
          "entities": [{"value": "nyc",
                        "start": 11,
                        "end": 24,
                        "entity": "city",
                       }]
        },
        {
          "text": "I got a new flat in NYC.",
          "intent": "inform_relocation",
          "entities": [{"value": "nyc",
                        "start": 20,
                        "end": 23,
                        "entity": "city",
                       }]
        }]

    this component will allow you to map the entities ``New York City`` and ``NYC`` to ``nyc``. The entitiy
    extraction will return ``nyc`` even though the message contains ``NYC``. When this component changes an
    exisiting entity, it appends itself to the processor list of this entity.

ner_crf
~~~~~~~

:Short: conditional random field entity extraction
:Outputs: appends ``entities``
:Output-Example:

    .. code-block:: json

        {
            "entities": [{"value":"New York City",
                          "start": 20,
                          "end": 33,
                          "entity": "city",
                          "confidence": 0.874,
                          "extractor": "ner_crf"}]
        }

:Description:
    This component implements conditional random fields to do named entity recognition.
    CRFs can be thought of as an undirected Markov chain where the time steps are words
    and the states are entity classes. Features of the words (capitalisation, POS tagging,
    etc.) give probabilities to certain entity classes, as are transitions between
    neighbouring entity tags: the most likely set of tags is then calculated and returned.
:Configuration:
   .. code-block:: yaml

        pipeline:
        - name: "ner_crf"
          # The features are a ``[before, word, after]`` array with
          # before, word, after holding keys about which
          # features to use for each word, for example, ``"title"``
          # in array before will have the feature
          # "is the preceding word in title case?".
          # Available features are:
          # ``low``, ``title``, ``word3``, ``word2``, ``pos``,
          # ``pos2``, ``bias``, ``upper`` and ``digit``
          features: [["low", "title"], ["bias", "word3"], ["upper", "pos", "pos2"]]

          # The flag determines whether to use BILOU tagging or not. BILOU
          # tagging is more rigorous however
          # requires more examples per entity. Rule of thumb: use only
          # if more than 100 examples per entity.
          BILOU_flag: true

          # This is the value given to sklearn_crfcuite.CRF tagger before training.
          max_iterations: 50

          # This is the value given to sklearn_crfcuite.CRF tagger before training.
          # Specifies the L1 regularization coefficient.
          L1_c: 1.0

          # This is the value given to sklearn_crfcuite.CRF tagger before training.
          # Specifies the L2 regularization coefficient.
          L2_c: 1e-3

.. _section_pipeline_duckling:

ner_duckling
~~~~~~~~~~~~
:Short: Adds duckling support to the pipeline to unify entity types (e.g. to retrieve common date / number formats)
:Outputs: appends ``entities``
:Output-Example:

    .. code-block:: json

        {
            "entities": [{"end": 53,
                          "entity": "time",
                          "start": 48,
                          "value": "2017-04-10T00:00:00.000+02:00",
                          "confidence": 1.0,
                          "extractor": "ner_duckling"}]
        }

:Description:
    Duckling allows to recognize dates, numbers, distances and other structured entities
    and normalizes them (for a reference of all available entities
    see `the duckling documentation <https://duckling.wit.ai/#getting-started>`_).
    Please be aware that duckling tries to extract as many entity types as possible without
    providing a ranking. For example, if you specify both ``number`` and ``time`` as dimensions
    for the duckling component, the component will extract two entities: ``10`` as a number and
    ``in 10 minutes`` as a time from the text ``I will be there in 10 minutes``. In such a
    situation, your application would have to decide which entity type is be the correct one.
    The extractor will always return `1.0` as a confidence, as it is a rule
    based system.

:Configuration:
    Configure which dimensions, i.e. entity types, the :ref:`duckling component <section_pipeline_duckling>` to extract.
    A full list of available dimensions can be found in the `duckling documentation <https://duckling.wit.ai/>`_.

    .. code-block:: yaml

        pipeline:
        - name: "ner_duckling"
          # dimensions to extract
          dimensions: ["time", "number", "amount-of-money", "distance"]



Creating new Components
-----------------------
You can create a custom Component to perform a specific task which NLU doesn't currently offer (e.g. sentiment analysis).
A glimpse into the code of ``rasa_nlu.components.Component`` will reveal
which functions need to be implemented to create a new component.
You can add these to your pipeline by adding the module path to your pipeline, e.g. if you have a module called ``sentiment``
containing a ``SentimentAnalyzer`` class:

    .. code-block:: yaml

        pipeline:
        - name: "sentiment.SentimentAnalyzer"


Component Lifecycle
-------------------
Every component can implement several methods from the ``Component`` base class; in a pipeline these different methods
will be called in a specific order. Lets assume, we added the following pipeline to our config:
``"pipeline": ["Component A", "Component B", "Last Component"]``.
The image shows the call order during the training of this pipeline :

.. image:: _static/images/component_lifecycle.png

Before the first component is created using the ``create`` function, a so called ``context`` is created (which is
nothing more than a python dict). This context is used to pass information between the components. For example,
one component can calculate feature vectors for the training data, store that within the context and another
component can retrieve these feature vectors from the context and do intent classification.

Initially the context is filled with all configuration values, the arrows in the image show the call order
and visualize the path of the passed context. After all components are trained and persisted, the
final context dictionary is used to persist the model's metadata.
