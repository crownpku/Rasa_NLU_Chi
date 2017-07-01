Rasa NLU for Chinese, a fork from RasaHQ/rasa_nlu.

Files you should have:

1. data/total_word_feature_extractor_chi.dat 
Trained from Chinese corpus by MITIE wordrep tools (takes 2-3 days for training)

2. data/examples/rasa/demo-rasa_chi.json
Should add as much examples as possible.

Usage:

1. Clone this project, and run
```
python setup.py install
```

2. Modify configuration. 
Currently for Chinese we have two pipelines:

Use MITIE+Jieba (config_mitie_chi.json):
["nlp_mitie", "tokenizer_jieba", "ner_jieba_mitie", "ner_synonyms", "intent_classifier_jieba_mitie"]

Use MITIE+Jieba+sklearn (config_mitie_sklearn_chi.json):
["nlp_mitie", "tokenizer_jieba", "ner_jieba_mitie", "ner_synonyms", "intent_featurizer_jieba_mitie", "intent_classifier_sklearn"]

3. Train model by running:
```
python -m rasa_nlu.train -c config_mitie_chi.json
```
or
```
python -m rasa_nlu.train -c config_mitie_sklearn_chi.json
```
This will save you model at /models

4. Run the rasa_nlu server:
```
python -m rasa_nlu.server -c config_mitie_chi.json --server_model_dirs=./model_20170701_mitie_chi
```
or
```
python -m rasa_nlu.server -c config_mitie_sklearn_chi.json --server_model_dirs=./model_20170701_mitie_sklearn_chi
```
Change the configure json file and model path to your own.

5. Open a new terminal and now you can curl results from the server, for example:

```
$ curl -XPOST localhost:5000/parse -d '{"q":"我发烧了该吃什么药？"}' | python -mjson.tool
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   364  100   326  100    38  98519  11483 --:--:-- --:--:-- --:--:--  106k
{
    "entities": [
        {
            "end": 3,
            "entity": "disease",
            "extractor": "ner_jieba_mitie",
            "start": 1,
            "value": "\u53d1\u70e7"
        }
    ],
    "intent": {
        "confidence": 0.02073156639321614,
        "name": "medical"
    },
    "text": "\u6211\u53d1\u70e7\u4e86\u8be5\u5403\u4ec0\u4e48\u836f\uff1f"
}
```


TO DO LIST:

1. ~~Add module to use FudanNLP or THULAC to replace spacy to support Chinese~~
Or still use MITIE but train with Chinese corpus for new word embeddings (looking for more Chinese corpus and clean)

Done

2. Modify Rasa configuration for Chinese

Done

3. Add jieba as Chinese tokenizor into pipeline (training demos, user messages)

Done

4. Improve intent classification module

See issues

