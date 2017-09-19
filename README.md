

# Rasa NLU for Chinese, a fork from RasaHQ/rasa_nlu.

## [中文Blog](http://www.crownpku.com/2017/07/27/%E7%94%A8Rasa_NLU%E6%9E%84%E5%BB%BA%E8%87%AA%E5%B7%B1%E7%9A%84%E4%B8%AD%E6%96%87NLU%E7%B3%BB%E7%BB%9F.html)

![](http://www.crownpku.com/images/201707/5.jpg)
![](http://www.crownpku.com/images/201707/4.jpg)



### Files you should have:

* data/total_word_feature_extractor_zh.dat

Trained from Chinese corpus by MITIE wordrep tools (takes 2-3 days for training)


For training, please build the [MITIE Wordrep Tool](https://github.com/mit-nlp/MITIE/tree/master/tools/wordrep). Note that Chinese corpus should be tokenized first before feeding into the tool for training. Close-domain corpus that best matches user case works best.

A trained model from Chinese Wikipedia Dump and Baidu Baike can be downloaded from [中文Blog](http://www.crownpku.com/2017/07/27/%E7%94%A8Rasa_NLU%E6%9E%84%E5%BB%BA%E8%87%AA%E5%B7%B1%E7%9A%84%E4%B8%AD%E6%96%87NLU%E7%B3%BB%E7%BB%9F.html).


* data/examples/rasa/demo-rasa_zh.json

Should add as much examples as possible.

### Usage:


1. Clone this project, and run
```
python setup.py install
```

2. Modify configuration. 

Currently for Chinese we have two pipelines:

Use MITIE+Jieba (sample_configs/config_jieba_mitie.json):

["nlp_mitie", "tokenizer_jieba", "ner_mitie", "ner_synonyms", "intent_classifier_mitie"]

RECOMMENDED: Use MITIE+Jieba+sklearn (sample_configs/config_jieba_mitie_sklearn.json):

["nlp_mitie", "tokenizer_jieba", "ner_mitie", "ner_synonyms", "intent_featurizer_mitie", "intent_classifier_sklearn"]


3. Train model by running:
```
python -m rasa_nlu.train -c sample_configs/config_jieba_mitie.json
```
or
```
python -m rasa_nlu.train -c sample_configs/config_jieba_mitie_sklearn.json
```
This will save your model at /models


4. Run the rasa_nlu server:
```
python -m rasa_nlu.server -c sample_configs/config_jieba_mitie.json -p ./model_20170701_mitie_chi
```
or
```
python -m rasa_nlu.server -c sample_configs/config_jieba_mitie_sklearn.json -p ./model_20170701_mitie_sklearn_chi
```
Change the configure json file and model path to your own. If no model path is used (-p) then the most recent trained model will be used.


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
