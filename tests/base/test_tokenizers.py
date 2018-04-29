# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def test_whitespace():
    from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    tk = WhitespaceTokenizer()

    assert [t.text for t in tk.tokenize("Forecast for lunch")] == \
           ['Forecast', 'for', 'lunch']

    assert [t.offset for t in tk.tokenize("Forecast for lunch")] == \
           [0, 9, 13]

    assert [t.text for t in tk.tokenize("hey ńöñàśçií how're you?")] == \
           ['hey', 'ńöñàśçií', 'how\'re', 'you?']

    assert [t.offset for t in tk.tokenize("hey ńöñàśçií how're you?")] == \
           [0, 4, 13, 20]


def test_spacy(spacy_nlp):
    from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
    tk = SpacyTokenizer()

    text = "Forecast for lunch"
    assert [t.text for t in tk.tokenize(spacy_nlp(text))] == \
           ['Forecast', 'for', 'lunch']
    assert [t.offset for t in tk.tokenize(spacy_nlp(text))] == \
           [0, 9, 13]

    text = "hey ńöñàśçií how're you?"
    assert [t.text for t in tk.tokenize(spacy_nlp(text))] == \
           ['hey', 'ńöñàśçií', 'how', '\'re', 'you', '?']
    assert [t.offset for t in tk.tokenize(spacy_nlp(text))] == \
           [0, 4, 13, 16, 20, 23]


def test_mitie():
    from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer
    tk = MitieTokenizer()

    text = "Forecast for lunch"
    assert [t.text for t in tk.tokenize(text)] == \
           ['Forecast', 'for', 'lunch']
    assert [t.offset for t in tk.tokenize(text)] == \
           [0, 9, 13]

    text = "hey ńöñàśçií how're you?"
    assert [t.text for t in tk.tokenize(text)] == \
           ['hey', 'ńöñàśçií', 'how', '\'re', 'you', '?']
    assert [t.offset for t in tk.tokenize(text)] == \
           [0, 4, 13, 16, 20, 23]


def test_jieba():
    from rasa_nlu.tokenizers.jieba_tokenizer import JiebaTokenizer
    tk = JiebaTokenizer()

    assert [t.text for t in tk.tokenize("我想去吃兰州拉面")] == \
           ['我', '想', '去', '吃', '兰州', '拉面']

    assert [t.offset for t in tk.tokenize("我想去吃兰州拉面")] == \
           [0, 1, 2, 3, 4, 6]

    assert [t.text for t in tk.tokenize("Micheal你好吗？")] == \
           ['Micheal', '你好', '吗', '？']

    assert [t.offset for t in tk.tokenize("Micheal你好吗？")] == \
           [0, 7, 9, 10]
