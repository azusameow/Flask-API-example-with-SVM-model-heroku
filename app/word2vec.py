from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import jieba
from sklearn.metrics import accuracy_score
import pandas as pd
from gensim.models import Word2Vec
import jieba
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from typing import List
import speech_recognition as sr
import jieba
import zhconv


#1 簡繁轉換 停用詞下載
spacy.cli.download("zh_core_web_sm")  # 下載 spacy 中文模組
spacy.cli.download("en_core_web_sm")  # 下載 spacy 英文模組

nlp_zh = spacy.load("zh_core_web_sm") # 載入 spacy 中文模組
nlp_en = spacy.load("en_core_web_sm") # 載入 spacy 英文模組
print('--\n')
print(f"中文停用詞 Total={len(nlp_zh.Defaults.stop_words)}: {list(nlp_zh.Defaults.stop_words)[:20]} ...")
STOPWORDS =  nlp_zh.Defaults.stop_words | \
             nlp_en.Defaults.stop_words | \
             set(["\n", "\r\n", "\t", " ", ""])

# 將簡體停用詞轉成繁體，擴充停用詞表
for word in STOPWORDS.copy():
    STOPWORDS.add(zhconv.convert(word, "zh-tw"))

filepath = './model/詐騙手法.csv'
data = pd.read_csv(filepath)
filepath1 = './model/自訂義辭典.txt'
jieba.load_userdict(filepath1)
def preprocess_text(text):
    words = jieba.lcut(text)
    words = ' '.join(words)
    processed_text = re.sub(r'[^\w\s]', '', words)
    return processed_text
def sentence_to_vec(sentence, model):
    vectors = []
    for word in sentence:
        if word in model:
            vectors.append(model[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)  # 如果沒有有效的單詞向量，則返回零向量

    
def preprocess_and_tokenize(
    text_list: List[str], token_min_len: int = 1, token_max_len: int = 15, lower: bool = True
) -> List[str]:
    preprocessed_texts = []
    for text in text_list:
        if lower:
            text = text.lower()
        text = zhconv.convert(text, "zh-tw")
        tokens = [
            token for token in jieba.cut(text, cut_all=False)
            if token_min_len <= len(token) <= token_max_len and token not in STOPWORDS
        ]
        preprocessed_texts.extend(tokens)  # 使用extend代替append，将内部的列表添加到外部列表中
    return preprocessed_texts

texts = [preprocess_and_tokenize(text) for text in data['關鍵字']]
filepath2 = './model/tmunlp_1.6B_WB_100dim_2020v1.bin.gz'

model = gensim.models.KeyedVectors.load_word2vec_format(filepath2, 
                                                        unicode_errors='ignore', 
                                                        binary=True)

    # -*- coding: UTF-8 -*-
import pickle
import gzip

# 載入Model
with gzip.open('./model/svm.pgz', 'rb') as f:
    SvmModel = pickle.load(f)

def analyze_text(text):
    preprocessed_texts = preprocess_and_tokenize([text])
    print(preprocessed_texts)
    preprocessed_vector = sentence_to_vec(preprocessed_texts, model)  # 获取单个句子向量
    prediction_proba = SvmModel.predict_proba([preprocessed_vector])[0][1]  # 注意传入列表形式
    return prediction_proba