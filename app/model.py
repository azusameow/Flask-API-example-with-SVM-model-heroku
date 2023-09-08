# -*- coding: UTF-8 -*-
import pickle
import gzip

# 載入Model
with gzip.open('./model/svm.pgz', 'rb') as f:
    SvmModel = pickle.load(f)

def predict(input):
    pred = SvmModel.predict_proba([input])[0][1]  # Assuming pred is a string result from the model
    float_pred = float(pred)  # Convert to float if necessary
    return float_pred
