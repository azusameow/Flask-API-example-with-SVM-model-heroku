# -*- coding: UTF-8 -*-
import numpy as np
import word2vec
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
CORS(app)
run_with_ngrok(app)
@app.route('/')
def index():
    return 'hello!!'

@app.route('/predict', methods=['POST'])
def postInput():
    # 取得前端傳過來的數值
    insertValues = request.get_json()
    x1=insertValues['sentence']

    result = word2vec.analyze_text(x1)
    return jsonify({'return': float(result)})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=3000,debug=True)