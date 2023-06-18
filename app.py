#Programming Assignment 5
#Jordan James 1001879608
#CSE 6332-002
import gzip
from tensorflow.python.ops.numpy_ops import np_config
import os
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from numpy import dot
from numpy.linalg import norm
from tensorflow.keras.losses import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

app = Flask(__name__)

# Tensorflow Hub URL for Universal Sentence Encoder
MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

# KerasLayer
sentence_encoder_layer = hub.KerasLayer(MODEL_URL,
                                        input_shape=[],
                                        dtype=tf.string,
                                        trainable=False,
                                        name="use")
                                        

df = pd.read_csv("static/titles.csv")
df2 = pd.read_csv("static/titles2.csv")
df = pd.concat([df,df2], ignore_index=True)

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')


@app.route('/results', methods=['POST'])
def results():
    arg = request.form.get('name')
    emb = sentence_encoder_layer([arg])
    v = []
    for i in range(200):
        with gzip.open('static/embeds/'+str(i)+'.npy.gz', 'rb') as f:
            embeds = np.load(f)
            for j,e in enumerate(embeds):
                cos_sim = dot(emb, e)/(norm(emb)*norm(e))
                if cos_sim > .4:
                    if i < 124:
                        v.append((cos_sim,i*3645 + j))
                    else:
                        v.append((cos_sim,123*3645 + (i-123)*3644 + j + 1))
    results = []
    for b in sorted(v, reverse=True)[:200]:
        results.append(df['title'][b[1]])
    if len(results) > 0:
       return render_template('results.html', name = results)
    else:
       return redirect(url_for('index'))

if __name__ == '__main__':
   app.run()
