
import numpy as np
import pandas as pd
import torch

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from scipy.special import expit
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed

def getLabels():
    df = pd.read_csv("intent_models\metrics.csv")
    labels = df.label.unique()
    return labels

labels = getLabels()
def getModels(labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    maxLabel = 12
    count = 0
    for label in labels:
        models[label] = DistilBertForSequenceClassification.from_pretrained(f'intent_models\{label}').to(device)
        count += 1
        if count == maxLabel:
            break
    return models
'''
def getModels(labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for label in labels:
        models[label] = DistilBertForSequenceClassification.from_pretrained(f'intent_models\{label}').to(device)
    return models
'''

def predict(model, tokenizer, text, label):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoding = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**encoding)
    prob = expit(outputs['logits'].cpu().detach().numpy()).tolist()[0][1]
    return prob


app = Flask(__name__)
models = getModels(labels)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

@app.route('/predict', methods=['POST'])
def predictAll():
    text = request.json['text']
    results = {}

    def run_model(model, text, label):
        return predict(model, tokenizer, text, label)

    with ThreadPoolExecutor(max_workers=23) as executor:
        futures = {executor.submit(run_model, model, text, label): label for label, model in models.items()}
        for future in as_completed(futures):
            label = futures[future]
            results[label] = future.result()

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=False)