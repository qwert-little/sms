import numpy as np
import pandas as pd
import re
from flask import Flask, render_template, request
import pickle, json, random
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)









app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open("suggestions.json", 'r') as f:
    suggestions = json.load(f)

def analyseSentiment(resp):
    with open('count_vectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
    resp = cv.transform([resp])
    
    y_pred = model.predict(resp)
    pred = y_pred[0]
    for suggestion in suggestions['suggestions']:
            if suggestion['label'] == '1':
                spos = random.choice(suggestion['response'])
            else:
                sneu = random.choice(suggestion['response'])

    if pred==1:
        result = "Spam"
        sugg = spos
    else:
        result = "Not Spam"
        sugg = sneu

    return result, sugg

@app.route('/')
def home():
    return render_template("home.html")
@app.route('/result', methods=['GET', 'POST'])
def result():
    txt = request.form.get("txt")
    print(txt)
    if request.method=='POST':
        if txt!="":
            res =transform_text(txt)
            print(res)
            result, sugg = analyseSentiment(res)
            return render_template("result.html", result = result, sugg = sugg, txt=txt)
        else:
            return render_template('home.html')
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug = True)