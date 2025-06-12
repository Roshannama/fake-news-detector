from flask import Flask, request, render_template
import pickle
import joblib

import numpy as np
app = Flask(__name__)

with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    input_vec = vectorizer.transform([input_text])
    pred = model.predict(input_vec)
    prob = model.predict_proba(input_vec)

    label = 'FAKE' if pred[0] == 1 else 'REAL'
    confidence = float(np.max(prob))

    return render_template('index.html', prediction_text=f'{label} News (Confidence: {confidence:.2f})')

if __name__ == '__main__':
    app.run(debug=True)
