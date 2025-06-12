from flask import Flask, request, render_template
import pickle
import numpy as np
import os


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
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
