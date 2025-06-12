 Fake News Detector 
A web application that detects whether a news article is FAKE or REAL using an XGBoost machine learning model and a TF-IDF vectorizer. Built using Flask and deployed on Render


 Features:-
 Trained XGBoost model on labeled Fake/Real news dataset

 TF-IDF vectorizer for text processing

 Real-time prediction from user input

 Fully responsive UI with Tailwind CSS

 Deployable on Render in one click


 

 | Component        | Technology            |
| ---------------- | --------------------- |
| Backend          | Python, Flask         |
| Machine Learning | XGBoost, Scikit-learn |
| NLP              | TF-IDF Vectorizer     |
| Frontend         | HTML, Tailwind CSS    |
| Deployment       | Render                |




├── app.py                  # Flask application
├── templates/
│   └── index.html          # Web interface
├── xgb_model.pkl           # Trained ML model
├── tfidf_vectorizer.pkl    # Trained vectorizer
├── requirements.txt        # Dependencies
├── render.yaml             # Render deployment config
└── README.md               # You are here :)


Dataset:-
Fake and Real News Dataset  on kaggle

Contributing
Pull requests are welcome! If you have feature requests or ideas, feel free to open an issue

