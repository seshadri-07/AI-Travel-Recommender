from flask import Flask, render_template, request
from ai_travel_model import load_and_train_model
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load AI model
dataset, model, enc_country, enc_category, enc_cost = load_and_train_model("destinations.csv")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_country = request.form['country']
    user_category = request.form['category']
    user_budget = request.form['budget']

    try:
        # Encode inputs
        country_enc = enc_country.transform([user_country])[0]
        category_enc = enc_category.transform([user_category])[0]

        # Predict cost probabilities
        X_input = np.array([[country_enc, category_enc]])
        probs = model.predict_proba(X_input)[0]
        top_cost_index = np.argsort(-probs)[:3]
        top_cost_labels = enc_cost.inverse_transform(top_cost_index)
        top_probs = probs[top_cost_index]

        # Summary for display
        ai_summary = "\n".join([
            f"{label}: {round(prob*100, 2)}% confidence"
            for label, prob in zip(top_cost_labels, top_probs)
        ])

        # Filter matching destinations
        filtered = dataset[
            (dataset["Country"].str.lower() == user_country.lower()) &
            (dataset["Category"].str.lower() == user_category.lower()) &
            (dataset["Cost of Living"].str.lower() == user_budget.lower())
        ]

        destinations = filtered.head(5).to_dict(orient="records")

        return render_template("results.html", ai_summary=ai_summary, destinations=destinations)

    except Exception as e:
        return render_template("results.html", ai_summary=f"⚠️ Error: {e}", destinations=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
