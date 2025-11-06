from flask import Flask, render_template, request
from ai_travel_model import load_and_train_model
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load AI model and encoders
dataset, model, enc_country, enc_category, enc_cost = load_and_train_model("destinations.csv")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # ✅ Convert all inputs to lowercase for case-insensitive matching
    user_country = request.form['country'].strip().lower()
    user_category = request.form['category'].strip().lower()
    user_budget = request.form['budget'].strip().lower()

    # 🔄 Find actual case-insensitive matches in the dataset for encoding
    try:
        # Map lowercase input to actual country/category from dataset
        matched_country = next((c for c in enc_country.classes_ if c.lower() == user_country), None)
        matched_category = next((c for c in enc_category.classes_ if c.lower() == user_category), None)

        if not matched_country or not matched_category:
            return render_template(
                "results.html",
                ai_summary=f"⚠️ Invalid input! Try one of these:\n"
                           f"Countries: {list(enc_country.classes_)[:10]}\n"
                           f"Categories: {list(enc_category.classes_)[:10]}",
                destinations=None
            )

        # Encode inputs for prediction
        country_enc = enc_country.transform([matched_country])[0]
        category_enc = enc_category.transform([matched_category])[0]

        # Predict probabilities
        X_input = np.array([[country_enc, category_enc]])
        probs = model.predict_proba(X_input)[0]
        top_cost_index = np.argsort(-probs)[:3]
        top_cost_labels = enc_cost.inverse_transform(top_cost_index)
        top_probs = probs[top_cost_index]

        # Build AI summary text
        ai_summary = "\n".join([
            f"{label}: {round(prob * 100, 2)}% confidence"
            for label, prob in zip(top_cost_labels, top_probs)
        ])

        # Filter dataset ignoring case
        filtered = dataset[
            (dataset["Country"].str.lower() == user_country) &
            (dataset["Category"].str.lower() == user_category) &
            (dataset["Cost of Living"].str.lower() == user_budget)
        ]

        destinations = filtered.head(5).to_dict(orient="records")

        return render_template("results.html", ai_summary=ai_summary, destinations=destinations)

    except Exception as e:
        return render_template("results.html", ai_summary=f"⚠️ Error: {str(e)}", destinations=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
