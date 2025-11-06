# ==========================================================
# 🌍 AI Travel Recommender (Standalone Python Model)
# Predicts best destinations using Gradient Boosting
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# ==========================================================
# 1️⃣ Load and preprocess dataset
# ==========================================================
def load_and_train_model(file_path="destinations.csv"):
    # Load dataset
    data = pd.read_csv(file_path, encoding="latin1")

    # Remove duplicates based on primary key
    data.drop_duplicates(subset=["Destination", "Region"], keep="first", inplace=True)

    # Keep only relevant columns
    selected_columns = [
        "Destination", "Region", "Cost of Living",
        "Country", "Category", "Description"
    ]
    dataset = data[selected_columns].dropna()

    # Encode categorical columns
    encoder_country = LabelEncoder()
    encoder_category = LabelEncoder()
    encoder_cost = LabelEncoder()

    dataset["Country_enc"] = encoder_country.fit_transform(dataset["Country"])
    dataset["Category_enc"] = encoder_category.fit_transform(dataset["Category"])
    dataset["Cost_enc"] = encoder_cost.fit_transform(dataset["Cost of Living"])

    # Features (X) and target (y)
    X = dataset[["Country_enc", "Category_enc"]]
    y = dataset["Cost_enc"]

    # Split and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("✅ Model trained successfully!")
    print(f"🔹 Dataset size: {len(dataset)} rows\n")
    
    return dataset, model, encoder_country, encoder_category, encoder_cost


# ==========================================================
# 2️⃣ Prediction Function
# ==========================================================
def predict_destinations(model, dataset, enc_country, enc_category, enc_cost,
                         user_country, user_category, user_budget):
    try:
        # Encode user input
        country_enc = enc_country.transform([user_country])[0]
        category_enc = enc_category.transform([user_category])[0]
    except ValueError:
        print("\n⚠️ Invalid input. Please check your country or category name.")
        print("\n✅ Available Countries:", list(enc_country.classes_)[:10], "...")
        print("✅ Available Categories:", list(enc_category.classes_)[:10], "...")
        return

    # Predict cost probabilities
    X_input = np.array([[country_enc, category_enc]])
    probs = model.predict_proba(X_input)[0]
    top_cost_index = np.argsort(-probs)[:3]
    top_cost_labels = enc_cost.inverse_transform(top_cost_index)
    top_probs = probs[top_cost_index]

    print(f"\n🤖 AI Prediction — Cost of Living Trends for {user_category} in {user_country}:")
    for i in range(3):
        print(f"➡️ {top_cost_labels[i]} — {top_probs[i]*100:.2f}% confidence")

    # Filter destinations by user preferences
    filtered = dataset[
        (dataset["Country"].str.lower() == user_country.lower()) &
        (dataset["Category"].str.lower() == user_category.lower()) &
        (dataset["Cost of Living"].str.lower() == user_budget.lower())
    ]

    # Display matching destinations
    if not filtered.empty:
        print(f"\n🏆 Top Destinations in {user_country} for {user_budget} budget:")
        for i, row in filtered.head(5).iterrows():
            print(f"\n🌆 {row['Destination']} ({row['Region']})")
            print(f"💰 Cost of Living: {row['Cost of Living']}")
            print(f"🏝️ Category: {row['Category']}")
            print(f"📍 Country: {row['Country']}")
            print(f"📝 Description: {row['Description']}\n")
    else:
        print(f"\n⚠️ No exact match found for {user_budget}-budget destinations in {user_country}. Try another option.")


# ==========================================================
# 3️⃣ Example Usage (User Input)
# ==========================================================
if __name__ == "__main__":
    # Load model
    dataset, model, enc_country, enc_category, enc_cost = load_and_train_model("destinations.csv")

    # Get user input
    print("=== 🌎 AI Travel Recommender ===")
    user_country = input("Enter a country: ").strip()
    user_category = input("Enter a category (e.g., City, Coastal Town, Lake): ").strip()
    user_budget = input("Enter your budget preference (Low, Medium, High, etc.): ").strip()

    # Predict destinations
    predict_destinations(model, dataset, enc_country, enc_category, enc_cost,
                         user_country, user_category, user_budget)

