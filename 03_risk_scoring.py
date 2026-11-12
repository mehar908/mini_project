import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Get probability of fraud (this is the key for risk)
y_proba = model.predict_proba(X_test)[:, 1]   # probability of class 1 (fraud)

# Create a simple Risk Level
def get_risk_level(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

# Make a small results table
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": model.predict(X_test),
    "Fraud_Probability": y_proba,
    "Risk_Level": [get_risk_level(p) for p in y_proba]
})

print("=== Sample of Risk Scoring ===")
print(results.head(10))

print("\n=== Risk Level Distribution ===")
print(results["Risk_Level"].value_counts())