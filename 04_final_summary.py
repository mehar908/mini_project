import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

# Predictions + Probabilities
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Risk function
def get_risk_level(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Fraud_Probability": y_proba.round(3),
    "Risk_Level": [get_risk_level(p) for p in y_proba]
})

print("="*50)
print("       BANK FRAUD DETECTION - FINAL SUMMARY")
print("="*50)

print("\n1. Dataset Info")
print(f"Total transactions     : {df.shape[0]}")
print(f"Fraud cases            : {df['Class'].sum()}")
print(f"Fraud percentage       : {df['Class'].mean()*100:.3f}%")

print("\n2. Model Performance (Fraud Class)")
print(classification_report(y_test, y_pred, digits=3))

print("\n3. Risk Level Distribution (on Test Data)")
print(results["Risk_Level"].value_counts())

print("\n4. Sample High Risk Transactions")
print(results[results["Risk_Level"] == "High Risk"].head(8))