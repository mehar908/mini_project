import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("creditcard.csv")

print("Class Distribution:")
print(df["Class"].value_counts())
print("\nPercentage:")
print(df["Class"].value_counts(normalize=True) * 100)

# ========== GRAPH 1: Count of Normal vs Fraud ==========
plt.figure(figsize=(6,4))
sns.countplot(x="Class", data=df)
plt.title("Normal (0) vs Fraud (1) Transactions")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# ========== GRAPH 2: Distribution of Transaction Amount ==========
plt.figure(figsize=(8,4))
sns.histplot(df["Amount"], bins=50, kde=True)
plt.title("Distribution of Transaction Amount")
plt.xlabel("Amount")
plt.show()

# ========== GRAPH 3: Amount by Class (Normal vs Fraud) ==========
plt.figure(figsize=(8,4))
sns.boxplot(x="Class", y="Amount", data=df)
plt.title("Transaction Amount: Normal vs Fraud")
plt.show()