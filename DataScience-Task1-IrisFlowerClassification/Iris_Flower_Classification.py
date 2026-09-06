"""
OASIS Infobyte SIP - Data Science Task 1
Iris Flower Classification

This script follows the same workflow as the accompanying notebook.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load the built-in Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("First five rows:")
print(df.head())

print("\nShape:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nNull values:")
print(df.isnull().sum())
print("\nDescriptive statistics:")
print(df.describe())
print("\nSpecies distribution:")
print(df["species"].value_counts())


# 2. Visualize feature relationships by species
sns.pairplot(df, hue="species", diag_kind="hist")
plt.suptitle("Iris Feature Relationships by Species", y=1.02)
plt.show()


# 3. Box plots for every feature
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, feature in zip(axes.ravel(), iris.feature_names):
    sns.boxplot(data=df, x="species", y=feature, ax=ax)
    ax.set_title(f"{feature.title()} by Species")
    ax.set_xlabel("Species")
    ax.set_ylabel(feature.title())
plt.tight_layout()
plt.show()


# 4. Feature selection discussion
print("""
Feature-selection observation:
Petal length and petal width are the most discriminative features because
their distributions show clearer separation among the species. Sepal
measurements overlap more strongly.
""")


# 5. Train/test split (80/20)
X = df[iris.feature_names]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 6. Train two classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print(f"\n{'=' * 60}\n{name}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, target_names=iris.target_names, digits=4
    ))


# 7. Compare models and identify the best model
best_model_name = max(results, key=results.get)
print("\nModel comparison:")
for name, score in results.items():
    print(f"{name}: {score:.2%}")

print(f"\nBest-performing model: {best_model_name}")
print("Justification: it achieved the highest test accuracy; its precision,")
print("recall and F1-score should also be considered from the reports above.")


# 8. Final prediction demonstration
best_model = models[best_model_name]
sample = X_test.iloc[[0]]
prediction = best_model.predict(sample)[0]

print("\nFinal prediction:")
print("Input measurements:")
print(sample)
print("Predicted species:", iris.target_names[prediction])
