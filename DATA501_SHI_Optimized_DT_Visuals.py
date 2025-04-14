# Load libraries
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# Import data
col_needed = [
    "Sargassum_Health_Index",
    "Marine_Biodiversity_Score",
    "Water_Temperature_C",
    "Dissolved_Oxygen_mg_L",
    "pH_Level",
    "Nutrient_Level_ppm",
]
data = pd.read_csv("bootstrapped_UP-5.csv", usecols=col_needed)

# Create health categories
data["Sargassum_Health_Index"] = data["Sargassum_Health_Index"].round().astype(int)
bins = [0, 40, 70, 100]  # Upper limits for Poor, Moderate, and Good
labels = ["Poor", "Moderate", "Good"]
reversed_labels = list(reversed(labels))
data["Health_Category"] = pd.cut(
    data["Sargassum_Health_Index"], bins=bins, labels=labels, include_lowest=True
)

# Define feature columns
feature_col = [
    "Water_Temperature_C",
    "Dissolved_Oxygen_mg_L",
    "pH_Level",
    "Nutrient_Level_ppm",
]
X = data[feature_col]
y_label = data["Health_Category"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_label, test_size=0.3, random_state=1, stratify=y_label
)

# Define parameter grid for hyperparameter tuning
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"],
}

# Grid search optimization
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=1),
    param_grid=param_grid,
    cv=10,  # Increased cross-validation folds
    scoring="accuracy",
    n_jobs=-1,
)

print(y_train.value_counts())  # Check training class distribution
print(y_test.value_counts())  # Check test class distribution

grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)

# Train the optimized decision tree
best_clf = grid_search.best_estimator_
y_pred_best = best_clf.predict(X_test)
best_accuracy = metrics.accuracy_score(y_test, y_pred_best)
print(f"Optimized Decision Tree Accuracy: {best_accuracy:.4f}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=reversed_labels,
    yticklabels=reversed_labels,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - Optimized Decision Tree (Accuracy: {best_accuracy:.4f})")
plt.tight_layout()
plt.savefig("SHI_optimized_decision_tree_cm.png")
plt.show()

# Decision tree visualization
feature_names = X.columns
dot_data = export_graphviz(
    best_clf,
    out_file=None,
    feature_names=feature_names,
    class_names=reversed_labels,
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph.render("optimized_decision_tree")
graph.view()

# Get feature importance from the best decision tree model
feature_importance = pd.Series(
    best_clf.feature_importances_, index=feature_col
).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=feature_importance.index, palette="Blues_r")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.subplots_adjust(left=0.3)
plt.title("SHI_Feature Importance in Optimized Decision Tree")
plt.savefig("SHI_Feature_Importance_DT.png")
plt.show()

# Generate detailed classification report for best model
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))
