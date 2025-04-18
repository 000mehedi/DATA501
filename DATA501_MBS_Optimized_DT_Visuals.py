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

# Create biodiversity categories
data["Marine_Biodiversity_Score"] = (
    data["Marine_Biodiversity_Score"].round().astype(int)
)
# Upper limits for Low, Moderate, and High
bins_biodiversity = [50, 65, 80, 100]
labels_biodiversity = ["Low", "Moderate", "High"]
reverse_labels_biodiversity = ["High", "Moderate", "Low"]
data["Biodiversity_Category"] = pd.cut(
    data["Marine_Biodiversity_Score"],
    bins=bins_biodiversity,
    labels=labels_biodiversity,
    include_lowest=True,
)

# APPROACH 1: Original features with optimized decision tree
print("\n=== APPROACH 1: Original Features with Optimized Decision Tree ===")
feature_col = [
    "Water_Temperature_C",
    "Sargassum_Health_Index",
    "Dissolved_Oxygen_mg_L",
    "pH_Level",
    "Nutrient_Level_ppm",
]
X = data[feature_col]
y_label_biodiversity = data["Biodiversity_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_label_biodiversity,
    test_size=0.3,
    random_state=1,
    stratify=y_label_biodiversity,
)

# Create and evaluate baseline model
baseline_clf = DecisionTreeClassifier(random_state=1, criterion="gini", max_depth=4)
baseline_clf.fit(X_train, y_train)
y_pred_baseline = baseline_clf.predict(X_test)
baseline_accuracy = metrics.accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

# Define parameter grid for hyperparameter tuning
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"],
}

# Create grid search
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=1),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

# Fit the model with grid search
grid_search.fit(X_train, y_train)

# Best parameters and model
print("Best parameters:", grid_search.best_params_)
best_clf = grid_search.best_estimator_

# Evaluate optimized model
y_pred_best = best_clf.predict(X_test)
best_accuracy = metrics.accuracy_score(y_test, y_pred_best)
print(f"Optimized Decision Tree Accuracy: {best_accuracy:.4f}")

# Decision tree visualization
feature_names = X.columns
dot_data = export_graphviz(
    best_clf,
    out_file=None,
    feature_names=feature_names,
    class_names=labels_biodiversity,
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph.render("MBS_optimized_decision_tree")
graph.view()

# Confusion matrix visualization

# Create label index mapping
label_to_index = {label: i for i, label in enumerate(np.unique(y_test))}

# Reorder the confusion matrix
# Note: Make sure cm is calculated with the `labels` param


cm = confusion_matrix(y_test, y_pred_best, labels=labels_biodiversity)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels_biodiversity,
    yticklabels=labels_biodiversity,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - Optimized Decision Tree (Accuracy: {best_accuracy:.4f})")
plt.tight_layout()
plt.savefig("MBS_optimized_decision_tree_cm.png")
plt.show()

# Get feature importance from the best decision tree model
feature_importance = pd.Series(
    best_clf.feature_importances_, index=feature_col
).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=feature_importance.index, palette="Blues_r")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Optimized Decision Tree")
plt.subplots_adjust(left=0.3)
plt.savefig("MBS_Feature_Importance_in_Optimized_Decision_Tree.png")
plt.show()

# Generate detailed classification report for best model
print("\nClassification Report:")
print(
    classification_report(
        y_test, y_pred_best, labels=reverse_labels_biodiversity, zero_division=0
    )
)
