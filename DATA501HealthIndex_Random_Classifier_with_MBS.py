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
col_needed = ['Sargassum_Health_Index', 'Marine_Biodiversity_Score', 'Water_Temperature_C',
              'Dissolved_Oxygen_mg_L', 'pH_Level', 'Nutrient_Level_ppm']
data = pd.read_csv("bootstrapped_UP-5.csv", usecols=col_needed)

# Create health categories
data['Sargassum_Health_Index'] = data['Sargassum_Health_Index'].round().astype(int)
bins = [0, 40, 70, 100]  # Upper limits for Poor, Moderate, and Good
labels = ['Poor', 'Moderate', 'Good']
reversed_labels = list(reversed(labels))
data['Health_Category'] = pd.cut(
    data['Sargassum_Health_Index'], bins=bins, labels=labels, include_lowest=True)

# Include bioscore from the beginning
feature_col = ['Water_Temperature_C', 'Dissolved_Oxygen_mg_L',
               'pH_Level', 'Nutrient_Level_ppm', 'Marine_Biodiversity_Score']
X = data[feature_col]
y_label = data['Health_Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_label, test_size=0.3, random_state=1, stratify=y_label)

# APPROACH 1: Original features with optimized decision tree
print("\n=== APPROACH 1: Original Features with Optimized Decision Tree ===")
baseline_clf = DecisionTreeClassifier(
    random_state=1, criterion="gini", max_depth=4)
baseline_clf.fit(X_train, y_train)
y_pred_baseline = baseline_clf.predict(X_test)
baseline_accuracy = metrics.accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

# Define parameter grid for hyperparameter tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=1),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
best_clf = grid_search.best_estimator_

y_pred_best = best_clf.predict(X_test)
best_accuracy = metrics.accuracy_score(y_test, y_pred_best)
print(f"Optimized Decision Tree Accuracy: {best_accuracy:.4f}")

# APPROACH 2: Feature Engineering
print("\n=== APPROACH 2: Feature Engineering ===")
data['Temp_DO_Interaction'] = data['Water_Temperature_C'] * data['Dissolved_Oxygen_mg_L']
data['pH_Nutrient_Interaction'] = data['pH_Level'] * data['Nutrient_Level_ppm']
data['Temp_Squared'] = data['Water_Temperature_C'] ** 2
data['DO_Squared'] = data['Dissolved_Oxygen_mg_L'] ** 2
data['pH_Squared'] = data['pH_Level'] ** 2

expanded_features = [
    'Water_Temperature_C', 'Dissolved_Oxygen_mg_L', 'pH_Level', 'Nutrient_Level_ppm',
    'Marine_Biodiversity_Score',  # Include bioscore
    'Temp_DO_Interaction', 'pH_Nutrient_Interaction',
    'Temp_Squared', 'DO_Squared', 'pH_Squared'
]

X_expanded = data[expanded_features]
X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(
    X_expanded, y_label, test_size=0.3, random_state=1, stratify=y_label)

exp_clf = DecisionTreeClassifier(random_state=1, **grid_search.best_params_)
exp_clf.fit(X_train_exp, y_train_exp)
y_pred_exp = exp_clf.predict(X_test_exp)
exp_accuracy = metrics.accuracy_score(y_test_exp, y_pred_exp)
print(f"Expanded Features Accuracy: {exp_accuracy:.4f}")

# APPROACH 3: Feature Importance and Selection
print("\n=== APPROACH 3: Feature Importance Analysis ===")
importances = best_clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importance ranking:")
for i in range(len(feature_col)):
    print(f"{i+1}. {feature_col[indices[i]]} ({importances[indices[i]]:.4f})")

top_n = min(3, len(feature_col))
top_features = [feature_col[i] for i in indices[:top_n]]
print(f"Top features: {top_features}")

X_top = data[top_features]
X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
    X_top, y_label, test_size=0.3, random_state=1, stratify=y_label)

top_clf = DecisionTreeClassifier(random_state=1, **grid_search.best_params_)
top_clf.fit(X_train_top, y_train_top)
y_pred_top = top_clf.predict(X_test_top)
top_accuracy = metrics.accuracy_score(y_test_top, y_pred_top)
print(f"Top Features Accuracy: {top_accuracy:.4f}")

# APPROACH 5: Random Forest
print("\n=== APPROACH 5: Random Forest Classifier ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = metrics.accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# Random Forest with all expanded features
rf_ext_model = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
rf_ext_model.fit(X_train_exp, y_train_exp)
y_pred_rf_ext = rf_ext_model.predict(X_test_exp)
rf_ext_accuracy = metrics.accuracy_score(y_test_exp, y_pred_rf_ext)
print(f"Random Forest with Expanded Features Accuracy: {rf_ext_accuracy:.4f}")

# Compare all
accuracies = {
    "Baseline": (baseline_accuracy, baseline_clf, X_test, y_test, y_pred_baseline),
    "Optimized DT": (best_accuracy, best_clf, X_test, y_test, y_pred_best),
    "Feature Engineering": (exp_accuracy, exp_clf, X_test_exp, y_test_exp, y_pred_exp),
    "Top Features": (top_accuracy, top_clf, X_test_top, y_test_top, y_pred_top),
    "Random Forest": (rf_accuracy, rf_model, X_test, y_test, y_pred_rf),
    "RF Extended": (rf_ext_accuracy, rf_ext_model, X_test_exp, y_test_exp, y_pred_rf_ext)
}

best_model_name = max(accuracies, key=lambda k: accuracies[k][0])
best_model_acc, best_model, best_X_test, best_y_test, best_y_pred = accuracies[best_model_name]

print(f"\n=== BEST MODEL: {best_model_name} with accuracy {best_model_acc:.4f} ===")
print("\nClassification Report:")
print(classification_report(best_y_test, best_y_pred))

cm = confusion_matrix(best_y_test, best_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=reversed_labels, yticklabels=reversed_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix - {best_model_name} (Accuracy: {best_model_acc:.4f})')
plt.tight_layout()
plt.savefig('best_model_confusion_matrix.png')
plt.show()

if not isinstance(best_model, RandomForestClassifier):
    feature_names = best_X_test.columns
    dot_data = export_graphviz(best_model, out_file=None,
                               feature_names=feature_names,
                               class_names=reversed_labels,
                               filled=True, rounded=True, special_characters=True,
                               max_depth=5)
    graph = graphviz.Source(dot_data)
    graph.render(f"decision_tree_{best_model_name}")
    graph.view()

if isinstance(best_model, RandomForestClassifier):
    feature_names = best_X_test.columns
    rf_importances = best_model.feature_importances_
    rf_indices = np.argsort(rf_importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances - Random Forest')
    plt.bar(range(len(feature_names)),
            [rf_importances[i] for i in rf_indices],
            align='center')
    plt.xticks(range(len(feature_names)),
               [feature_names[i] for i in rf_indices],
               rotation=90)
    plt.tight_layout()
    plt.savefig('rf_feature_importances.png')
    plt.show()

print("\nAnalysis complete! The best model has been identified and visualized.")
joblib.dump(best_model, f'sargassum_health_best_model.pkl')
print(f"Best model saved as 'sargassum_health_best_model.pkl'")
