import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
n_samples = len(X)
print("Dataset loaded by Malavika Gireesh 36")

model = DecisionTreeClassifier(random_state=42)

# --- Bootstrapping (Out-Of-Bag Evaluation) ---
boot_accuracy = []
boot_f1 = []
n_iterations = 10

for i in range(n_iterations):
    boot_indices = resample(np.arange(n_samples), replace=True, random_state=i)
    oob_indices = list(set(range(n_samples)) - set(boot_indices))

    if not oob_indices:
        continue

    X_boot, y_boot = X[boot_indices], y[boot_indices]
    X_test, y_test = X[oob_indices], y[oob_indices]

    model.fit(X_boot, y_boot)
    pred = model.predict(X_test)

    boot_accuracy.append(accuracy_score(y_test, pred))
    boot_f1.append(f1_score(y_test, pred, average="weighted"))

print("Bootstrapping Results")
print("Average Accuracy :", round(np.mean(boot_accuracy), 4))
print("Average F1 Score :", round(np.mean(boot_f1), 4))

# --- K-Fold Cross Validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = []
cv_f1 = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    cv_accuracy.append(accuracy_score(y_test, pred))
    cv_f1.append(f1_score(y_test, pred, average="weighted"))

print("\nK-Fold Cross Validation Results")
print("Average Accuracy :", round(np.mean(cv_accuracy), 4))
print("Average F1 Score :", round(np.mean(cv_f1), 4))
