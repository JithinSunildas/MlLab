import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load Iris Dataset
iris = datasets.load_iris()

# Use only Petal Length and Petal Width
X = iris.data[:, [2, 3]]

# Binary Classification: Setosa = 1, Non-Setosa = 0
y = (iris.target == 0).astype(int)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Linear SVM
svm_model = SVC(kernel="linear", C=1)
svm_model.fit(X_train, y_train)

# Prediction & Metrics
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Plot Decision Boundary
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = svm_model.decision_function(xy)
Z = Z.reshape(XX.shape)

# Decision Boundary & Margins
ax.contour(
    XX,
    YY,
    Z,
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.8,
    linestyles=["--", "-", "--"],
)

# Support Vectors
ax.scatter(
    svm_model.support_vectors_[:, 0],
    svm_model.support_vectors_[:, 1],
    s=120,
    linewidth=1.5,
    facecolors="none",
    edgecolors="black",
    label="Support Vectors",
)

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Linear SVM Decision Boundary and Margin")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
