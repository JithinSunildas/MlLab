import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC
from torchvision.datasets import FashionMNIST

train_data = FashionMNIST(root="./data", train=True, download=True)
test_data = FashionMNIST(root="./data", train=False, download=True)

X_train = train_data.data.numpy()
y_train = train_data.targets.numpy()
X_test = test_data.data.numpy()
y_test = test_data.targets.numpy()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

X_train = X_train[:10000]
y_train = y_train[:10000]
X_test = X_test[:2000]
y_test = y_test[:2000]

linear_svm = SVC(kernel="linear")
linear_svm.fit(X_train, y_train)
linear_pred = linear_svm.predict(X_test)

poly_svm = SVC(kernel="poly", degree=3)
poly_svm.fit(X_train, y_train)
poly_pred = poly_svm.predict(X_test)

rbf_svm = SVC(kernel="rbf")
rbf_svm.fit(X_train, y_train)
rbf_pred = rbf_svm.predict(X_test)


def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{model_name}")
    print("-" * 40)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


evaluate_model(y_test, linear_pred, "Linear SVM")
evaluate_model(y_test, poly_pred, "Polynomial SVM")
evaluate_model(y_test, rbf_pred, "RBF SVM")
