import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
digits = load_digits()
X = digits.data

# 2. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Evaluate K-Means across k values
k_values = range(2, 11)
inertia = []
silhouette = []

print("K\tInertia\t\tSilhouette Score")
for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X_scaled)

    inertia.append(model.inertia_)
    score = silhouette_score(X_scaled, labels)
    silhouette.append(score)

    print(f"{k}\t{model.inertia_:.2f}\t\t{score:.4f}")

# 4. Plot Elbow Method (Inertia)
plt.figure(figsize=(6, 4))
plt.plot(k_values, inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# 5. Plot Silhouette Score vs K
plt.figure(figsize=(6, 4))
plt.plot(k_values, silhouette, marker="o", color="orange")
plt.title("Silhouette Score vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()
