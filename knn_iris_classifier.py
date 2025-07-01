
# K-Nearest Neighbors Classification using Iris Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv('/content/Iris.csv')

# Drop Id column if present
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)

# Separate features and labels
X = df.drop('Species', axis=1)
y = df['Species']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
target_names = le.classes_

# 2. Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 4. Try different values of K
k_values = range(1, 11)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K={k} → Accuracy: {acc:.2f}")

# 5. Plot accuracy vs K (with enhanced visualization)
best_k = k_values[np.argmax(accuracies)]
best_acc = max(accuracies)

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='--', color='b', label='Accuracy')
plt.axvline(x=best_k, color='r', linestyle=':', label=f'Best K = {best_k}')
plt.scatter(best_k, best_acc, color='red', s=100, zorder=5)
plt.title('Accuracy vs K value')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.ylim(min(accuracies)-0.005, max(accuracies)+0.005)
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()

print(f"\nBest K: {best_k}")
print(f"Best Accuracy: {best_acc:.2f}")

# 6. Final model with best K
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_pred_final = knn_final.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=target_names))

# 7. Visualize Decision Boundaries using 2 features (2D)
X_vis = df[['PetalLengthCm', 'PetalWidthCm']].values
y_vis = le.transform(df['Species'])
X_vis_scaled = StandardScaler().fit_transform(X_vis)

X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis_scaled, y_vis, test_size=0.2, random_state=42)

knn_vis = KNeighborsClassifier(n_neighbors=best_k)
knn_vis.fit(X_train_vis, y_train_vis)

# Create meshgrid
x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_vis_scaled[:, 0], X_vis_scaled[:, 1], c=y_vis, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel('Petal Length (Standardized)')
plt.ylabel('Petal Width (Standardized)')
plt.title(f'KNN Decision Boundary (k={best_k})')
plt.show()
