# ===================== COMMON IMPORTS =====================
import pandas as pd
from sklearn.model_selection import train_test_split

# ===================== LOAD DATA =====================
data = pd.read_csv("data.csv")

# For supervised learning
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ===================== 1. LINEAR REGRESSION =====================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)

lr_pred = lr_model.predict(X_test)

print("Linear Regression MSE:", mean_squared_error(Y_test, lr_pred))
print("Linear Regression R2:", r2_score(Y_test, lr_pred))


# ===================== 2. LOGISTIC REGRESSION =====================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, Y_train)

log_pred = log_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(Y_test, log_pred))


# ===================== 3. KNN =====================
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)

knn_pred = knn_model.predict(X_test)

print("KNN Accuracy:", accuracy_score(Y_test, knn_pred))


# ===================== 4. DECISION TREE =====================
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)

dt_pred = dt_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(Y_test, dt_pred))


# ===================== 5. SVM =====================
from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, Y_train)

svm_pred = svm_model.predict(X_test)

print("SVM Accuracy:", accuracy_score(Y_test, svm_pred))


# ===================== 6. RANDOM FOREST =====================
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)

rf_pred = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(Y_test, rf_pred))


# ===================== 7. BOOSTING =====================
from sklearn.ensemble import AdaBoostClassifier

adb_model = AdaBoostClassifier()
adb_model.fit(X_train, Y_train)

adb_pred = adb_model.predict(X_test)

print("AdaBoost Accuracy:", accuracy_score(Y_test, adb_pred))


# ===================== 8. K-MEANS =====================
from sklearn.cluster import KMeans

# Use only features (no Y)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

print("K-Means Labels:")
print(kmeans.labels_)


# ===================== 9. PCA =====================
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("PCA Reduced Data:")
print(X_reduced)
