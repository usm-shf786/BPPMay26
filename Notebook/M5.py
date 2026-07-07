import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "loan_risk_prediction_dataset.xlsx")

df = pd.read_excel(DATA_PATH, sheet_name="Loan Data Cleaned")

rows_before = len(df)
df = df[df["Income"] >= 0].reset_index(drop=True)
print(f"Dropped {rows_before - len(df)} rows with negative Income")

print("Shape (rows, columns):", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn data types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isna().sum())

print("\nSummary statistics:")
print(df.describe())

for col in ["Gender", "Education", "City", "EmploymentType"]:
    print(f"\nUnique values in {col}: {df[col].unique()}")

print("\nLoanApproved value counts:")
print(df["LoanApproved"].value_counts())

PLOT_DIR = os.path.join(BASE_DIR, "eda_plots")
os.makedirs(PLOT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

numeric_cols = ["Age", "Income", "LoanAmount", "CreditScore", "YearsExperience"]
categorical_cols = ["Gender", "Education", "City", "EmploymentType"]

plt.figure(figsize=(5, 4))
sns.countplot(x="LoanApproved", data=df)
plt.title("Loan Approval Class Balance")
plt.xlabel("LoanApproved (0 = rejected, 1 = approved)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "01_target_balance.png"), dpi=120)
plt.show()

df[numeric_cols].hist(figsize=(12, 8), bins=30, edgecolor="black")
plt.suptitle("Distributions of Numeric Features")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "02_numeric_histograms.png"), dpi=120)
plt.show()

plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols, start=1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.suptitle("Boxplots (Outlier Check) of Numeric Features")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "03_numeric_boxplots.png"), dpi=120)
plt.show()

plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols, start=1):
    plt.subplot(2, 3, i)
    sns.boxplot(x="LoanApproved", y=col, data=df)
    plt.title(f"{col} by LoanApproved")
plt.suptitle("Numeric Features Split by Loan Approval")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "04_numeric_vs_target.png"), dpi=120)
plt.show()

plt.figure(figsize=(14, 8))
for i, col in enumerate(categorical_cols, start=1):
    plt.subplot(2, 2, i)
    sns.countplot(x=col, hue="LoanApproved", data=df)
    plt.title(f"{col} vs LoanApproved")
    plt.xticks(rotation=30, ha="right")
plt.suptitle("Categorical Features Split by Loan Approval")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "05_categorical_vs_target.png"), dpi=120)
plt.show()

plt.figure(figsize=(8, 6))
corr = df[numeric_cols + ["LoanApproved"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap (Numeric Features + Target)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "06_correlation_heatmap.png"), dpi=120)
plt.show()

print(f"\nEDA plots saved to: {PLOT_DIR}")
print("Numeric correlations with LoanApproved:")
print(corr["LoanApproved"].sort_values(ascending=False))

education_order = {
    "Unknown": 0,
    "High School": 1,
    "Bachelors": 2,
    "Masters": 3,
    "PhD": 4,
}
df["Education"] = df["Education"].map(education_order)

print("\nEducation after ordinal encoding (value: count):")
print(df["Education"].value_counts().sort_index())

df = pd.get_dummies(
    df,
    columns=["Gender", "City", "EmploymentType"],
    drop_first=True,
)

print("\nColumns after one-hot encoding:")
print(df.columns.tolist())
print("\nFirst 5 rows after encoding:")
print(df.head())

numeric_features = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "YearsExperience",
]
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

print("\nNumeric features after standardisation (first 5 rows):")
print(df[numeric_features].head())
print("\nMean (~0) and std (~1) check:")
print(df[numeric_features].describe().loc[["mean", "std"]])

X = df.drop(columns=["LoanApproved"])
y = df["LoanApproved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining rows: {len(X_train)}, Test rows: {len(X_test)}")

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print("\n=== Logistic Regression performance ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.3f}")
print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.3f}")

print("\nConfusion matrix (rows = actual, cols = predicted):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

X_cluster = df.drop(columns=["LoanApproved"])

inertias = []
silhouettes = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_cluster, labels))
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(k_range), inertias, marker="o")
ax1.set_title("Elbow Method (Inertia vs k)")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Inertia (within-cluster SSE)")
ax2.plot(list(k_range), silhouettes, marker="o", color="darkorange")
ax2.set_title("Silhouette Score vs k")
ax2.set_xlabel("Number of clusters (k)")
ax2.set_ylabel("Silhouette score")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "07_kmeans_elbow_silhouette.png"), dpi=120)
plt.show()

best_k = k_range[int(pd.Series(silhouettes).idxmax())]
print(f"\nChosen k (highest silhouette): {best_k}")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_cluster)

print("\nCluster sizes:")
print(df["Cluster"].value_counts().sort_index())

print("\nLoan approval rate per cluster:")
print(df.groupby("Cluster")["LoanApproved"].mean().round(3))

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_cluster)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=coords[:, 0],
    y=coords[:, 1],
    hue=df["Cluster"],
    palette="tab10",
    s=25,
    alpha=0.7,
)
plt.title(f"K-Means Clusters (k={best_k}) projected with PCA")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "08_kmeans_pca_scatter.png"), dpi=120)
plt.show()

print(f"\nK-Means plots saved to: {PLOT_DIR}")