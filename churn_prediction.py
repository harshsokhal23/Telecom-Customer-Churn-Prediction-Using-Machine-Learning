
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
df = pd.read_csv(r"C:\INT234 Project\Telco-Customer-Churn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Remove customer ID
df.drop("customerID", axis=1, inplace=True)

# 3. Convert TotalCharges into numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# 4. Fill missing TotalCharges
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# 5. Convert target into numbers
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 6. Convert text columns into numeric columns
df = pd.get_dummies(df, drop_first=True)

# 7. Separate input and output
X = df.drop("Churn", axis=1).astype(float)
y = df["Churn"]

# Fill any remaining missing input values
X = X.fillna(X.median())

# 8. Define four classification algorithms
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000)
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC()
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42,
        n_estimators=100
    )
}

# 9. Define three train-test splits
splits = {
    "80:20": 0.20,
    "70:30": 0.30,
    "75:25": 0.25
}

all_results = []

# 10. Train and test the models
for split_name, test_size in splits.items():

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    print("\nTrain-Test Split:", split_name)
    print("-" * 65)

    split_results = []

    for name, model in models.items():

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        prediction = model.predict(X_test)

        # Calculate performance
        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(
            y_test, prediction, zero_division=0
        )
        recall = recall_score(
            y_test, prediction, zero_division=0
        )
        f1 = f1_score(
            y_test, prediction, zero_division=0
        )

        result = {
            "Split": split_name,
            "Algorithm": name,
            "Accuracy": round(accuracy * 100, 2),
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1 Score": round(f1 * 100, 2)
        }

        split_results.append(result)
        all_results.append(result)

    # Display comparison table
    table = pd.DataFrame(split_results)
    print(table.drop(columns=["Split"]).to_string(index=False))

    # Save separate table for each split
    filename = "results_" + split_name.replace(":", "_") + ".csv"
    table.to_csv(filename, index=False)

# 11. Save combined comparison results
results = pd.DataFrame(all_results)
results.to_csv("all_model_results.csv", index=False)

print("\nAll classification results saved successfully!")

# 12. K-Means clustering (fifth algorithm)
print("\nK-Means Clustering")
print("-" * 40)

# Scale features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create two customer clusters
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
df["Cluster"] = clusters
df["Actual_Churn"] = y.to_numpy()

# Count customers in each cluster
print("\nCustomers in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# Compare actual churn rate in each cluster
print("\nChurn rate in each cluster (%):")
print(
    (df.groupby("Cluster")["Actual_Churn"].mean() * 100).round(2)
)

# Save cluster results
df.to_csv("customers_with_clusters.csv", index=False)

print("\nK-Means results saved successfully!")
print("\nProject completed successfully!")
