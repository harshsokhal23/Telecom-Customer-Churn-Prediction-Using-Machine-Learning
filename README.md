# 📊 Telecom Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Telecom Customer Churn Prediction is a machine learning project that analyses customer data to predict whether a customer is likely to leave a telecom company.

The project uses Python and machine learning algorithms to analyse customer characteristics, compare model performance, and explore customer segments using clustering. The results can help telecom companies better understand customer churn and plan customer retention strategies.

## 🎯 Objectives

- Analyse customer data to understand churn patterns.
- Clean and preprocess the dataset for machine learning.
- Train and compare multiple classification algorithms.
- Evaluate models using Accuracy, Precision, Recall, and F1-Score.
- Compare model performance across different train-test splits.
- Apply K-Means clustering to explore customer segments.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualisation:** Matplotlib, Seaborn
- **Dataset:** Telco Customer Churn CSV

## 📂 Project Structure

```text
Telecom-Customer-Churn-Prediction/
│
├── Telco-Customer-Churn.csv
├── churn_prediction.py
├── results_80_20.csv
├── results_70_30.csv
├── results_75_25.csv
├── all_model_results.csv
├── customers_with_clusters.csv
└── README.md
```

*Note: The filenames shown are suggested repository names. Include only the files that you actually upload to GitHub.*

## 📊 Dataset Description

The project uses the Telco Customer Churn dataset, containing **7,043 customer records and 21 original columns**.

The dataset includes information about:

- Customer demographics
- Tenure and account details
- Contract types
- Internet and telephone services
- Monthly and total charges
- Payment methods
- Customer churn status

The target variable is `Churn`, where `Yes` represents a customer who left and `No` represents a customer who stayed.

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

1. Load the dataset using Pandas.
2. Remove the `customerID` column.
3. Convert `TotalCharges` into numeric format.
4. Replace missing `TotalCharges` values with the median.
5. Encode the churn target as numerical values.
6. Convert categorical features using one-hot encoding.
7. Separate the input features and target variable.
8. Fill any remaining missing input values using column medians.
9. Standardise features where required by the selected algorithms.

## 🤖 Machine Learning Algorithms

The project compares four supervised classification algorithms:

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Predicts the probability of customer churn |
| Decision Tree | Uses decision rules to classify customers |
| Support Vector Machine (SVM) | Separates churn and non-churn classes |
| Random Forest | Combines multiple decision trees for classification |

The project also applies **K-Means Clustering** to group customers based on their encoded features.

## 📈 Model Evaluation

Each classification algorithm is tested using three train-test splits:

- 80:20
- 70:30
- 75:25

The models are evaluated using:

- **Accuracy:** Percentage of correct predictions.
- **Precision:** Proportion of predicted churn cases that are actually churn cases.
- **Recall:** Proportion of actual churn cases correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.

## 🏆 Results

The following table presents the best recorded results for each classification algorithm across the three train-test splits.

| Algorithm | Best Accuracy | Best F1-Score |
|---|---:|---:|
| Logistic Regression | 80.88% | 61.00% |
| Decision Tree | 74.17% | 50.75% |
| SVM | 79.37% | 55.87% |
| Random Forest | 78.89% | 55.58% |

**Best-performing model:** Logistic Regression achieved the highest recorded accuracy of **80.88%** and F1-score of **61.00%**, using the 70:30 train-test split.

### K-Means Clustering Results

| Cluster | Customers | Observed Churn Rate |
|---|---:|---:|
| Cluster 0 | 5,517 | 31.83% |
| Cluster 1 | 1,526 | 7.40% |

The clusters show different observed churn rates. These rates are descriptive results, not predictive accuracy measurements.

## 🚀 Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Telecom-Customer-Churn-Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Telecom-Customer-Churn-Prediction
```

### 3. Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 4. Add the Dataset

Place `Telco-Customer-Churn.csv` in the project folder.

Update the CSV path in the Python script if necessary. For example:

```python
df = pd.read_csv("Telco-Customer-Churn.csv")
```

### 5. Run the Project

```bash
python churn_prediction.py
```

The script trains the classification models, displays evaluation results, saves the comparison tables as CSV files, and generates the K-Means cluster output.

## 💡 Applications

- Customer retention analysis
- Telecom customer behaviour analysis
- Churn risk assessment
- Customer segmentation
- Data-driven business decision-making

## 🔮 Future Scope

- Implement advanced algorithms such as XGBoost.
- Develop a web application for churn prediction.
- Integrate interactive dashboards using Power BI.
- Improve model performance through hyperparameter tuning.
- Explore explainable machine learning techniques.
- Use updated customer data for periodic predictions.

## ⚠️ Limitations

- Model performance depends on the available dataset and selected features.
- The current project uses offline CSV data.
- Predictions require further validation before real-world deployment.
- Clustering identifies groups but does not directly predict churn.
- Customer data privacy and security must be considered for practical use.

## 👨‍💻 Author

**Harsh Sokhal**

B.Tech CSE — Lovely Professional University (LPU)

## 📄 License

This project is intended for educational and learning purposes. Add an appropriate open-source license if you wish to permit reuse or redistribution.

## ⭐ Acknowledgement

This project was developed as a learning exercise to understand machine learning, customer churn analysis, model evaluation, and customer segmentation using Python.
