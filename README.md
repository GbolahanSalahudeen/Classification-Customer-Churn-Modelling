📉 Customer Churn Prediction Using Machine Learning

> A supervised machine learning project to predict customer churn using Random Forest, Logistic Regression, SVM, KNN, and XGBoost.
Goal: Help businesses retain customers and boost profitability using predictive analytics.




---

📌 Overview

This project develops a predictive model to identify customer churn using five machine learning algorithms:

Random Forest

Logistic Regression

Support Vector Machine (SVM)

K-Nearest Neighbor (KNN)

Gradient Boosting (XGBoost)


By analyzing customer attributes like credit score, age, tenure, and account balance, the model predicts whether a customer is likely to leave or stay. Since acquiring new customers is costlier than retaining existing ones, churn prediction helps businesses develop smarter retention strategies.

> 💡 Retaining just 5% more customers can increase profits by 25–95%.




📋 Project Description

This is a binary classification problem in supervised learning — predicting whether a customer will churn (1) or not churn (0).

The project uses multiple machine learning models to ensure performance robustness and reliability. Performance is evaluated not just by accuracy, but also by precision, recall, and specificity due to the real-world cost of misclassifying churners.


---

🧾 Dataset

The dataset contains the following features:

**Credit Score:**	Numerical score indicating creditworthiness

**Gender**:	Male or Female

**Geography:**	Country of the customer

**Age**:	Customer's age

**Tenure**:	Years with the bank

**Balance**:	Account balance

**Number of Products**:	Number of services/products used

**Credit Card	Binary**: (has credit card or not)

**Active Member	Binary**: (active customer or not)

**Estimated Salary**:	Annual income estimate

Churn	Target variable (1 = churned, 0 = not churned)




> ⚠️ Note: The dataset is not included in this repository due to size/privacy constraints. Please provide your own dataset in .csv format with the above features.




---

⚙️Methodology


The following algorithms were implemented:

✅ **Random Forest:** Ensemble learning using multiple decision trees.

✅ **Logistic Regression:** Linear model for binary outcomes.

✅ **SVM (Support Vector Machine):** Maximizes margin between churners and non-churners.

✅ **K-Nearest Neighbor (KNN):** Classifies based on proximity to similar examples.

✅ **Gradient Boosting (XGBoost):** Efficient boosting algorithm optimized for performance.


#Preprocessing Steps:

Encoding categorical variables

Scaling numerical features

Splitting data into training and test sets



---

📊 Evaluation Metrics

Accuracy alone can be misleading, especially for imbalanced datasets. The following metrics were used for a more complete evaluation:

Confusion Matrix

TP (True Positives): Correctly predicted churners

TN (True Negatives): Correctly predicted non-churners

FP: Non-churners incorrectly predicted as churners

FN: Churners incorrectly predicted as non-churners


Precision = TP / (TP + FP)

Recall (Sensitivity) = TP / (TP + FN)

Specificity = TN / (TN + FP)


> 🎯 In most business contexts, False Negatives (missed churners) can lead to revenue loss, so high recall is critical.

---

✅ Results
| Algorithm           | Accuracy | Precision | Recall | Notes                                 |
|---------------------|----------|-----------|--------|----------------------------------------|
| Random Forest       | 86.90%   | 88%       | 97%    | High accuracy and recall               |
| Logistic Regression | 81.05%   | 83%       | 96%    | Good baseline performance              |
| SVM                 | 80.35%   | 80%       | 100%   | Perfect recall, but moderate precision |
| KNN                 | 82.50%   | 86%       | 94%    | Strong balance across metrics          |
| Gradient Boosting   | 86.65%   | 88%       | 92%    | Second-best performance overall        |

