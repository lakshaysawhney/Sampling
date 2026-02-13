# Sampling Techniques and Model Performance on an Imbalanced Credit Card Dataset

## Overview
This repository contains an academic implementation that evaluates how different sampling techniques affect the predictive accuracy of multiple machine learning models on a highly imbalanced credit card dataset. The workflow includes balancing the dataset, computing a statistically motivated sample size, generating multiple samples using distinct sampling strategies, and benchmarking model accuracy on a held-out test set.

## Dataset
- Source (GitHub): https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv
- Local file used in this repository: `Creditcard_data.csv`
- Target column: `Class` (binary classification)

## Methodology

### 1) Loading and Balancing the Dataset
The dataset is initially highly imbalanced. To reduce class bias during training, the notebook balances the dataset using random oversampling of the minority class:

- Majority class: `Class = 0`
- Minority class: `Class = 1`
- Technique: `sklearn.utils.resample` with replacement to match majority count
- Final step: shuffle to remove ordering effects

### 2) Train-Test Split
After balancing, the dataset is split into:
- Test set: 20% (stratified)
- Pool set: 80% (used as the sampling pool)

This ensures evaluation is performed on a fixed hold-out test set while sampling strategies operate only on the pool.

### 3) Sample Size Calculation
The notebook computes the sample size using Cochran’s formula (95% confidence level, 5% margin of error, p = 0.5):

- Calculated Sample Size: 385

### 4) Sampling Techniques Implemented
Five sampling strategies are applied to the pool to generate five samples of size 385:

1. Simple Random Sampling
2. Stratified Sampling (equal draw per class from the pool)
3. Systematic Sampling (fixed interval selection)
4. Cluster Sampling (KMeans clustering into 10 clusters, then cluster selection until sample size is met)
5. Bootstrap Sampling (sampling with replacement)

### 5) Machine Learning Models Evaluated
Each sampled dataset is used to train the following models:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- Gradient Boosting Classifier

### 6) Evaluation Metric
- Metric: Accuracy (%)
- Evaluation: Predictions are made on the fixed hold-out test set.

## Results (Accuracy Percentage)

The following accuracy values are the exact outputs produced by the notebook:

| Model | Simple Random Sampling | Stratified Sampling | Systematic Sampling | Cluster Sampling | Bootstrap Sampling |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 93.137255 | 93.790850 | 92.483660 | 78.758170 | 93.464052 |
| Decision Tree | 98.039216 | 99.019608 | 97.385621 | 80.392157 | 97.712418 |
| Random Forest | 99.346405 | 99.673203 | 99.673203 | 81.699346 | 99.673203 |
| SVM | 68.627451 | 67.973856 | 69.607843 | 60.130719 | 68.954248 |
| Gradient Boosting | 98.692810 | 99.019608 | 99.346405 | 82.026144 | 98.692810 |

## Best Sampling Technique per Model (From the Results Above)

- Logistic Regression: Stratified Sampling (93.790850)
- Decision Tree: Stratified Sampling (99.019608)
- Random Forest: Stratified Sampling, Systematic Sampling, Bootstrap Sampling (tie at 99.673203)
- SVM: Systematic Sampling (69.607843)
- Gradient Boosting: Systematic Sampling (99.346405)

## Key Observations
- Stratified Sampling performs strongly for Logistic Regression, Decision Tree, and Random Forest.
- Systematic Sampling achieves the highest accuracy for SVM and Gradient Boosting.
- Cluster Sampling shows consistently lower accuracy across all models in this implementation, indicating that cluster selection via KMeans may reduce representativeness for this task.

## Notes
- This implementation uses accuracy as the evaluation metric to match assignment requirements.
- For imbalanced classification tasks in practice, additional metrics such as precision, recall, F1-score, and ROC-AUC are often evaluated alongside accuracy.

## Author
- Lakshay Sawhney
