# Neural Network Model Report for Alphabet Soup Charity Funding Prediction

## Overview of the Analysis

The purpose of this analysis was to develop a binary classifier to predict whether organizations funded by Alphabet Soup would successfully utilize their funding, using a dataset of over 34,000 organizations. The goal was to achieve a predictive accuracy greater than 75% using machine learning and neural networks.

The dataset included metadata such as application type, affiliation, classification, use case, organization type, status, income amount, special considerations, and requested funding amount. Through preprocessing, model design, and four optimization attempts, the analysis aimed to maximize the neural network’s predictive performance while considering alternative approaches to meet the accuracy target.

---

## Results

### Data Preprocessing

**Target Variable:**  
- `IS_SUCCESSFUL`: A binary indicator (0 or 1) representing whether an organization effectively used Alphabet Soup’s funding.

**Feature Variables:**  
All columns except `EIN`, `NAME`, `STATUS`, and `SPECIAL_CONSIDERATIONS` (in optimized models). Key features include:
- `APPLICATION_TYPE`: Type of application submitted
- `AFFILIATION`: Sector of industry affiliation
- `CLASSIFICATION`: Government organization classification
- `USE_CASE`: Purpose of funding
- `ORGANIZATION`: Type of organization
- `INCOME_AMT`: Income classification of the organization
- `ASK_AMT`: Requested funding amount (log-transformed or binned in optimized models)

**Removed Variables:**
- `EIN`, `NAME`: Non-predictive identifiers
- `STATUS`, `SPECIAL_CONSIDERATIONS`: Dropped in Optimized_2/3/4 due to low variance

---

### Model Architecture and Training

| Model         | Hidden Layers         | Activation      | Dropout | Notes |
|---------------|-----------------------|------------------|---------|-------|
| **Original**  | 80, 30                | relu / sigmoid   | None    | Baseline model |
| **Optimized_1**| 100, 50, 20           | tanh / sigmoid   | None    | Added complexity and different activation |
| **Optimized_2**| 120, 60, 30           | tanh / sigmoid   | 20%     | Dropout added, low-variance features removed |
| **Optimized_3**| 120, 60, 30           | relu / sigmoid   | 30%     | Improved activation, increased regularization |
| **Optimized_4**| 80, 40                | relu / sigmoid   | 30%     | Simplified, added EarlyStopping |

---

### Performance Metrics

| Model         | Accuracy   | Loss     |
|---------------|------------|----------|
| Original      | 72.63%     | 0.5578   |
| Optimized_1   | 72.23%     | 0.5768   |
| Optimized_2   | 73.08%     | 0.5516   |
| Optimized_3   | **73.13%** | 0.5533   |
| Optimized_4   | 72.99%     | 0.5551   |

**Target Accuracy:** >75% — *not achieved*

---

## Optimization Strategies

### Feature Engineering
- Binned `APPLICATION_TYPE` and `CLASSIFICATION` using thresholds (<200, <500, <1000) to reduce noise
- Log-transformed `ASK_AMT` (Optimized_1/2/3); binned into quantiles in Optimized_4
- Dropped `STATUS` and `SPECIAL_CONSIDERATIONS` due to low variance

### Architecture and Activation
- Varied layer count (2–3 layers) and neuron depth (up to 120)
- Tested `tanh` and `relu` activations
- Introduced Dropout layers (20–30%) to reduce overfitting

### Training Adjustments
- Epoch tuning: 100, 150, 50, and EarlyStopping (~15 epochs)
- Learning rate adjustment (0.0005 in Optimized_4)

---

## Summary

The neural network model reached a maximum test accuracy of **73.13%** with the Optimized_3 architecture. Although extensive tuning was performed—such as feature engineering, log transformation, activation function trials, and regularization techniques—the model failed to meet the 75% target.

### Final Thoughts

The plateau in performance suggests the dataset may have intrinsic complexity or noise limiting the neural network's predictive power. While neural networks are powerful, they may not always be the best fit for tabular datasets with mixed categorical and numerical data.

---

## Recommendation

**Try a Random Forest Classifier.**  
- **Advantages:**
  - Handles mixed feature types well
  - Robust to outliers and noise
  - Provides feature importance scores
  - Less prone to overfitting on tabular data

Tuning `n_estimators`, `max_depth`, and other hyperparameters may help surpass the 75% accuracy benchmark more reliably than deep neural networks in this context.
