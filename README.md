# GradientBoosting-Diabetes
# Gradient Boosting Regression: Diabetes Dataset

## Overview
This project implements Gradient Boosting Regression on the scikit-learn diabetes dataset. The goal is to compare the performance of different weak learner complexities (stumps, 10-node trees, and 100-node trees) in predicting diabetes progression based on medical features.

## Dataset
The dataset used is the **Diabetes dataset** from `sklearn.datasets`, which consists of medical features to predict a continuous target variable (disease progression).

## Objective
- Train and evaluate Gradient Boosting models with different weak learners:
  - **Stumps**: `max_depth=1`
  - **10-node trees**: `max_depth=3`
  - **100-node trees**: `max_depth=7`
- Compare their test performance using **Mean Squared Error (MSE)**.
- Visualize the impact of weak learner complexity on model performance.
- Identify signs of overfitting with increasing complexity.

## Steps
1. **Load the dataset**
   - Use `load_diabetes()` from `sklearn.datasets`.
   - Split into 80% training and 20% testing.

2. **Train Gradient Boosting Models**
   - Use `GradientBoostingRegressor` with `n_estimators=200`.
   - Train models with different `max_depth` values (1, 3, 7).

3. **Evaluate Performance**
   - Compute **Mean Squared Error (MSE)** on test data.
   - Plot MSE against the number of estimators.
   - Use different colors to distinguish tree complexities:
     - Blue: Stumps (`max_depth=1`)
     - Red: 10-node trees (`max_depth=3`)
     - Green: 100-node trees (`max_depth=7`)

4. **Analyze Overfitting**
   - Observe where test MSE increases, indicating overfitting for deeper trees.
   - Compare with AdaBoost results from Exercise 1 (classification task).

## Installation
Ensure you have Python and the required libraries installed:
```bash
pip install numpy matplotlib scikit-learn
```

## Usage
Run the script:
```bash
python gradient_boosting_diabetes.py
```

## Results
- A line plot showing test MSE vs. number of estimators for different weak learners.
- Observations on how tree complexity affects performance and overfitting.

## Author
Developed as part of an exercise in boosting algorithms.

## License
This project is open-source under the MIT License.

