# Presentation Summary

## Problem

Airlines receive large volumes of reviews containing both numerical service ratings and detailed passenger narratives. The project evaluates how these sources can be used together to predict recommendation and satisfaction.

## Workflow

1. Inspect and clean approximately 23,000 reviews.
2. Convert recommendation to a binary target and overall rating to numeric form.
3. Combine review titles and bodies.
4. Build structured, TF-IDF, and VADER sentiment features.
5. Compare three feature configurations across three classifiers and three regressors.
6. Interpret performance, feature importance, and statistical comparisons.

## Models

- Recommendation: Logistic Regression, Random Forest, XGBoost.
- Overall rating: Linear Regression, Ridge Regression, Lasso Regression.

## Evaluation

Classification used accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices. Regression used MAE, R², actual-versus-predicted plots, and residual plots. Additional analyses compared feature configurations and model families.

## Takeaways

The presentation emphasised the complementary value of service ratings and review text, the strong performance of combined features, and the operational potential of identifying service dimensions associated with passenger advocacy and satisfaction.

This is an accessible summary of the original slides; personal identifiers and private collaboration links have been removed.
