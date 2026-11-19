# Project Summary

## Objective

Predict passenger recommendation and overall satisfaction from airline reviews, then assess whether combining structured service ratings with unstructured text improves predictive performance.

## Research questions

1. Which service dimensions are most associated with recommendation and overall rating?
2. Which classification and regression models perform best?
3. Do combined structured and text features outperform either source alone?
4. Do tree-based classifiers and regularised regression models outperform their linear baselines?

## Data and preprocessing

The project used approximately 23,000 public Kaggle airline reviews. Review titles and bodies were combined, normalised, tokenised, stripped of standard and airline-specific stopwords, POS-tagged, and lemmatised. TF-IDF features were limited to 2,000 unigram/bigram terms with minimum document frequency 5 and maximum document frequency 0.8. Four VADER sentiment components were appended.

Structured inputs included seven service ratings and one-hot encoded traveller, cabin, and verification categories. Missing sub-ratings were set to zero and missing categories to `Unknown`.

## Experiments

Three feature configurations were compared: structured-only, text-only, and combined. Recommendation models were Logistic Regression, Random Forest, and XGBoost. Rating models were Linear, Ridge, and Lasso Regression. The original report evaluated held-out predictions with classification and regression metrics, visual diagnostics, feature importance, and paired significance tests.

## Original conclusions

The group found that service ratings and review text provide complementary signals. Combined features were the strongest overall configuration, XGBoost was the leading classification approach, and Ridge Regression was the leading rating model. Cabin staff service and value-related measures were important themes in the interpretation.

## Limitations

- Public reviews are self-selected and do not represent all passengers.
- Airlines, routes, and time periods can introduce confounding.
- Missing sub-ratings encoded as zero may conflate absence with poor service.
- Lexicon sentiment does not fully capture sarcasm, context, or multilingual expression.
- Paired tests on one holdout split should be supplemented by repeated or nested cross-validation.
- Association and prediction do not establish causal service effects.

## Portfolio refinement

The cleaned pipeline splits records before fitting categorical encoding, TF-IDF, and scaling. This prevents information from the test set influencing feature construction. Local Colab paths, shell-only commands, student IDs, and stored notebook outputs were removed.
