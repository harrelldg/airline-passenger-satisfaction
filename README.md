# Airline Passenger Satisfaction and Recommendation

**Predictive Modeling of Passenger Satisfaction and Recommendation in the Airline Industry Based on Reviews**

This project predicts whether a passenger recommends an airline and their overall rating by combining structured service ratings with review text. It compares structured-only, text-only, and combined feature configurations across classification and regression models.

## Methods

- Structured features: seven service sub-ratings plus traveller type, seat type, and verification status.
- Text features: cleaned review title and body, TF-IDF unigrams/bigrams, and four VADER sentiment scores.
- Classification: Logistic Regression, Random Forest, and XGBoost for `Recommended`.
- Regression: Linear, Ridge, and Lasso Regression for `Overall_Rating`.
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, MAE, R², confusion matrices, residual analysis, feature importance, and paired significance tests.

The repository version fits one-hot encoding, TF-IDF, and scaling on training data only, correcting the preprocessing-before-split leakage in the original workflow.

## Data

Use the public [Airline Reviews dataset on Kaggle](https://www.kaggle.com/datasets/juhibhojani/airline-reviews). Download the CSV to `data/airline_reviews.csv`; see `data/README.md`. The dataset is not redistributed in this repository.

## Reproduce

```bash
python -m pip install -r requirements.txt
python src/airline_analysis.py --data data/airline_reviews.csv
```

The script downloads the required NLTK resources on first run. Text preprocessing and model comparison can take several minutes. Metrics and charts are produced locally.

## Documentation

- `docs/project-summary.md`: research questions, methodology, findings, and limitations.
- `docs/presentation-summary.md`: accessible summary of the project presentation.
- `src/airline_analysis.py`: cleaned, reproducible analysis pipeline.
