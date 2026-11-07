# Purchase Predictor

A simple, privacy-friendly web application that estimates a customer's probability of purchase from gender, age, and estimated salary. The interface runs completely in the browser and is backed by the logistic-regression model trained in `LogisticRegression.ipynb`.

## Included files

- `index.html` — interactive, responsive web application; no server or dependencies required.
- `Purchase_Logistic.csv` — training dataset (400 observations).
- `LogisticRegression.ipynb` — exploratory analysis, preprocessing, training, and evaluation notebook.

## Quick start

1. Download or clone this repository.
2. Open `index.html` in a current browser.
3. Select gender and enter a valid age and estimated annual salary.
4. Select **Calculate likelihood** to see the probability and the classification at the 50% threshold.

No installation, API key, or internet connection is required.

## Input rules

The form validates all user entries before calculating.

| Field | Accepted values |
| --- | --- |
| Gender | Female or Male |
| Age | Whole number from 18 to 60 |
| Estimated salary | 15,000 to 150,000, in increments of 1,000 |

The numeric limits match the range represented in the supplied data. Rejecting values outside this range reduces the chance of presenting an ungrounded extrapolation as a prediction.

## Model details

The notebook reads the CSV, label-encodes gender (`Female = 0`, `Male = 1`), standardizes the three features with `StandardScaler`, and fits scikit-learn `LogisticRegression`. The web app embeds the scaler statistics and fitted model coefficients after training on the full dataset, then computes:

`sigmoid(intercept + coefficient × standardized feature values)`

The probability shown is therefore the model's probability for class `Purchased = 1`, not a manually invented score.

The notebook's evaluation split is 80/20 with `random_state=42`. Its reported test results are approximately: accuracy 90.0%, precision 91.7%, recall 78.6%, F1 84.6%, ROC-AUC 97.3%, and log loss 0.239.

## Design and privacy

- Responsive layout for desktop and mobile.
- Clear validation messages prevent empty, non-numeric, fractional-age, and out-of-range entries.
- No analytics, network calls, cookies, or persistence: entered data remains in the browser.
- Results include a plain-language interpretation alongside the exact probability.

