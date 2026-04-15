# Comparing Logistic Regression and Random Forest

Multiclass classification task comparing two ML algorithms on a dataset of 1100 samples with 36 numerical features and 4 target classes.

## Problem

Classify samples into one of four classes (0, 1, 2, 3) using supervised learning. The dataset is imbalanced — class 1 is the most frequent, class 3 the rarest. SMOTE was applied to handle class imbalance.

## Models & Results

Both models were tuned using `RandomizedSearchCV` with 5-fold cross-validation.

| Model | Best CV F1 (macro) | Test Accuracy |
|---|---|---|
| Logistic Regression | 0.65 | — |
| Random Forest | **0.89** | **0.72** |

**Logistic Regression** best params: `C=0.1, penalty=l2, solver=saga`

**Random Forest** best params: `n_estimators=200, max_depth=20, min_samples_split=5`

## Conclusion

Random Forest significantly outperforms Logistic Regression on this dataset, achieving higher F1 macro score and better precision-recall balance across all classes.

## Stack

Python 3.11 · scikit-learn · pandas · numpy · imbalanced-learn · matplotlib · seaborn
