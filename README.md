# Telco Customer Churn Prediction

Predicting which telecom customers are likely to cancel their subscription so retention teams can intervene before it happens.

**Live demo → [churn-predictior-app.streamlit.app](https://churn-predictior-app.streamlit.app)**

---

## Overview

Binary classification on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle) : 7,043 customers, 21 features, ~27% churn rate.

Since missing a churner is more costly than wrongly flagging a loyal customer, the primary metric is **Recall**, with F1 as a secondary check.

The model outputs a probability score, so the retention team can prioritize by risk level rather than just getting a binary yes/no.


---

## What I did

Started with a full EDA, the biggest finding was how much **contract type** drives churn. Month-to-month customers churn at 43% vs 3% for two-year contracts. Tenure tells a similar story: most churners are gone within the first 10 months.

Dropped `TotalCharges` (r=0.83 with tenure, redundant) and `gender` (26.9% vs 26.2% churn rate, basically noise). Added `auto_payment` as a binary feature, it ended up in LightGBM's top 15 importances.

Preprocessing handled with a `ColumnTransformer` : standard scaling for numerics, one-hot encoding for categoricals, fitted on train only. 80/20 stratified split, model selection via 5-fold CV on train only so the test set stayed untouched until the end. Tested 8 configurations (4 models × normal vs class-balanced).

---

## Results

| Metric | LightGBM (Balanced) | Logistic Regression (Balanced) |
|---|---:|---:|
| Accuracy | **0.75** | 0.73 |
| Recall (Churn) | 0.738 | **0.789** |
| Precision (Churn) | **0.52** | 0.50 |
| F1 (Churn) | **0.612** | 0.610 |
| ROC-AUC | 0.833 | **0.836** |

Both models perform very similarly. The choice depends on the use case:
- **LightGBM** → better F1 and precision, use when retention budget is limited
- **Logistic Regression** → higher Recall, use when missing a churner is the worst outcome

---

## Feature importance

Both models agree on the two strongest signals: **tenure** and **MonthlyCharges**.

LR surfaces `Contract_Two year` as the top coefficient (customers on longer contracts churn far less). 
LightGBM absorbs this into tenure, since long-tenure customers tend to be on longer contracts anyway. 
The `auto_payment` engineered feature appears in LightGBM's top 15, validating the decision to add it.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Limitations

- Static snapshot : the model doesn't capture trends over time (e.g. a customer whose bill has been increasing for 3 consecutive months)
- Risk thresholds (40% / 70%) are not calibrated to actual business costs

## Future improvements

**Soft-Voting Ensemble** : instead of picking a single winner, combine the predicted probabilities of all 4 balanced models. This smooths out individual model biases and generally improves stability, worth testing given how close the scores are.

---

## Stack

Python · pandas · numpy · scikit-learn · LightGBM · XGBoost · Streamlit · joblib