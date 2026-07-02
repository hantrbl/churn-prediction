

-------------------------------------------------------------------------------------

## Problem Definition

Telco loses revenue each time a customer churns. This project builds a 
binary classifier to predict which customers are likely to churn, using 
a static snapshot of 7,043 customers.

Since missing a churner (false negative) is far more costly than 
wrongly flagging a loyal customer (false positive), we optimize 
for **Recall** on the positive class, with F1 as secondary metric.
Accuracy is explicitly excluded as a primary metric due to class 
imbalance (~27% churn rate).

The model output is a probability score, enabling the retention team 
to prioritize interventions by risk level.

| Métrique          | LightGBM (Balanced) | Logistic Regression (Balanced) |
| ----------------- | ------------------: | -----------------------------: |
| Accuracy          |            **0.75** |                           0.73 |
| Recall (Churn)    |               0.738 |                      **0.789** |
| Precision (Churn) |            **0.52** |                           0.50 |
| F1 (Churn)        |           **0.612** |                          0.610 |
| ROC-AUC           |               0.833 |                      **0.836** |


### Future Improvements

**Soft-Voting Ensemble** : instead of picking a single winner, combine the predicted 
probabilities of all 4 balanced models into a single prediction. This smooths out 
individual model biases and generally improves stability — worth testing if the gap 
between models is small.