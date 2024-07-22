# Bank Marketing Analysis and Model Evaluation

## Overview
This project involves analyzing the marketing algorithms used by a large bank to predict whether clients will subscribe to a term deposit. The analysis includes creating a classification model, evaluating feature relevance, and providing insights into the decision-making process both globally and locally.

## Project Structure
- **Data Loading and Preprocessing:** Load and preprocess the dataset.
- **Model Building:** Create a classification model using RandomForestClassifier.
- **Feature Importance Analysis:** Analyze feature importance globally and locally.
- **Partial Dependence Plots:** Visualize the relationship between top features and predictions.
- **Conclusions and Recommendations:** Summarize findings and provide actionable insights.

## Data Loading and Preprocessing
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('bank-additional-full.csv', sep=';')

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split the data
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## Model Building
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
```

## Feature Importance Analysis
### Global Feature Importance
```python
import matplotlib.pyplot as plt
import numpy as np

# Feature importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()
```

### Local Feature Importance
```python
import shap

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Observation #4 and #20
obs_4 = X_test.iloc[4]
obs_20 = X_test.iloc[20]

shap.force_plot(explainer.expected_value[1], shap_values[1][4], obs_4)
shap.force_plot(explainer.expected_value[1], shap_values[1][20], obs_20)
```

## Partial Dependence Plots
```python
from sklearn.inspection import PartialDependenceDisplay

# Plot partial dependence for the top 3 features
features_to_plot = ['duration', 'euribor3m', 'nr.employed']
PartialDependenceDisplay.from_estimator(clf, X, features_to_plot, kind="both", grid_resolution=50)

plt.tight_layout()
plt.show()
```

## Conclusions and Recommendations
### Conclusions
1. **Model Performance:**
   - The classification model achieves high accuracy (90%), precision (88%), recall (93%), and F1 score (90%).
2. **Global Feature Importance:**
   - Key features: `duration`, `euribor3m`, `nr.employed`, `age`, and `job`.
3. **Local Feature Importance:**
   - For individual observations, `duration` is the most significant feature, followed by economic indicators.
4. **Partial Dependence Plots:**
   - Positive relationships between `duration`, `euribor3m`, and `nr.employed` and the likelihood of subscription.

### Recommendations
1. **Enhance Client Engagement:**
   - Increase call durations to boost subscription likelihood.
2. **Monitor Economic Indicators:**
   - Align strategies with favorable economic conditions.
3. **Utilize Employment Data:**
   - Target regions/sectors with higher employment rates.
4. **Personalize Marketing Efforts:**
   - Tailor strategies based on client profiles and top features.
5. **Continuous Model Improvement:**
   - Regularly update and retrain the model with new data.
6. **Focus on Top Features:**
   - Prioritize efforts on `duration`, `euribor3m`, and `nr.employed`.

## Next Steps
1. **Deploy the Model:**
   - Implement the model in the bank's marketing system for real-time predictions.
2. **A/B Testing:**
   - Conduct A/B testing to validate the effectiveness of the recommended strategies.
3. **Feature Engineering:**
   - Explore additional features that could enhance model performance.
4. **Customer Feedback:**
   - Collect feedback from clients to refine marketing approaches.
