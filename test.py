import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score

# Assuming df is your DataFrame and it has been defined already
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
features = ['Constituency ∇', 'Party', 'Criminal Case', 'Total Assets', 'Liabilities', 'state']
target = 'Education'

# Convert categorical variables to numeric
le = LabelEncoder()

# Combine the training and test data before fitting the LabelEncoder
combined = pd.concat([df[features], df_test[features]])

for feature in features:
    le.fit(combined[feature])
    df[feature] = le.transform(df[feature])
    df_test[feature] = le.transform(df_test[feature])

X = df[features]
y = le.fit_transform(df[target])
X_test = df_test[features]

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Perform grid search to find best hyperparameters
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
grid_search.fit(X, y)

# Get best model from grid search
best_rf = grid_search.best_estimator_

# Train best model on full training data
best_rf.fit(X, y)

# Predict on test set
predictions = best_rf.predict(X_test)

# Convert the numeric predictions back to the original classes
predictions = le.inverse_transform(predictions)

# Write the predictions to a CSV file
submission_df = pd.DataFrame({'ID': df_test['ID'], 'Education': predictions})
submission_df.to_csv('submission.csv', index=False)