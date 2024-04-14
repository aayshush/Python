import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB

def custom_convert(amount):
    if 'Crore' in amount:
        return float(amount.replace(' Crore+', '')) * 10000  # 1 Crore = 10000 lacs
    elif 'Lac' in amount:
        return float(amount.replace(' Lac+', ''))   # Value is already in lacs
    elif 'Thou' in amount:
        return float(amount.replace(' Thou+', '')) / 100  # Convert thousands to lacs
    elif 'Hund' in amount:
        return float(amount.replace(' Hund+', '')) / 10000   # Convert hundreds to lacs
    else:
        return float(amount)

# Load the data
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Convert Total Assets and Liabilities to numerical values and apply Min-Max scaling
scaler = MinMaxScaler()
df['Total Assets'] = scaler.fit_transform(df['Total Assets'].apply(custom_convert).values.reshape(-1, 1))
df['Liabilities'] = scaler.fit_transform(df['Liabilities'].apply(custom_convert).values.reshape(-1, 1))

df_test['Total Assets'] = scaler.transform(df_test['Total Assets'].apply(custom_convert).values.reshape(-1, 1))
df_test['Liabilities'] = scaler.transform(df_test['Liabilities'].apply(custom_convert).values.reshape(-1, 1))

# Define features and target
features = ['Party', 'Criminal Record', 'Total Assets', 'Liabilities', 'State']
target = 'Education Level'

# Preprocessing for numerical features
numeric_features = ['Criminal Record', 'Total Assets', 'Liabilities']
numeric_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

# Preprocessing for categorical features
categorical_features = ['Party', 'State']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model (Bernoulli Naive Bayes)
custom_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', BernoulliNB())
])

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train the model on the training data
custom_model.fit(X_train, y_train)

# Make predictions on the training data
predictions_train = custom_model.predict(X_train)

# Make predictions on the test data
predictions_test = custom_model.predict(df_test[features])

# Write the predictions to a CSV file (for test data)
submission_df = pd.DataFrame({'ID': df_test['ID'], 'Education Level': predictions_test})
submission_df.to_csv('submission_custom.csv', index=False)
