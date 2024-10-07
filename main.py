import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Function to convert currency to numeric
def currency_to_num(currency_str):
    return pd.to_numeric(currency_str.str.replace(r'[^\d.]', '', regex=True))

# Preprocessing the data
train_df['Total Assets'] = currency_to_num(train_df['Total Assets'])
train_df['Liabilities'] = currency_to_num(train_df['Liabilities'])

# Handling missing values
imputer = SimpleImputer(strategy='mean')
train_df[['Total Assets', 'Liabilities']] = imputer.fit_transform(train_df[['Total Assets', 'Liabilities']])

# Encoding the target variable and other categorical variables
label_encoder = LabelEncoder()
train_df['Education'] = label_encoder.fit_transform(train_df['Education'])

# Create dummies but make sure not to drop the 'Education' column yet
features_df = pd.get_dummies(train_df.drop(['ID', 'Candidate', 'Constituency ∇'], axis=1), drop_first=True)

# Splitting the dataset
X = features_df.drop('Education', axis=1)
y = features_df['Education']

X_train, _, y_train,  y_train, y_val= train_test_split(X, y, test_size=0.1, random_state=49)

# Model training with hyperparameter tuning
model = BernoulliNB()
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0, ]
}
grid_search = GridSearchCV(model, param_grid, cv=9, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

# Preprocess the test data
test_df['Total Assets'] = currency_to_num(test_df['Total Assets'])
test_df['Liabilities'] = currency_to_num(test_df['Liabilities'])
test_df[['Total Assets', 'Liabilities']] = imputer.transform(test_df[['Total Assets', 'Liabilities']])
test_df = pd.get_dummies(test_df.drop(['Candidate', 'Constituency ∇'], axis=1), drop_first=True)

# Predict on test data
test_predictions = grid_search.predict(test_df.drop('ID', axis=1))

# Create a DataFrame from test_predictions
submission_df = pd.DataFrame({
    'ID': test_df['ID'],  # Ensure 'ID' is included
    'Education': label_encoder.inverse_transform(test_predictions)
})

# Save to CSV
submission_df.to_csv('submissionbnb2.csv', index=False)
print("Submission saved!")
