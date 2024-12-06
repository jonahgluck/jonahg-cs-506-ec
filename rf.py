import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Basic EDA and preprocessing
def preprocess_data(df, is_train=True):
    # Extract temporal features
    df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
    df['trans_hour'] = pd.to_datetime(df['trans_time'], format='%H:%M:%S', errors='coerce').dt.hour
    df['trans_day'] = df['trans_date'].dt.day
    df['trans_month'] = df['trans_date'].dt.month
    df['trans_year'] = df['trans_date'].dt.year
    
    # Compute distance between merchant and cardholder
    df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
    
    # Encode categorical features
    categorical_columns = ['category', 'gender', 'state', 'job']
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df.fillna(0, inplace=True)
    
    # Drop unnecessary columns
    drop_columns = ['trans_num', 'trans_date', 'trans_time', 'first', 'last', 'dob', 'merchant', 'street', 'city', 'zip']
    df.drop(columns=drop_columns, inplace=True)
    
    # Split target if training data
    if is_train:
        target = df['is_fraud']
        df.drop(columns=['is_fraud'], inplace=True)
        return df, target
    else:
        return df

# Preprocess datasets
X, y = preprocess_data(train_df)
X_test = preprocess_data(test_df, is_train=False)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
model.fit(X_train, y_train)

# Predictions
y_val_pred = model.predict(X_val)
f1 = f1_score(y_val, y_val_pred)

print("Validation F1 Score:", f1)
print("Classification Report:\n", classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# Test predictions and submission
test_predictions = model.predict(X_test)
submission = pd.DataFrame({'id': test_df['id'], 'is_fraud': test_predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file created.")

