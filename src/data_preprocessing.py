import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_data(df):
    """
    Preprocesses the insurance dataset:
    - Drops missing values and duplicates
    - Encodes categorical variables
    - Splits into train/test
    - Scales 'age' and 'bmi' using MinMaxScaler
    - Saves the scaler to models/scalers/minmax_scaler.pkl

    Returns:
    - X_train, X_test, y_train, y_test, scaler
    """
    df = df.copy()

    # Drop missing and duplicate entries
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Encode 'smoker' as 0/1
    le = LabelEncoder()
    df['smoker'] = le.fit_transform(df['smoker'])

    # One-hot encode 'sex'
    df = pd.get_dummies(df, columns=['sex'], drop_first=True, dtype=int)

    # Encode 'region'
    region_map = {'southwest': 0, 'northwest': 1, 'northeast': 2, 'southeast': 3}
    df['region'] = df['region'].map(region_map)

    # Split features and target
    X = df.drop('charges', axis=1)
    y = df['charges']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numerical columns
    scaler = MinMaxScaler()
    X_train[['age', 'bmi']] = scaler.fit_transform(X_train[['age', 'bmi']]).astype(float)
    X_test[['age', 'bmi']] = scaler.transform(X_test[['age', 'bmi']]).astype(float)

    # Save the scaler using pathlib
    scaler_path = Path("models/scalers/minmax_scaler.pkl")
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    print(f"✅ Data preprocessing complete. Scaler saved to {scaler_path}")

    return X_train, X_test, y_train, y_test, scaler
