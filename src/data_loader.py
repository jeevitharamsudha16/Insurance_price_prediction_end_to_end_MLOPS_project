import pandas as pd
import os

def load_data(filename='insurance.csv', data_dir='data'):
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(df.head())
    print("✅ Data loaded successfully!")
    print(f"📄 Shape: {df.shape}")
    return df