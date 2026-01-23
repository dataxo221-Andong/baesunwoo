import pandas as pd
import os

pkl_path = 'models/failure100.pkl'

if not os.path.exists(pkl_path):
    print(f"Error: {pkl_path} not found.")
    exit()

try:
    df = pd.read_pickle(pkl_path)
    print(f"Loaded DataFrame with shape: {df.shape}")
    print("Columns:", df.columns)
    
    if 'failureType' in df.columns:
        print("\nFirst 5 'failureType' values:")
        print(df['failureType'].head().values)
        
        print("\n'failureType' value counts (raw):")
        try:
            print(df['failureType'].value_counts())
        except Exception as e:
            print(f"Could not print value_counts: {e}")
            
        # Check structure of the first element
        first_val = df['failureType'].iloc[0]
        print(f"\nType of first element: {type(first_val)}")
        print(f"First element content: {first_val}")

        if isinstance(first_val, list) or isinstance(first_val, np.ndarray):
             if len(first_val) > 0:
                 print(f"df['failureType'].iloc[0][0] content: {first_val[0]}")
                 if isinstance(first_val[0], list) or isinstance(first_val[0], np.ndarray):
                      if len(first_val[0]) > 0:
                          print(f"df['failureType'].iloc[0][0][0] content: {first_val[0][0]}")
                
except Exception as e:
    print(f"An error occurred: {e}")
