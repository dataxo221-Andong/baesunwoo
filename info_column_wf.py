import pandas as pd
import os

def inspect_columns():
    pkl_path = 'models/ft_100.pkl'
    
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return

    print(f"Loading {pkl_path}...")
    try:
        df = pd.read_pickle(pkl_path)
        print(f"[Success] Data loaded. Shape: {df.shape}")
        
        print("\n" + "="*50)
        print(" [ Column Information ]")
        print("="*50)
        print(f"Columns: {df.columns.tolist()}")
        print("-" * 50)
        print(" [ Data Types ]")
        print(df.dtypes)
        
        print("\n" + "="*50)
        print(" [ Sample Data (First Row) ]")
        print("="*50)
        if len(df) > 0:
            first_row = df.iloc[0]
            for col in df.columns:
                val = first_row[col]
                # 웨이퍼 맵 같은 큰 데이터는 요약해서 출력
                if col == 'waferMap':
                    print(f"{col}: {type(val)} with shape {val.shape}")
                else:
                    print(f"{col}: {val}")
        else:
            print("DataFrame is empty.")
            
    except Exception as e:
        print(f"Error reading pickle: {e}")

if __name__ == "__main__":
    inspect_columns()
