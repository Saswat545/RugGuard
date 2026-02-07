# download_dataset_v2.py - Use Kaggle dataset instead

import pandas as pd
import requests
import zipfile
import os

print("Downloading Smart Contract Dataset...")
print("="*60)

# Create data folders
os.makedirs('data/contracts', exist_ok=True)
os.makedirs('data/scams', exist_ok=True)
os.makedirs('data/legit', exist_ok=True)

# We'll use a GitHub dataset (publicly available, no API limits)
# This is a real dataset used in academic papers

url = "https://raw.githubusercontent.com/smartbugs/smartbugs-wild/master/contracts/reentrancy.csv"

try:
    # Download contract list
    print("Downloading contract metadata...")
    df = pd.read_csv(url)
    print(f"✓ Downloaded {len(df)} contract records")
    
    # Save it
    df.to_csv('data/contracts/dataset.csv', index=False)
    print("✓ Saved to data/contracts/dataset.csv")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nLet's use Plan B: Manual dataset creation...")

print("="*60)