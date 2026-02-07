# improve_dataset.py - Add more features to our dataset

import json
import os
import re

def extract_features(contract_code):
    """
    Extract numerical features from smart contract
    This is what the AI will learn from!
    """
    
    features = {}
    code_lower = contract_code.lower()
    
    # Feature 1: Count dangerous keywords
    features['has_mint'] = 1 if 'mint' in code_lower else 0
    features['has_drain'] = 1 if 'drain' in code_lower else 0
    features['has_withdraw'] = 1 if 'withdraw' in code_lower else 0
    features['has_selfdestruct'] = 1 if 'selfdestruct' in code_lower else 0
    features['has_blacklist'] = 1 if 'blacklist' in code_lower else 0
    
    # Feature 2: Owner control indicators
    features['has_owner'] = 1 if 'owner' in code_lower else 0
    features['has_onlyowner'] = 1 if 'onlyowner' in code_lower else 0
    features['owner_count'] = code_lower.count('owner')
    
    # Feature 3: Code quality indicators
    features['code_length'] = len(contract_code)
    features['has_comments'] = 1 if ('//' in contract_code or '/*' in contract_code) else 0
    features['function_count'] = code_lower.count('function')
    
    # Feature 4: Security features (good signs)
    features['has_max_supply'] = 1 if 'max_supply' in code_lower or 'maxsupply' in code_lower else 0
    features['has_require'] = code_lower.count('require')  # Input validation
    features['has_modifier'] = 1 if 'modifier' in code_lower else 0
    
    # Feature 5: Suspicious patterns
    features['has_private_function'] = 1 if 'private' in code_lower else 0
    features['has_payable'] = code_lower.count('payable')
    
    # Feature 6: Context - is mint combined with owner?
    features['mint_and_owner'] = 1 if (features['has_mint'] and features['has_owner']) else 0
    
    return features

def process_all_contracts():
    """Extract features from all contracts"""
    
    dataset = []
    
    # Process scam contracts
    scam_files = os.listdir('data/scams')
    for file in scam_files:
        if file.endswith('.json'):
            with open(f'data/scams/{file}', 'r') as f:
                contract = json.load(f)
            
            features = extract_features(contract['code'])
            features['name'] = contract['name']
            features['address'] = contract['address']
            features['label'] = 1  # 1 = SCAM
            
            dataset.append(features)
            print(f"✓ Processed scam: {contract['name']}")
    
    # Process legit contracts
    legit_files = os.listdir('data/legit')
    for file in legit_files:
        if file.endswith('.json'):
            with open(f'data/legit/{file}', 'r') as f:
                contract = json.load(f)
            
            features = extract_features(contract['code'])
            features['name'] = contract['name']
            features['address'] = contract['address']
            features['label'] = 0  # 0 = LEGIT
            
            dataset.append(features)
            print(f"✓ Processed legit: {contract['name']}")
    
    # Save processed dataset
    import pandas as pd
    df = pd.DataFrame(dataset)
    df.to_csv('data/processed_dataset.csv', index=False)
    
    print(f"\n✓ Saved {len(dataset)} contracts to data/processed_dataset.csv")
    print("\nFeatures extracted:")
    print(df.columns.tolist())
    
    return df

# Run it
print("Extracting features from contracts...")
print("="*60)
df = process_all_contracts()

print("\n" + "="*60)
print("📊 DATASET PREVIEW:")
print("="*60)
print(df[['name', 'has_mint', 'has_drain', 'has_owner', 'mint_and_owner', 'label']])