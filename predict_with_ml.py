# predict_with_ml.py - Use our trained AI!

import joblib
import json
import pandas as pd

# Load trained model
model = joblib.load('models/rugguard_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

print("RugGuard AI - Smart Scam Detector")
print("="*60)

def extract_features(contract_code):
    """Same as before"""
    features = {}
    code_lower = contract_code.lower()
    
    features['has_mint'] = 1 if 'mint' in code_lower else 0
    features['has_drain'] = 1 if 'drain' in code_lower else 0
    features['has_withdraw'] = 1 if 'withdraw' in code_lower else 0
    features['has_selfdestruct'] = 1 if 'selfdestruct' in code_lower else 0
    features['has_blacklist'] = 1 if 'blacklist' in code_lower else 0
    features['has_owner'] = 1 if 'owner' in code_lower else 0
    features['has_onlyowner'] = 1 if 'onlyowner' in code_lower else 0
    features['owner_count'] = code_lower.count('owner')
    features['code_length'] = len(contract_code)
    features['has_comments'] = 1 if ('//' in contract_code or '/*' in contract_code) else 0
    features['function_count'] = code_lower.count('function')
    features['has_max_supply'] = 1 if 'max_supply' in code_lower or 'maxsupply' in code_lower else 0
    features['has_require'] = code_lower.count('require')
    features['has_modifier'] = 1 if 'modifier' in code_lower else 0
    features['has_private_function'] = 1 if 'private' in code_lower else 0
    features['has_payable'] = code_lower.count('payable')
    features['mint_and_owner'] = 1 if (features['has_mint'] and features['has_owner']) else 0
    
    return features

def predict_scam_ml(contract_file):
    """Predict using ML model"""
    
    with open(contract_file, 'r') as f:
        contract = json.load(f)
    
    # Extract features
    features = extract_features(contract['code'])
    
    # Convert to DataFrame (what sklearn expects)
    X = pd.DataFrame([features])[feature_columns]
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    result = {
        'name': contract['name'],
        'address': contract['address'],
        'prediction': 'SCAM' if prediction == 1 else 'LEGIT',
        'scam_probability': probability[1] * 100,
        'legit_probability': probability[0] * 100,
        'true_label': contract.get('label', 'UNKNOWN')
    }
    
    return result

# Test on all contracts
import os

print("\n🧪 TESTING ML MODEL:")
print("="*60)

all_files = []
all_files += [f'data/scams/{f}' for f in os.listdir('data/scams') if f.endswith('.json')]
all_files += [f'data/legit/{f}' for f in os.listdir('data/legit') if f.endswith('.json')]

correct = 0
total = 0

for file in all_files:
    result = predict_scam_ml(file)
    total += 1
    
    is_correct = (
        (result['true_label'] == 'SCAM' and result['prediction'] == 'SCAM') or
        (result['true_label'] == 'LEGIT' and result['prediction'] == 'LEGIT')
    )
    
    if is_correct:
        correct += 1
        status = "✓ CORRECT"
    else:
        status = "✗ WRONG"
    
    print(f"\n{status}: {result['name']}")
    print(f"  True: {result['true_label']} | Predicted: {result['prediction']}")
    print(f"  Scam Probability: {result['scam_probability']:.1f}%")

print("\n" + "="*60)
print(f"📈 ACCURACY: {(correct/total)*100:.1f}% ({correct}/{total})")
print("="*60)