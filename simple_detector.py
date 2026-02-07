# simple_detector.py - Our first scam detector!

import json
import os
import re

print("RugGuard - Simple Smart Contract Scam Detector")
print("="*60)

# Define scam patterns (keywords that indicate scams)
SCAM_KEYWORDS = {
    'mint': 10,      # High risk: unlimited token creation
    'drain': 10,     # High risk: fund theft
    'withdraw': 5,   # Medium risk: might be legitimate withdrawal
    'onlyOwner': 7,  # Medium-high risk: owner privileges
    'selfdestruct': 10,  # High risk: contract destruction
    'private': 3,    # Low risk: might hide something
    'blacklist': 8,  # High risk: can block users
    'pause': 4,      # Low-medium risk: legitimate feature sometimes
}

def analyze_contract(contract_code):
    """
    Analyze smart contract code for scam indicators
    Returns risk score (0-100)
    """
    risk_score = 0
    found_indicators = []
    
    # Convert code to lowercase for easier matching
    code_lower = contract_code.lower()
    
    # Check for each scam keyword
    for keyword, weight in SCAM_KEYWORDS.items():
        if keyword.lower() in code_lower:
            risk_score += weight
            found_indicators.append(f"{keyword} (risk: {weight})")
    
    # Additional checks
    
    # Check 1: Very short contract (might be rushed scam)
    if len(contract_code) < 500:
        risk_score += 5
        found_indicators.append("Very short contract (risk: 5)")
    
    # Check 2: No comments (poor documentation)
    if '//' not in contract_code and '/*' not in contract_code:
        risk_score += 3
        found_indicators.append("No code comments (risk: 3)")
    
    # Check 3: Multiple owner functions
    owner_count = code_lower.count('owner')
    if owner_count > 5:
        risk_score += owner_count
        found_indicators.append(f"Many owner references (risk: {owner_count})")
    
    # Cap at 100
    risk_score = min(risk_score, 100)
    
    return risk_score, found_indicators

def predict_scam(contract_file):
    """Load contract and predict if it's a scam"""
    
    with open(contract_file, 'r') as f:
        contract = json.load(f)
    
    # Analyze the code
    risk_score, indicators = analyze_contract(contract['code'])
    
    # Make prediction
    if risk_score >= 30:
        prediction = "SCAM"
        confidence = min(risk_score, 95)  # Don't claim 100% certainty
    elif risk_score >= 15:
        prediction = "SUSPICIOUS"
        confidence = risk_score
    else:
        prediction = "LIKELY SAFE"
        confidence = 100 - risk_score
    
    return {
        'name': contract['name'],
        'address': contract['address'],
        'prediction': prediction,
        'risk_score': risk_score,
        'confidence': confidence,
        'indicators': indicators,
        'true_label': contract.get('label', 'UNKNOWN')
    }

# Test on all contracts
print("\n📊 TESTING ON SCAM CONTRACTS:")
print("-"*60)

scam_files = [f'data/scams/{f}' for f in os.listdir('data/scams') if f.endswith('.json')]
scam_predictions = []

for file in scam_files:
    result = predict_scam(file)
    scam_predictions.append(result)
    
    print(f"\n✓ Contract: {result['name']}")
    print(f"  Address: {result['address']}")
    print(f"  TRUE LABEL: {result['true_label']}")
    print(f"  PREDICTION: {result['prediction']} (risk: {result['risk_score']}/100)")
    print(f"  Indicators found:")
    for indicator in result['indicators'][:5]:  # Show top 5
        print(f"    - {indicator}")

print("\n" + "="*60)
print("\n📊 TESTING ON LEGITIMATE CONTRACTS:")
print("-"*60)

legit_files = [f'data/legit/{f}' for f in os.listdir('data/legit') if f.endswith('.json')]
legit_predictions = []

for file in legit_files:
    result = predict_scam(file)
    legit_predictions.append(result)
    
    print(f"\n✓ Contract: {result['name']}")
    print(f"  Address: {result['address']}")
    print(f"  TRUE LABEL: {result['true_label']}")
    print(f"  PREDICTION: {result['prediction']} (risk: {result['risk_score']}/100)")
    if result['indicators']:
        print(f"  Indicators found:")
        for indicator in result['indicators'][:3]:
            print(f"    - {indicator}")
    else:
        print(f"  ✓ No scam indicators found!")

# Calculate accuracy
print("\n" + "="*60)
print("📈 MODEL PERFORMANCE:")
print("="*60)

correct = 0
total = len(scam_predictions) + len(legit_predictions)

# Check scams
for pred in scam_predictions:
    if pred['prediction'] in ['SCAM', 'SUSPICIOUS']:
        correct += 1
        
# Check legit
for pred in legit_predictions:
    if pred['prediction'] == 'LIKELY SAFE':
        correct += 1

accuracy = (correct / total) * 100

print(f"Total contracts tested: {total}")
print(f"Correctly classified: {correct}")
print(f"Accuracy: {accuracy:.1f}%")
print("="*60)

if accuracy >= 75:
    print("🎉 Great! Your detector works!")
elif accuracy >= 50:
    print("👍 Good start! We'll improve this with ML.")
else:
    print("📚 Needs work, but that's expected for v1!")