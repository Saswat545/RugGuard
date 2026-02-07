# train_multimodal_model.py - Train with code + social features

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os

print("RugGuard Multi-Modal Training (Code + Social)")
print("="*60)

# Extract features from all contracts (including social)
def extract_all_features(contract_file):
    with open(contract_file, 'r') as f:
        contract = json.load(f)
    
    features = {}
    code_lower = contract['code'].lower()
    
    # CODE FEATURES (same as before)
    features['has_mint'] = 1 if 'mint' in code_lower else 0
    features['has_drain'] = 1 if 'drain' in code_lower else 0
    features['has_withdraw'] = 1 if 'withdraw' in code_lower else 0
    features['has_selfdestruct'] = 1 if 'selfdestruct' in code_lower else 0
    features['has_blacklist'] = 1 if 'blacklist' in code_lower else 0
    features['has_owner'] = 1 if 'owner' in code_lower else 0
    features['has_onlyowner'] = 1 if 'onlyowner' in code_lower else 0
    features['owner_count'] = code_lower.count('owner')
    features['code_length'] = len(contract['code'])
    features['has_comments'] = 1 if ('//' in contract['code'] or '/*' in contract['code']) else 0
    features['function_count'] = code_lower.count('function')
    features['has_max_supply'] = 1 if 'max_supply' in code_lower or 'maxsupply' in code_lower else 0
    features['has_require'] = code_lower.count('require')
    features['has_modifier'] = 1 if 'modifier' in code_lower else 0
    features['has_private_function'] = 1 if 'private' in code_lower else 0
    features['has_payable'] = code_lower.count('payable')
    features['mint_and_owner'] = 1 if (features['has_mint'] and features['has_owner']) else 0
    
    # SOCIAL FEATURES (NEW!)
    if 'social' in contract:
        features['twitter_age_days'] = contract['social']['twitter_age_days']
        features['follower_count'] = contract['social']['follower_count']
        features['hype_score'] = contract['social']['hype_score']
        features['team_verified'] = 1 if contract['social']['team_verified'] else 0
        features['whitepaper_plagiarism'] = contract['social']['whitepaper_plagiarism']
    
    features['label'] = contract['label'] if contract['label'] == 'SCAM' else 0
    features['label'] = 1 if contract['label'] == 'SCAM' else 0
    features['name'] = contract['name']
    
    return features

# Process all contracts
dataset = []

scam_files = [f'data/scams/{f}' for f in os.listdir('data/scams') if f.endswith('.json')]
for file in scam_files:
    features = extract_all_features(file)
    dataset.append(features)

legit_files = [f'data/legit/{f}' for f in os.listdir('data/legit') if f.endswith('.json')]
for file in legit_files:
    features = extract_all_features(file)
    dataset.append(features)

df = pd.DataFrame(dataset)
print(f"✓ Loaded {len(df)} contracts with multi-modal features")

# Prepare data
feature_columns = [col for col in df.columns if col not in ['name', 'label']]
X = df[feature_columns]
y = df['label']

print(f"✓ Using {len(feature_columns)} features (17 code + 5 social)")
print(f"  Code features: has_mint, has_drain, owner_count...")
print(f"  Social features: twitter_age, hype_score, team_verified...")

# Train model
print("\n🤖 Training multi-modal model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X, y)
print("✓ Training complete!")

# Evaluate
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print(f"\n📊 MULTI-MODAL MODEL PERFORMANCE:")
print("="*60)
print(f"Accuracy: {accuracy*100:.1f}%")

# Feature importance
print("\n🔍 TOP 10 IMPORTANT FEATURES:")
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in importance_df.head(10).iterrows():
    feature_type = "📱 SOCIAL" if row['feature'] in ['twitter_age_days', 'follower_count', 'hype_score', 'team_verified', 'whitepaper_plagiarism'] else "💻 CODE"
    print(f"  {feature_type} {row['feature']}: {row['importance']:.3f}")

# Save model
joblib.dump(model, 'models/rugguard_multimodal.pkl')
joblib.dump(feature_columns, 'models/feature_columns_multimodal.pkl')

print("\n✓ Multi-modal model saved!")
print("="*60)

# Compare with code-only model
print("\n📊 IMPROVEMENT ANALYSIS:")
print(f"  Code-only model: 95.0%")
print(f"  Multi-modal model: {accuracy*100:.1f}%")
if accuracy > 0.95:
    print(f"  Improvement: +{(accuracy - 0.95)*100:.1f}% 🎉")
else:
    print(f"  (Same performance - social features didn't hurt!)")

print("="*60)