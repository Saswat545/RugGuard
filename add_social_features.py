# add_social_features.py - Add social media indicators

import json
import os
import random

print("Adding social media features to contracts...")
print("="*60)

# Simulate social media data for each contract

scam_social_patterns = {
    "twitter_age_days": [5, 10, 15, 20, 30],  # New accounts
    "follower_count": [500, 1000, 2000, 5000],  # Bought followers
    "hype_score": [0.8, 0.85, 0.9, 0.95, 1.0],  # High hype language
    "team_verified": [False, False, False, True],  # Usually not verified
    "whitepaper_plagiarism": [0.7, 0.8, 0.9, 0.95],  # Copied content
}

legit_social_patterns = {
    "twitter_age_days": [365, 500, 730, 1000, 1500],  # Established
    "follower_count": [10000, 50000, 100000, 500000],  # Organic growth
    "hype_score": [0.1, 0.2, 0.3, 0.4, 0.5],  # Professional tone
    "team_verified": [True, True, True, True],  # Verified team
    "whitepaper_plagiarism": [0.0, 0.05, 0.1, 0.15],  # Original content
}

# Add social features to scam contracts
scam_files = sorted([f for f in os.listdir('data/scams') if f.endswith('.json')])
for file in scam_files:
    filepath = f'data/scams/{file}'
    with open(filepath, 'r') as f:
        contract = json.load(f)
    
    # Add social features
    contract['social'] = {
        'twitter_age_days': random.choice(scam_social_patterns['twitter_age_days']),
        'follower_count': random.choice(scam_social_patterns['follower_count']),
        'hype_score': random.choice(scam_social_patterns['hype_score']),
        'team_verified': random.choice(scam_social_patterns['team_verified']),
        'whitepaper_plagiarism': random.choice(scam_social_patterns['whitepaper_plagiarism']),
    }
    
    with open(filepath, 'w') as f:
        json.dump(contract, f, indent=2)
    
    print(f"✓ Added social features to {contract['name']}")

print()

# Add social features to legit contracts
legit_files = sorted([f for f in os.listdir('data/legit') if f.endswith('.json')])
for file in legit_files:
    filepath = f'data/legit/{file}'
    with open(filepath, 'r') as f:
        contract = json.load(f)
    
    # Add social features
    contract['social'] = {
        'twitter_age_days': random.choice(legit_social_patterns['twitter_age_days']),
        'follower_count': random.choice(legit_social_patterns['follower_count']),
        'hype_score': random.choice(legit_social_patterns['hype_score']),
        'team_verified': random.choice(legit_social_patterns['team_verified']),
        'whitepaper_plagiarism': random.choice(legit_social_patterns['whitepaper_plagiarism']),
    }
    
    with open(filepath, 'w') as f:
        json.dump(contract, f, indent=2)
    
    print(f"✓ Added social features to {contract['name']}")

print("="*60)
print("✓ Social media features added to all contracts")
print("="*60)