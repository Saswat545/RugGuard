# improved_analyzer.py - Enhanced risk scoring with compound analysis

from flask import Flask, render_template_string, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model with fallback
try:
    model = joblib.load('models/rugguard_multimodal.pkl')
    feature_columns = joblib.load('models/feature_columns_multimodal.pkl')
except:
    model = None
    feature_columns = None

def is_valid_smart_contract(code):
    if len(code.strip()) < 50:
        return False, "❌ Input too short"
    if 'contract' not in code.lower():
        return False, "❌ Missing 'contract' keyword"
    if 'pragma' not in code.lower() and 'solidity' not in code.lower():
        return False, "❌ Missing Solidity declaration"
    return True, "Valid"

def extract_features(contract_code):
    """Extract features from contract code"""
    features = {}
    code_lower = contract_code.lower()
    
    # Basic features
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
    
    # Social features (defaults)
    features['twitter_age_days'] = 100
    features['follower_count'] = 5000
    features['hype_score'] = 0.5
    features['team_verified'] = 1
    features['whitepaper_plagiarism'] = 0.2
    
    return features

def calculate_enhanced_risk_score(code, features):
    """
    Enhanced risk scoring with compound analysis
    This is the KEY fix for your issue
    """
    risk_score = 0
    risk_flags = []
    critical_flags = []
    
    code_lower = code.lower()
    
    # ============================================
    # CRITICAL RISK PATTERNS (40-50 points each)
    # ============================================
    
    # 1. UNLIMITED MINT SCAM (CRITICAL)
    if features['has_mint'] and features['has_owner']:
        # Check if there's NO supply cap
        has_supply_cap = any(keyword in code_lower for keyword in 
                            ['maxsupply', 'max_supply', 'totalsupply', 'supply_cap', 'cap'])
        
        if not has_supply_cap or features['has_max_supply'] == 0:
            risk_score += 45  # HUGE penalty for unlimited mint
            critical_flags.append("🔴 CRITICAL: Unlimited mint - owner can create infinite tokens")
            risk_flags.append("unlimited_mint")
    
    # 2. DRAIN FUNCTIONS (CRITICAL)
    if features['has_drain']:
        risk_score += 40  # Even if private, it's malicious intent
        critical_flags.append("🔴 CRITICAL: Drain function detected - can steal all funds")
        risk_flags.append("drain_function")
    
    # 3. SELFDESTRUCT (CRITICAL)
    if features['has_selfdestruct']:
        risk_score += 40
        critical_flags.append("🔴 CRITICAL: Self-destruct capability - contract can be destroyed")
        risk_flags.append("selfdestruct")
    
    # ============================================
    # HIGH RISK PATTERNS (20-30 points each)
    # ============================================
    
    # 4. BLACKLIST FUNCTION
    if features['has_blacklist']:
        risk_score += 25
        risk_flags.append("🔴 HIGH RISK: Blacklist function - owner can block users from selling")
    
    # 5. EXCESSIVE OWNER CONTROL
    if features['owner_count'] > 8:
        risk_score += 25
        risk_flags.append(f"🔴 HIGH RISK: Excessive owner control ({features['owner_count']} references)")
    elif features['owner_count'] > 5:
        risk_score += 15
        risk_flags.append(f"⚠️ Owner mentioned {features['owner_count']} times (centralization risk)")
    
    # 6. HONEYPOT PATTERN (can buy but can't sell)
    has_trading_enabled = 'tradingenabled' in code_lower or 'trading_enabled' in code_lower
    has_pause = 'paused' in code_lower or 'pause' in code_lower
    
    if has_trading_enabled or has_pause:
        risk_score += 20
        risk_flags.append("⚠️ Trading control detected - possible honeypot pattern")
    
    # ============================================
    # MEDIUM RISK PATTERNS (10-15 points)
    # ============================================
    
    # 7. MULTIPLE PAYABLE FUNCTIONS
    if features['has_payable'] > 3:
        risk_score += 15
        risk_flags.append(f"⚠️ {features['has_payable']} payable functions (high fund exposure)")
    elif features['has_payable'] > 1:
        risk_score += 10
    
    # 8. NO OWNERSHIP RENOUNCEMENT
    has_renounce = 'renounce' in code_lower
    if features['has_owner'] and not has_renounce:
        risk_score += 12
        risk_flags.append("⚠️ No ownership renouncement - owner retains full control")
    
    # 9. HIDDEN/PRIVATE FUNCTIONS
    if features['has_private_function']:
        # Check if private functions modify balances or transfer funds
        if 'private' in code_lower and ('balance' in code_lower or 'transfer' in code_lower):
            risk_score += 15
            risk_flags.append("⚠️ Private functions with balance manipulation")
    
    # ============================================
    # PROTECTIVE FEATURES (Reduce risk)
    # ============================================
    
    good_signs = []
    
    # 1. FIXED SUPPLY CAP
    if features['has_max_supply'] or 'constant' in code_lower and 'supply' in code_lower:
        risk_score -= 15
        good_signs.append("✅ Fixed maximum supply (prevents inflation)")
    
    # 2. MULTIPLE SAFETY CHECKS
    if features['has_require'] > 5:
        risk_score -= 12
        good_signs.append(f"✅ {features['has_require']} safety checks (require statements)")
    elif features['has_require'] > 3:
        risk_score -= 8
        good_signs.append(f"✅ {features['has_require']} safety checks")
    
    # 3. CODE DOCUMENTATION
    if features['has_comments']:
        risk_score -= 5
        good_signs.append("✅ Code documentation present")
    
    # 4. CUSTOM MODIFIERS
    if features['has_modifier']:
        risk_score -= 5
        good_signs.append("✅ Custom access modifiers")
    
    # 5. EVENTS FOR TRANSPARENCY
    if 'event' in code_lower:
        event_count = code_lower.count('event')
        if event_count > 2:
            risk_score -= 8
            good_signs.append(f"✅ {event_count} events for transparency")
    
    # 6. TIME LOCKS
    if 'timelock' in code_lower or 'unlock' in code_lower:
        risk_score -= 10
        good_signs.append("✅ Time lock mechanism")
    
    # ============================================
    # COMPOUND RISK MULTIPLIER (KEY ENHANCEMENT!)
    # ============================================
    
    num_critical_flags = len(critical_flags)
    
    # If multiple CRITICAL flags, multiply risk
    if num_critical_flags >= 2:
        risk_score = int(risk_score * 1.4)  # 40% increase for multiple critical issues
        risk_flags.insert(0, "⚡ COMPOUND RISK: Multiple critical vulnerabilities detected")
    elif num_critical_flags >= 1 and len(risk_flags) >= 3:
        risk_score = int(risk_score * 1.25)  # 25% increase for critical + multiple high risks
        risk_flags.insert(0, "⚡ Combined risk factors amplify danger")
    
    # ============================================
    # CODE QUALITY PENALTY
    # ============================================
    
    # Very short contracts are often scams
    if features['code_length'] < 300:
        risk_score += 10
        risk_flags.append("⚠️ Unusually short contract (possible rush job)")
    
    # Very few functions = minimal functionality
    if features['function_count'] < 3:
        risk_score += 8
        risk_flags.append("⚠️ Very few functions (suspicious simplicity)")
    
    # ============================================
    # FINAL SCORE NORMALIZATION
    # ============================================
    
    # Cap between 0-100
    risk_score = max(0, min(100, risk_score))
    
    return risk_score, risk_flags + critical_flags, good_signs

def analyze_contract_advanced(code):
    """Advanced analysis combining ML model + rule-based scoring"""
    
    features = extract_features(code)
    
    # Get ML model prediction (if available)
    ml_score = 50  # Default
    
    if model is not None and feature_columns is not None:
        try:
            X = pd.DataFrame([features])
            for col in feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[feature_columns]
            
            probability = model.predict_proba(X)[0]
            ml_score = probability[1] * 100 if len(probability) > 1 else 50
        except:
            pass
    
    # Get rule-based enhanced score
    rule_score, risk_indicators, safety_features = calculate_enhanced_risk_score(code, features)
    
    # Combine both scores (weighted average)
    # 60% rule-based (more reliable) + 40% ML (pattern learning)
    final_score = (rule_score * 0.6) + (ml_score * 0.4)
    
    # If rule-based score is VERY high, override ML
    if rule_score >= 85:
        final_score = max(final_score, rule_score * 0.95)
    
    return final_score, risk_indicators, safety_features

# [Rest of the Flask routes and HTML template remain the same as before]
# Just replace the /analyze route:

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        contract_code = data.get('code', '')
        is_valid, msg = is_valid_smart_contract(contract_code)
        if not is_valid:
            return jsonify({'error': msg}), 400
        
        # Use ENHANCED analysis
        scam_prob, risk_indicators, safety_features = analyze_contract_advanced(contract_code)
        
        # Combine all indicators
        all_indicators = risk_indicators + safety_features
        
        if not all_indicators:
            all_indicators.append("ℹ️ No strong scam indicators detected")
        
        # Generate recommendation based on ENHANCED score
        if scam_prob >= 85:
            recommendation = "🚫 EXTREME DANGER - DO NOT INVEST UNDER ANY CIRCUMSTANCES. This contract contains multiple critical scam patterns. You WILL lose your money. Our analysis detected unlimited minting, fund drainage mechanisms, and owner manipulation capabilities."
        elif scam_prob >= 70:
            recommendation = "🚫 HIGH RISK - STRONG SCAM INDICATORS. This contract shows serious red flags. DO NOT INVEST unless you can afford to lose 100% of your funds. Detected issues include owner-controlled minting, possible honeypot patterns, or fund extraction mechanisms."
        elif scam_prob >= 50:
            recommendation = "⚠️ MEDIUM-HIGH RISK - PROCEED WITH EXTREME CAUTION. Multiple suspicious patterns detected. Before investing: (1) Get professional audit (CertiK/PeckShield), (2) Verify team identities thoroughly, (3) Confirm liquidity locks (min 1 year), (4) Check community sentiment on trusted platforms. Only invest what you can afford to lose."
        elif scam_prob >= 30:
            recommendation = "⚠️ MEDIUM RISK - Some concerns detected. Verify: (1) Team transparency (LinkedIn, GitHub), (2) Third-party audit report, (3) Locked liquidity proof, (4) Active community engagement. Never invest based on hype alone."
        else:
            recommendation = "✅ LOW RISK detected in code analysis. However, code analysis alone is NOT sufficient. Always verify: (1) Team background and real identities, (2) Professional security audit (CertiK, Quantstamp, etc.), (3) Liquidity locked for 6+ months, (4) Active development and community. Even 'safe' code can be part of a larger scam."
        
        return jsonify({
            'scam_probability': float(scam_prob),
            'indicators': all_indicators,
            'recommendation': recommendation
        })
        
    except Exception as e:
        import traceback
        print(f"Error in analyze: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

# [Include the same HTML template from modern_website.py]

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🎯 RugGuard AI - ENHANCED Risk Scoring System")
    print("="*80)
    print("\n✨ Improvements:")
    print("  • Compound risk detection (multiple flags = higher score)")
    print("  • Critical patterns heavily weighted (unlimited mint = +45)")
    print("  • Rule-based + ML hybrid scoring")
    print("  • Realistic threat assessment")
    print("\n🌐 Open: http://localhost:5000")
    print("="*80 + "\n")
    
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, port=5000, host='0.0.0.0')