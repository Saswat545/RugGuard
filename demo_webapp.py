# demo_webapp.py - FIXED VERSION with working buttons

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re

app = Flask(__name__)

# Load trained model
model = joblib.load('models/rugguard_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

def is_valid_smart_contract(code):
    """Validate if input is actually a smart contract"""
    
    if len(code.strip()) < 50:
        return False, "❌ Input too short - Smart contracts are typically 50+ characters"
    
    if 'contract' not in code.lower():
        return False, "❌ Missing 'contract' keyword - This doesn't appear to be a smart contract"
    
    has_pragma = 'pragma' in code.lower()
    has_solidity = 'solidity' in code.lower()
    
    if not (has_pragma or has_solidity):
        return False, "❌ Missing Solidity declaration - Not a valid smart contract"
    
    has_function = 'function' in code.lower()
    has_mapping = 'mapping' in code.lower()
    has_constructor = 'constructor' in code.lower()
    
    if not (has_function or has_mapping or has_constructor):
        return False, "❌ No functions or mappings found - Doesn't look like executable code"
    
    spam_keywords = ['invitation', 'birthday', 'wedding', 'party', 'anniversary', 
                     'recipe', 'ingredients', 'grocery', 'shopping list',
                     'dear', 'sincerely', 'regards']
    
    code_lower = code.lower()
    for keyword in spam_keywords:
        if keyword in code_lower:
            return False, f"❌ Detected '{keyword}' - This looks like {keyword} text, not code!"
    
    programming_symbols = ['{', '}', '(', ')', ';']
    symbol_count = sum(code.count(symbol) for symbol in programming_symbols)
    
    if symbol_count < 5:
        return False, "❌ Missing programming syntax - Need curly braces, parentheses, semicolons"
    
    return True, "✓ Valid smart contract detected"

def extract_features(contract_code):
    """Extract features from smart contract code"""
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

def extract_contract_name(code):
    """Extract contract name from code"""
    match = re.search(r'contract\s+(\w+)', code, re.IGNORECASE)
    if match:
        return match.group(1)
    return "Unknown"

@app.route('/')
def home():
    html = '''
<!DOCTYPE html>
<html>
<head>
    <title>RugGuard - AI Smart Contract Scam Detector</title>
    <meta charset="UTF-8">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            color: #667eea;
            font-size: 36px;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            font-size: 16px;
        }
        .info-box {
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .info-box h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .info-box ul {
            margin-left: 20px;
            color: #555;
        }
        .info-box code {
            background: #e0e7ff;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 14px;
        }
        textarea {
            width: 100%;
            height: 350px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-family: "Courier New", monospace;
            font-size: 13px;
            resize: vertical;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .button-container {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr;
            gap: 10px;
            margin-top: 15px;
        }
        button {
            padding: 15px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-analyze {
            background: #667eea;
            color: white;
        }
        .btn-analyze:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-example {
            background: #e0e7ff;
            color: #667eea;
        }
        .btn-example:hover {
            background: #c7d2fe;
        }
        .btn-clear {
            background: #fee;
            color: #e11d48;
        }
        .btn-clear:hover {
            background: #fdd;
        }
        #result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 10px;
            display: none;
            animation: slideIn 0.5s ease;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .scam {
            background: #fee;
            border: 2px solid #ff0000;
        }
        .safe {
            background: #efe;
            border: 2px solid #00cc00;
        }
        .warning {
            background: #fff4e6;
            border: 2px solid #ff9900;
        }
        .error {
            background: #ffe6e6;
            border: 2px solid #ff4444;
        }
        .risk-score {
            font-size: 64px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        .contract-name {
            text-align: center;
            font-size: 18px;
            color: #666;
            margin-bottom: 20px;
        }
        .indicators-list {
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
        .indicators-list ul {
            margin-left: 20px;
        }
        .indicators-list li {
            margin: 8px 0;
            line-height: 1.6;
        }
        .recommendation-box {
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #999;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ RugGuard AI</h1>
            <p class="subtitle">Smart Contract Scam Detector - Powered by Machine Learning</p>
        </div>
        
        <div class="info-box">
            <h3>📋 What is this?</h3>
            <p>RugGuard analyzes Solidity smart contracts to detect potential cryptocurrency scams (rug pulls) before you invest.</p>
            <br>
            <h3>✅ Required: Valid Smart Contract Code</h3>
            <ul>
                <li>Must be Solidity code (starts with <code>pragma solidity</code>)</li>
                <li>Must contain <code>contract</code> keyword</li>
                <li>Must have functions, mappings, or constructor</li>
            </ul>
        </div>
        
        <textarea id="contractCode" placeholder="Paste Solidity smart contract code here...

Example:
pragma solidity ^0.8.0;

contract MyToken {
    mapping(address => uint256) public balances;
    
    function transfer(address to, uint256 amount) public {
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
}"></textarea>
        
        <div class="button-container">
            <button class="btn-analyze" onclick="analyzeContract()">🔍 Analyze Contract</button>
            <button class="btn-example" onclick="loadExample()">📝 Example</button>
            <button class="btn-clear" onclick="clearCode()">🗑️ Clear</button>
        </div>
        
        <div id="result"></div>
        
        <div class="footer">
            <p>🎓 Built for International Conference Submission</p>
            <p>⚠️ Always do your own research before investing in cryptocurrency</p>
        </div>
    </div>
    
    <script>
        function analyzeContract() {
            const code = document.getElementById("contractCode").value;
            
            if (!code.trim()) {
                alert("⚠️ Please paste smart contract code first!");
                return;
            }
            
            const resultDiv = document.getElementById("result");
            resultDiv.style.display = "block";
            resultDiv.className = "";
            resultDiv.innerHTML = "<div style='text-align: center; padding: 40px;'><div style='font-size: 48px; margin-bottom: 20px;'>🔄</div><h2>Analyzing Smart Contract...</h2><p style='color: #666; margin-top: 10px;'>Validating code structure and checking for scam patterns</p></div>";
            
            fetch("/analyze", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({code: code})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    displayError(data.error);
                } else {
                    displayResult(data);
                }
            })
            .catch(error => {
                displayError("Network error: " + error);
            });
        }
        
        function displayError(message) {
            const resultDiv = document.getElementById("result");
            resultDiv.className = "error";
            resultDiv.innerHTML = "<h2 style='text-align: center; color: #ff4444;'>⚠️ Invalid Input</h2><div class='recommendation-box' style='text-align: center; color: #ff4444;'>" + message + "</div><div style='margin-top: 20px; padding: 15px; background: white; border-radius: 5px;'><h3>💡 Tips:</h3><ul><li>Make sure you are pasting Solidity smart contract code</li><li>Code must start with <code>pragma solidity</code></li><li>Click Example button to see a sample</li></ul></div>";
        }
        
        function displayResult(data) {
            const resultDiv = document.getElementById("result");
            const riskScore = data.scam_probability;
            
            let className = "safe";
            let emoji = "✅";
            let verdict = "LIKELY SAFE";
            let color = "#00cc00";
            
            if (riskScore >= 70) {
                className = "scam";
                emoji = "🚨";
                verdict = "HIGH RISK - LIKELY SCAM";
                color = "#ff0000";
            } else if (riskScore >= 40) {
                className = "warning";
                emoji = "⚠️";
                verdict = "MEDIUM RISK - PROCEED WITH CAUTION";
                color = "#ff9900";
            }
            
            let indicatorsList = "";
            for (let i = 0; i < data.indicators.length; i++) {
                indicatorsList += "<li>" + data.indicators[i] + "</li>";
            }
            
            resultDiv.className = className;
            resultDiv.innerHTML = "<h2 style='text-align: center;'>" + emoji + " " + verdict + "</h2><div class='contract-name'>Contract: <strong>" + data.contract_name + "</strong></div><div class='risk-score' style='color: " + color + ";'>" + riskScore.toFixed(1) + "%</div><p style='text-align: center; font-size: 16px; color: #666; margin-bottom: 20px;'>Scam Probability Score</p><div class='indicators-list'><h3>🔍 Analysis Details:</h3><ul>" + indicatorsList + "</ul></div><div class='recommendation-box'><h3 style='margin-bottom: 10px;'>💡 Recommendation:</h3><p>" + data.recommendation + "</p></div>";
        }
        
        function loadExample() {
            const exampleCode = "pragma solidity ^0.8.0;\\n\\ncontract ScamToken {\\n    mapping(address => uint256) public balances;\\n    address public owner;\\n    \\n    constructor() {\\n        owner = msg.sender;\\n    }\\n    \\n    function transfer(address to, uint256 amount) public {\\n        balances[msg.sender] -= amount;\\n        balances[to] += amount;\\n    }\\n    \\n    // SCAM: Owner can mint unlimited tokens\\n    function mint(uint256 amount) public {\\n        require(msg.sender == owner);\\n        balances[owner] += amount;\\n    }\\n    \\n    // SCAM: Owner can drain all funds\\n    function drain() private {\\n        payable(owner).transfer(address(this).balance);\\n    }\\n}";
            
            document.getElementById("contractCode").value = exampleCode.replace(/\\n/g, "\n");
        }
        
        function clearCode() {
            document.getElementById("contractCode").value = "";
            document.getElementById("result").style.display = "none";
        }
    </script>
</body>
</html>
'''
    return html

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        contract_code = data.get('code', '')
        
        # Validate
        is_valid, validation_message = is_valid_smart_contract(contract_code)
        
        if not is_valid:
            return jsonify({'error': validation_message})
        
        # Extract contract name
        contract_name = extract_contract_name(contract_code)
        
        # Extract features
        features = extract_features(contract_code)
        
        # Convert to DataFrame
        X = pd.DataFrame([features])[feature_columns]
        
        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        scam_prob = probability[1] * 100
        
        # Generate indicators
        indicators = []
        if features['has_mint'] and features['has_owner']:
            indicators.append("🚨 Owner-controlled mint function detected (CRITICAL RISK)")
        if features['has_drain']:
            indicators.append("🚨 Drain function found (CRITICAL RISK - can steal all funds)")
        if features['has_selfdestruct']:
            indicators.append("🚨 Self-destruct capability (CRITICAL RISK)")
        if features['has_blacklist']:
            indicators.append("⚠️ Blacklist function (owner can prevent users from selling)")
        if features['owner_count'] > 5:
            indicators.append(f"⚠️ Excessive owner control ({features['owner_count']} owner references)")
        if features['has_payable'] > 2:
            indicators.append(f"⚠️ Multiple payable functions ({features['has_payable']})")
        
        # Good signs
        if features['has_max_supply']:
            indicators.append("✅ Fixed maximum supply detected (GOOD SIGN)")
        if features['has_require'] > 3:
            indicators.append(f"✅ {features['has_require']} safety checks (require statements) found")
        if features['has_comments']:
            indicators.append("✅ Code documentation present (GOOD SIGN)")
        if features['has_modifier']:
            indicators.append("✅ Custom modifiers for access control (GOOD SIGN)")
        
        if len(indicators) == 0:
            indicators.append("ℹ️ No strong scam indicators detected in this contract")
        
        # Recommendation
        if scam_prob >= 70:
            recommendation = "🚫 DO NOT INVEST - This contract shows multiple scam patterns. High probability of rug pull."
        elif scam_prob >= 40:
            recommendation = "⚠️ PROCEED WITH EXTREME CAUTION - Contract has suspicious patterns. Verify project team, audit reports, and liquidity locks before investing."
        else:
            recommendation = "✅ Low risk detected based on code analysis. However, always verify: (1) Team identity, (2) Audit reports, (3) Liquidity locks, (4) Community feedback before investing."
        
        return jsonify({
            'prediction': 'SCAM' if prediction == 1 else 'LEGIT',
            'contract_name': contract_name,
            'scam_probability': scam_prob,
            'indicators': indicators,
            'recommendation': recommendation
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 RugGuard AI - Smart Contract Analyzer")
    print("="*60)
    print("\n✨ Features:")
    print("  • Smart contract validation")
    print("  • AI-powered scam detection")
    print("  • Real-time risk scoring")
    print("\n📍 Open your browser and go to:")
    print("\n   👉 http://localhost:5000\n")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)