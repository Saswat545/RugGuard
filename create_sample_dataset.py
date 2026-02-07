# create_sample_dataset.py - Create simplified dataset for learning

import json
import os

# Create folders
os.makedirs('data/scams', exist_ok=True)
os.makedirs('data/legit', exist_ok=True)

print("Creating sample smart contract dataset...")
print("="*60)

# SCAM Contract Example 1: Hidden Mint Function
scam_contract_1 = {
    "address": "0xSCAM001",
    "name": "FakeMoonToken",
    "code": """
pragma solidity ^0.8.0;

contract FakeMoonToken {
    mapping(address => uint256) public balances;
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    // Normal transfer function (looks legit)
    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
    
    // SCAM: Owner can mint unlimited tokens (HIDDEN!)
    function mint(uint256 amount) public {
        require(msg.sender == owner, "Only owner");
        balances[owner] += amount;  // SCAM: Creates tokens from nothing
    }
    
    // SCAM: Owner can drain all funds
    function drain() private {
        payable(owner).transfer(address(this).balance);  // SCAM: Steals money
    }
}
    """,
    "label": "SCAM",
    "scam_indicators": [
        "mint function - owner can create unlimited tokens",
        "drain function - can steal all funds",
        "no liquidity lock",
        "owner has too much control"
    ]
}

# SCAM Contract Example 2: Honeypot (can buy but can't sell)
scam_contract_2 = {
    "address": "0xSCAM002",
    "name": "HoneyPotToken",
    "code": """
pragma solidity ^0.8.0;

contract HoneyPotToken {
    mapping(address => uint256) public balances;
    address private owner;
    bool public tradingEnabled = false;  // SCAM: Initially disabled
    
    function buy() public payable {
        balances[msg.sender] += msg.value * 1000;  // You CAN buy
    }
    
    function sell(uint256 amount) public {
        require(tradingEnabled, "Trading disabled");  // SCAM: Can't sell!
        // This will always fail because tradingEnabled = false
    }
    
    // Only owner can enable trading (and they never will)
    function enableTrading() public {
        require(msg.sender == owner);
        tradingEnabled = true;  // SCAM: Owner never calls this
    }
}
    """,
    "label": "SCAM",
    "scam_indicators": [
        "honeypot pattern - can buy but can't sell",
        "trading permanently disabled",
        "owner control over trading"
    ]
}

# LEGIT Contract Example 1: Standard ERC20
legit_contract_1 = {
    "address": "0xLEGIT001",
    "name": "StandardToken",
    "code": """
pragma solidity ^0.8.0;

contract StandardToken {
    mapping(address => uint256) public balances;
    uint256 public totalSupply;
    
    constructor(uint256 _initialSupply) {
        totalSupply = _initialSupply;
        balances[msg.sender] = _initialSupply;
    }
    
    // Standard transfer (safe)
    function transfer(address to, uint256 amount) public returns (bool) {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
        return true;
    }
    
    // Check balance (safe)
    function balanceOf(address account) public view returns (uint256) {
        return balances[account];
    }
    
    // No mint function - supply is fixed ✓
    // No owner privileges ✓
    // No drain function ✓
}
    """,
    "label": "LEGIT",
    "safety_features": [
        "no mint function - fixed supply",
        "no owner privileges",
        "standard ERC20 implementation",
        "no hidden functions"
    ]
}

# LEGIT Contract Example 2: Transparent with Security
legit_contract_2 = {
    "address": "0xLEGIT002",
    "name": "SafeToken",
    "code": """
pragma solidity ^0.8.0;

contract SafeToken {
    mapping(address => uint256) public balances;
    uint256 public constant MAX_SUPPLY = 1000000;
    uint256 public totalSupply;
    
    constructor() {
        totalSupply = MAX_SUPPLY;
        balances[msg.sender] = MAX_SUPPLY;
    }
    
    function transfer(address to, uint256 amount) public returns (bool) {
        require(to != address(0), "Invalid address");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        balances[msg.sender] -= amount;
        balances[to] += amount;
        
        return true;
    }
    
    // Public, transparent functions only ✓
    // No owner controls ✓
    // Fixed max supply ✓
}
    """,
    "label": "LEGIT",
    "safety_features": [
        "fixed maximum supply",
        "no owner address",
        "transparent functions",
        "input validation"
    ]
}

# Save scam contracts
scam_contracts = [scam_contract_1, scam_contract_2]
for i, contract in enumerate(scam_contracts, 1):
    filename = f'data/scams/scam_{i}.json'
    with open(filename, 'w') as f:
        json.dump(contract, f, indent=2)
    print(f"✓ Created {filename}")
    print(f"  Indicators: {', '.join(contract['scam_indicators'][:2])}")

print()

# Save legit contracts
legit_contracts = [legit_contract_1, legit_contract_2]
for i, contract in enumerate(legit_contracts, 1):
    filename = f'data/legit/legit_{i}.json'
    with open(filename, 'w') as f:
        json.dump(contract, f, indent=2)
    print(f"✓ Created {filename}")
    print(f"  Features: {', '.join(contract['safety_features'][:2])}")

print("="*60)
print(f"✓ Dataset created: {len(scam_contracts)} scams + {len(legit_contracts)} legit")
print("="*60)

# Create summary
summary = {
    "total_contracts": len(scam_contracts) + len(legit_contracts),
    "scam_count": len(scam_contracts),
    "legit_count": len(legit_contracts),
    "scam_files": [f"scam_{i}.json" for i in range(1, len(scam_contracts)+1)],
    "legit_files": [f"legit_{i}.json" for i in range(1, len(legit_contracts)+1)]
}

with open('data/dataset_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✓ Summary saved to data/dataset_summary.json")