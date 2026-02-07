# add_more_contracts.py - Expand our dataset

import json
import os

os.makedirs('data/scams', exist_ok=True)
os.makedirs('data/legit', exist_ok=True)

print("Adding more training contracts...")
print("="*60)

# SCAM Example 3: Blacklist Function
scam_3 = {
    "address": "0xSCAM003",
    "name": "BlacklistToken",
    "code": """
pragma solidity ^0.8.0;

contract BlacklistToken {
    mapping(address => uint256) public balances;
    mapping(address => bool) public blacklist;
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    function transfer(address to, uint256 amount) public {
        require(!blacklist[msg.sender], "Blacklisted");  // SCAM: Can block users
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
    
    function addToBlacklist(address user) public {
        require(msg.sender == owner);
        blacklist[user] = true;  // SCAM: Owner can prevent selling
    }
}
    """,
    "label": "SCAM"
}

# SCAM Example 4: Selfdestruct
scam_4 = {
    "address": "0xSCAM004",
    "name": "SelfDestructToken",
    "code": """
pragma solidity ^0.8.0;

contract SelfDestructToken {
    mapping(address => uint256) public balances;
    address payable owner;
    
    constructor() {
        owner = payable(msg.sender);
    }
    
    function transfer(address to, uint256 amount) public {
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
    
    function destroy() public {
        require(msg.sender == owner);
        selfdestruct(owner);  // SCAM: Can destroy contract and steal funds
    }
}
    """,
    "label": "SCAM"
}

# SCAM Example 5: High Tax
scam_5 = {
    "address": "0xSCAM005",
    "name": "HighTaxToken",
    "code": """
pragma solidity ^0.8.0;

contract HighTaxToken {
    mapping(address => uint256) public balances;
    address public owner;
    uint256 public sellTax = 50;  // SCAM: 50% tax on selling!
    
    constructor() {
        owner = msg.sender;
    }
    
    function transfer(address to, uint256 amount) public {
        uint256 taxAmount = (amount * sellTax) / 100;
        balances[owner] += taxAmount;  // SCAM: Owner takes 50%
        balances[to] += (amount - taxAmount);
    }
}
    """,
    "label": "SCAM"
}

# LEGIT Example 3: With Max Supply
legit_3 = {
    "address": "0xLEGIT003",
    "name": "CappedToken",
    "code": """
pragma solidity ^0.8.0;

contract CappedToken {
    mapping(address => uint256) public balances;
    uint256 public totalSupply;
    uint256 public constant MAX_SUPPLY = 1000000;
    
    constructor() {
        totalSupply = MAX_SUPPLY;
        balances[msg.sender] = MAX_SUPPLY;
    }
    
    function transfer(address to, uint256 amount) public returns (bool) {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        require(to != address(0), "Invalid address");
        
        balances[msg.sender] -= amount;
        balances[to] += amount;
        
        return true;
    }
    
    // No mint function - supply is fixed ✓
    // No owner variable ✓
}
    """,
    "label": "LEGIT"
}

# LEGIT Example 4: With Comments and Modifiers
legit_4 = {
    "address": "0xLEGIT004",
    "name": "ProfessionalToken",
    "code": """
pragma solidity ^0.8.0;

/**
 * @title Professional Token
 * @dev Standard ERC20 implementation with safety features
 */
contract ProfessionalToken {
    mapping(address => uint256) public balances;
    uint256 public totalSupply;
    
    // Events for transparency
    event Transfer(address indexed from, address indexed to, uint256 value);
    
    constructor(uint256 _initialSupply) {
        totalSupply = _initialSupply;
        balances[msg.sender] = _initialSupply;
    }
    
    /**
     * @dev Transfer tokens with safety checks
     */
    function transfer(address to, uint256 amount) public returns (bool) {
        require(to != address(0), "Cannot transfer to zero address");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        balances[msg.sender] -= amount;
        balances[to] += amount;
        
        emit Transfer(msg.sender, to, amount);
        return true;
    }
    
    // Well-documented ✓
    // Safety checks ✓
    // No owner controls ✓
}
    """,
    "label": "LEGIT"
}

# LEGIT Example 5: With Require Statements
legit_5 = {
    "address": "0xLEGIT005",
    "name": "SecureToken",
    "code": """
pragma solidity ^0.8.0;

contract SecureToken {
    mapping(address => uint256) public balances;
    uint256 public totalSupply;
    
    modifier validAddress(address _addr) {
        require(_addr != address(0), "Invalid address");
        _;
    }
    
    constructor(uint256 _supply) {
        require(_supply > 0, "Supply must be positive");
        totalSupply = _supply;
        balances[msg.sender] = _supply;
    }
    
    function transfer(address to, uint256 amount) 
        public 
        validAddress(to) 
        returns (bool) 
    {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        require(amount > 0, "Amount must be positive");
        
        balances[msg.sender] -= amount;
        balances[to] += amount;
        
        return true;
    }
    
    // Multiple require statements ✓
    // Modifiers for validation ✓
    // No dangerous functions ✓
}
    """,
    "label": "LEGIT"
}

# Save all new contracts
new_scams = [scam_3, scam_4, scam_5]
new_legit = [legit_3, legit_4, legit_5]

for i, contract in enumerate(new_scams, 3):
    filename = f'data/scams/scam_{i}.json'
    with open(filename, 'w') as f:
        json.dump(contract, f, indent=2)
    print(f"✓ Created {filename} - {contract['name']}")

for i, contract in enumerate(new_legit, 3):
    filename = f'data/legit/legit_{i}.json'
    with open(filename, 'w') as f:
        json.dump(contract, f, indent=2)
    print(f"✓ Created {filename} - {contract['name']}")

print("="*60)
print(f"✓ Added {len(new_scams)} scams + {len(new_legit)} legit contracts")
print(f"✓ Total dataset now: 5 scams + 5 legit = 10 contracts")
print("="*60)