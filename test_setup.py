# test_setup.py - Check if everything is installed

print("Testing RugGuard Setup...")
print("=" * 50)

# Test 1: Python version
import sys
print(f"✓ Python version: {sys.version}")

# Test 2: NumPy
try:
    import numpy as np
    print("✓ NumPy installed")
except:
    print("✗ NumPy NOT installed - run: pip install numpy")

# Test 3: Pandas
try:
    import pandas as pd
    print("✓ Pandas installed")
except:
    print("✗ Pandas NOT installed")

# Test 4: PyTorch
try:
    import torch
    print(f"✓ PyTorch installed - version {torch.__version__}")
except:
    print("✗ PyTorch NOT installed")

# Test 5: Web3
try:
    from web3 import Web3
    print("✓ Web3 installed (blockchain connector)")
except:
    print("✗ Web3 NOT installed")

# Test 6: API Keys
try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    etherscan_key = os.getenv('ETHERSCAN_API_KEY')
    if etherscan_key:
        print(f"✓ Etherscan API key loaded: {etherscan_key[:10]}...")
    else:
        print("✗ Etherscan API key not found in .env file")
        
except Exception as e:
    print(f"✗ Error loading .env file: {e}")
    print("  Install with: pip install python-dotenv")

print("=" * 50)
print("Setup test complete!")