# download_data.py - Get sample smart contracts

import requests
import json
import os
from web3 import Web3
from dotenv import load_dotenv

# Load API keys
load_dotenv()
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')

# Connect to Ethereum
w3 = Web3(Web3.HTTPProvider(f'https://mainnet.infura.io/v3/{os.getenv("INFURA_PROJECT_ID")}'))

print("Connected to Ethereum:", w3.is_connected())

# Known scam contracts (from rugdoc.io)
KNOWN_SCAMS = [
    '0x6982508145454ce325ddbe47a25d4ec3d2311933',  # PEPE (scam version)
    '0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce',  # SHIB (fake)
    # I'll give you 100 more addresses
]

# Known legitimate contracts
KNOWN_LEGIT = [
    '0xdac17f958d2ee523a2206206994597c13d831ec7',  # USDT (Tether)
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',  # USDC
    # I'll give you 100 more addresses
]

def download_contract_code(address):
    """Download smart contract source code from Etherscan"""
    
    url = f"https://api.etherscan.io/api"
    params = {
        'module': 'contract',
        'action': 'getsourcecode',
        'address': address,
        'apikey': ETHERSCAN_API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data['status'] == '1':
        source_code = data['result'][0]['SourceCode']
        contract_name = data['result'][0]['ContractName']
        
        return {
            'address': address,
            'name': contract_name,
            'code': source_code
        }
    else:
        print(f"Error downloading {address}: {data['message']}")
        return None

# Download first 10 scams
print("Downloading scam contracts...")
scam_data = []
for i, address in enumerate(KNOWN_SCAMS[:10]):  # Start with just 10
    print(f"  [{i+1}/10] {address}...")
    contract = download_contract_code(address)
    if contract:
        scam_data.append(contract)
        # Save to file
        with open(f'data/scams/{address}.json', 'w') as f:
            json.dump(contract, f, indent=2)

print(f"✓ Downloaded {len(scam_data)} scam contracts")

# Download first 10 legit
print("Downloading legitimate contracts...")
legit_data = []
for i, address in enumerate(KNOWN_LEGIT[:10]):
    print(f"  [{i+1}/10] {address}...")
    contract = download_contract_code(address)
    if contract:
        legit_data.append(contract)
        with open(f'data/legit/{address}.json', 'w') as f:
            json.dump(contract, f, indent=2)

print(f"✓ Downloaded {len(legit_data)} legitimate contracts")

print("\n" + "="*50)
print(f"TOTAL DATASET: {len(scam_data)} scams + {len(legit_data)} legit = {len(scam_data) + len(legit_data)} contracts")
print("="*50)