// popup.js - Extension logic

const API_URL = "http://127.0.0.1:5000/analyze";

document.getElementById('analyzeBtn').addEventListener('click', analyzeContract);
document.getElementById('clearBtn').addEventListener('click', clearInput);

function analyzeContract() {
  const code = document.getElementById('contractCode').value;
  const resultDiv = document.getElementById('result');
  
  if (!code.trim()) {
    alert('⚠️ Please paste smart contract code first!');
    return;
  }
  
  // Show loading
  resultDiv.style.display = 'block';
  resultDiv.className = '';
  resultDiv.innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <p>Analyzing smart contract...</p>
    </div>
  `;
  
  // Call API
  fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input: code })
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      showError(data.error);
    } else {
      showResult(data);
    }
  })
  .catch(error => {
    showError('Connection error. Make sure Flask server is running on localhost:5000');
  });
}

function showResult(data) {
  const resultDiv = document.getElementById('result');
  const risk = data.scam_probability;
  
  let className = 'result-safe';
  let emoji = '✅';
  let verdict = 'LIKELY SAFE';
  let color = '#00cc00';
  
  if (risk >= 70) {
    className = 'result-scam';
    emoji = '🚨';
    verdict = 'HIGH RISK SCAM';
    color = '#ff0000';
  } else if (risk >= 40) {
    className = 'result-warning';
    emoji = '⚠️';
    verdict = 'MEDIUM RISK';
    color = '#ff9900';
  }
  
  let indicatorsList = data.indicators.map(ind => `<li>${ind}</li>`).join('');
  
  resultDiv.className = className;
  resultDiv.innerHTML = `
    <div class="verdict">${emoji} ${verdict}</div>
    <div class="risk-score" style="color: ${color};">${risk.toFixed(1)}%</div>
    <p style="text-align: center; font-size: 12px; color: #666; margin-bottom: 15px;">Scam Probability</p>
    <div class="indicators">
      <strong>Analysis:</strong>
      <ul>${indicatorsList}</ul>
    </div>
    <div style="margin-top: 15px; padding: 10px; background: white; border-radius: 5px; font-size: 12px;">
      <strong>Recommendation:</strong><br>
      ${data.recommendation}
    </div>
  `;
}

function showError(message) {
  const resultDiv = document.getElementById('result');
  resultDiv.className = 'result-scam';
  resultDiv.innerHTML = `
    <div class="verdict">⚠️ ERROR</div>
    <p style="padding: 10px; font-size: 13px;">${message}</p>
  `;
}

function clearInput() {
  document.getElementById('contractCode').value = '';
  document.getElementById('result').style.display = 'none';
}