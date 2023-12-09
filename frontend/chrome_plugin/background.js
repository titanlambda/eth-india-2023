chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && /^http/.test(tab.url)) {
    let url = new URL(tab.url);
    console.log(url)

    if (url.hostname === 'polygonscan.com' && url.pathname.startsWith('/address/')) {
      let contractAddress = url.pathname.split('/')[2];
      fetchRiskScorePoly(contractAddress, tabId);
    } 
    else if (url.hostname === 'etherscan.io' && url.pathname.startsWith('/address/')) {
      let contractAddress = url.pathname.split('/')[2];
      fetchRiskScoreEther(contractAddress, tabId);
    }
    else if (url.hostname === 'zkevm.polygonscan.com' && url.pathname.startsWith('/address/')) {
      let contractAddress = url.pathname.split('/')[2];
      fetchRiskScorePolyZKEVM(contractAddress, tabId);
    } 
    else if (url.hostname === 'scrollscan.com' && url.pathname.startsWith('/address/')) {
      let contractAddress = url.pathname.split('/')[2];
      fetchRiskScoreScroll(contractAddress, tabId);
    }  
    else if (url.hostname === 'celoscan.io' && url.pathname.startsWith('/address/')) {
      let contractAddress = url.pathname.split('/')[2];
      fetchRiskScoreCelo(contractAddress, tabId);
    } 
    else if (url.hostname === 'basescan.org' && url.pathname.startsWith('/address/')) {
      let contractAddress = url.pathname.split('/')[2];
      fetchRiskScoreBase(contractAddress, tabId);
    } 
    else if (url.hostname === 'explorer.mantle.xyz' && url.pathname.startsWith('/address/')) {
      let contractAddress = url.pathname.split('/')[2];
      fetchRiskScoreMantle(contractAddress, tabId);
    } 
    
  }
});

// function fetchRiskScoreDummy(tabId) {
//   let riskScore = Math.floor(Math.random() * 100);
//   // Send a single message to the content script with the score and summary
//   chrome.tabs.sendMessage(tabId, {
//     action: 'updateRiskScore',
//     color: getRiskColor(riskScore),
//     score: riskScore,
//     summary: "summary"
//   });
// }

function fetchRiskScorePoly(contractAddress, tabId) {
  let apiUrl = `http://localhost:8080/api/get_risk_score_polygon_mainnet_verified_contract?smart_contract_address=${contractAddress}`;
  fetch(apiUrl)
    .then(response => response.json())
    .then(data => {
      let riskScore = data.risk_score;
      let color = getRiskColor(riskScore);
      let summary = data.result_summary;

      // Send a single message to the content script with the score and summary
      chrome.tabs.sendMessage(tabId, {
        action: 'updateRiskScore',
        color: color,
        score: riskScore,
        summary: summary
      });
    });
}

function fetchRiskScoreEther(contractAddress, tabId) {
  let apiUrl = `http://localhost:8080/api/get_risk_score_ethereum_mainnet_verified_contract?smart_contract_address=${contractAddress}`;
  fetch(apiUrl)
    .then(response => response.json())
    .then(data => {
      let riskScore = data.risk_score;
      let color = getRiskColor(riskScore);
      let summary = data.result_summary;

      // Send a single message to the content script with the score and summary
      chrome.tabs.sendMessage(tabId, {
        action: 'updateRiskScore',
        color: color,
        score: riskScore,
        summary: summary
      });
    });
}

function fetchRiskScorePolyZKEVM(contractAddress, tabId) {
  let apiUrl = `http://localhost:8080/api/get_risk_score_polygon_zkevm_mainnet_verified_contract?smart_contract_address=${contractAddress}`;
  fetch(apiUrl)
    .then(response => response.json())
    .then(data => {
      let riskScore = data.risk_score;
      let color = getRiskColor(riskScore);
      let summary = data.result_summary;

      // Send a single message to the content script with the score and summary
      chrome.tabs.sendMessage(tabId, {
        action: 'updateRiskScore',
        color: color,
        score: riskScore,
        summary: summary
      });
    });
}

function fetchRiskScoreScroll(contractAddress, tabId) {
  let apiUrl = `http://localhost:8080/api/get_risk_score_scroll_mainnet_verified_contract?smart_contract_address=${contractAddress}`;
  fetch(apiUrl)
    .then(response => response.json())
    .then(data => {
      let riskScore = data.risk_score;
      let color = getRiskColor(riskScore);
      let summary = data.result_summary;

      // Send a single message to the content script with the score and summary
      chrome.tabs.sendMessage(tabId, {
        action: 'updateRiskScore',
        color: color,
        score: riskScore,
        summary: summary
      });
    });
}

function fetchRiskScoreCelo(contractAddress, tabId) {
  let apiUrl = `http://localhost:8080/api/get_risk_score_celo_mainnet_verified_contract?smart_contract_address=${contractAddress}`;
  fetch(apiUrl)
    .then(response => response.json())
    .then(data => {
      let riskScore = data.risk_score;
      let color = getRiskColor(riskScore);
      let summary = data.result_summary;

      // Send a single message to the content script with the score and summary
      chrome.tabs.sendMessage(tabId, {
        action: 'updateRiskScore',
        color: color,
        score: riskScore,
        summary: summary
      });
    });
}

function fetchRiskScoreBase(contractAddress, tabId) {
  let apiUrl = `http://localhost:8080/api/get_risk_score_base_mainnet_verified_contract?smart_contract_address=${contractAddress}`;
  fetch(apiUrl)
    .then(response => response.json())
    .then(data => {
      let riskScore = data.risk_score;
      let color = getRiskColor(riskScore);
      let summary = data.result_summary;

      // Send a single message to the content script with the score and summary
      chrome.tabs.sendMessage(tabId, {
        action: 'updateRiskScore',
        color: color,
        score: riskScore,
        summary: summary
      });
    });
}

function fetchRiskScoreMantle(contractAddress, tabId) {
  let apiUrl = `http://localhost:8080/api/get_risk_score_mantle_mainnet_verified_contract?smart_contract_address=${contractAddress}`;
  fetch(apiUrl)
    .then(response => response.json())
    .then(data => {
      let riskScore = data.risk_score;
      let color = getRiskColor(riskScore);
      let summary = data.result_summary;

      // Send a single message to the content script with the score and summary
      chrome.tabs.sendMessage(tabId, {
        action: 'updateRiskScore',
        color: color,
        score: riskScore,
        summary: summary
      });
    });
}

function getRiskColor(riskScore) {
  let color;
  if (riskScore > 75) {
    color = 'red';
  } else if (riskScore > 50) {
    color = 'orange';
  } else if (riskScore > 25) {
    color = 'yellow';
  } else {
    color = 'green';
  }
  return color;
}
