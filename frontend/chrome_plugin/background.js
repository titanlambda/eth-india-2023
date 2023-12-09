chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && /^http/.test(tab.url)) {
      fetchRiskScoreDummy(tabId);
  }
});

function fetchRiskScoreDummy(tabId) {
  let riskScore = Math.floor(Math.random() * 100);
  // Send a single message to the content script with the score and summary
  chrome.tabs.sendMessage(tabId, {
    action: 'updateRiskScore',
    color: getRiskColor(riskScore),
    score: riskScore,
    summary: "summary"
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
