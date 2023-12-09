chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && /^http/.test(tab.url)) {
    let url = new URL(tab.url);
    console.log(url)
    if (url.hostname === 'zkevm.polygonscan.com' && url.pathname.startsWith('/address/')) {
      fetchRiskScoreDummy(tabId);
    } 
    else if (url.hostname === 'scrollscan.com' && url.pathname.startsWith('/address/')) {
      fetchRiskScoreDummy(tabId);
    }  
    else if (url.hostname === 'celoscan.io' && url.pathname.startsWith('/address/')) {
      fetchRiskScoreDummy(tabId);
    } 
    else if (url.hostname === 'explorer.mantle.xyz' && url.pathname.startsWith('/address/')) {
      fetchRiskScoreDummy(tabId);
    } 
    
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
