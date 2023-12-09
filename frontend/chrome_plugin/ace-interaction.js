function waitForAceEditor() {
  const targetNode = document.getElementById('ace-editor-contract-code');

  // If the targetNode is already available, initialize the ACE editor
  if (targetNode) {
    const aceEditor = ace.edit(targetNode);
    const value = aceEditor.getValue();
    setResultValue(value);
    return;
  }

  // Otherwise, wait for the targetNode to become available
  const observer = new MutationObserver((mutationsList, observer) => {
    for (const mutation of mutationsList) {
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        // Check if the targetNode is now available
        if (mutation.addedNodes[0].id === 'ace-editor-contract-code') {
          observer.disconnect(); // Stop observing changes
          const aceEditor = ace.edit(mutation.addedNodes[0]);
          const value = aceEditor.getValue();
          setResultValue(value);
          return;
        }
      }
    }
  });

  // Start observing changes in the DOM
  observer.observe(document.body, { childList: true, subtree: true });
}


waitForAceEditor();
