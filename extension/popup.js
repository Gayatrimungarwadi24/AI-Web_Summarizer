document.getElementById('summarize-btn').addEventListener('click', async () => {
    const format = document.getElementById('format-select').value;
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    
    resultDiv.innerText = "";
    loadingDiv.style.display = "block";

    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    // INJECT BOTH FILES: Readability first, then your content script
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['Readability.js', 'content.js'] 
    }, async (injectionResults) => {
        
        // Error Handling: If injection fails or returns null
        if (!injectionResults || !injectionResults[0].result) {
            loadingDiv.style.display = "none";
            resultDiv.innerHTML = "<strong>Extraction Blocked:</strong> This website's layout or security prevents text extraction. Try another page.";
            return;
        }

        const pageText = injectionResults[0].result;

        // Error Handling: If Readability couldn't find an article
        if (pageText === "NO_ARTICLE_FOUND") {
            loadingDiv.style.display = "none";
            resultDiv.innerText = "Could not detect a main article on this page.";
            return;
        }

        // Send text to your local FastAPI/Gemma 3 backend
        try {
            const response = await fetch('http://127.0.0.1:8000/summarize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: pageText, format: format })
            });
            
            const data = await response.json();
            loadingDiv.style.display = "none";
            resultDiv.innerText = data.summary;
            
        } catch (error) {
            loadingDiv.style.display = "none";
            resultDiv.innerText = "Error: Ensure your Python backend and Ollama are running!";
        }
    });
});