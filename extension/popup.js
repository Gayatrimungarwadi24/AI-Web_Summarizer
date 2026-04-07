document.getElementById('summarize-btn').addEventListener('click', async () => {
    const format = document.getElementById('format-select').value;
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    resultDiv.innerText = "";
    loadingDiv.style.display = "block";
    progressContainer.style.display = "block";
    progressText.style.display = "block";
    progressBar.style.width = "0%";
    progressText.innerText = "0%";

    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    // INJECT BOTH FILES: Readability first, then your content script
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['Readability.js', 'content.js'] 
    }, async (injectionResults) => {
        
        // Error Handling: If injection fails or returns null
        if (!injectionResults || !injectionResults[0].result) {
            loadingDiv.style.display = "none";
            progressContainer.style.display = "none";
            progressText.style.display = "none";
            resultDiv.innerHTML = "<strong>Extraction Blocked:</strong> This website's layout or security prevents text extraction. Try another page.";
            return;
        }

        const pageText = injectionResults[0].result;

        // Error Handling: If Readability couldn't find an article
        if (pageText === "NO_ARTICLE_FOUND") {
            loadingDiv.style.display = "none";
            progressContainer.style.display = "none";
            progressText.style.display = "none";
            resultDiv.innerText = "Could not detect a main article on this page.";
            return;
        }

        // Send text to your local FastAPI/Gemma 3 backend
        try {
            const response = await fetch('http://127.0.0.1:8000/summarize_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: pageText, format: format })
            });

            if (!response.ok) {
                let errDetail = "Backend error.";
                try {
                    const errJson = await response.json();
                    if (errJson && errJson.detail) errDetail = errJson.detail;
                } catch (_) {}
                throw new Error(errDetail);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let totalSteps = 0;

            const updateProgress = (current, total) => {
                if (!total) return;
                const pct = Math.max(0, Math.min(100, Math.round((current / total) * 100)));
                progressBar.style.width = `${pct}%`;
                progressText.innerText = `${pct}%`;
            };

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop();

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed) continue;
                    let msg;
                    try {
                        msg = JSON.parse(trimmed);
                    } catch (_) {
                        continue;
                    }

                    if (msg.type === "meta" && msg.total) {
                        totalSteps = msg.total;
                        updateProgress(0, totalSteps);
                    } else if (msg.type === "progress") {
                        updateProgress(msg.current || 0, msg.total || totalSteps);
                    } else if (msg.type === "error") {
                        throw new Error(msg.message || "Backend error.");
                    } else if (msg.type === "result") {
                        updateProgress(msg.current || msg.total || totalSteps, msg.total || totalSteps);
                        resultDiv.innerText = msg.summary || "No summary returned.";
                        loadingDiv.style.display = "none";
                    }
                }
            }
            
        } catch (error) {
            loadingDiv.style.display = "none";
            progressContainer.style.display = "none";
            progressText.style.display = "none";
            resultDiv.innerText = `Error: ${error.message || "Ensure your Python backend and Ollama are running!"}`;
        }
    });
});
