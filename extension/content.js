(() => {
    // 1. Check if Readability loaded properly
    if (typeof Readability === 'undefined') {
        console.error("Readability.js failed to load.");
        return null;
    }

    // 2. Clone the document. 
    // We do this so Readability doesn't accidentally alter the visual webpage the user is looking at.
    var documentClone = document.cloneNode(true); 
    
    try {
        // 3. Run the algorithm
        var article = new Readability(documentClone).parse();
        
        // 4. Check if it successfully found text
        if (article && article.textContent) {
            // Return the clean, plain text (removing excessive white space)
            return article.textContent.replace(/\s+/g, ' ').trim();
        } else {
            return "NO_ARTICLE_FOUND"; 
        }
    } catch (err) {
        console.error("Readability parsing error: ", err);
        return null;
    }
})();