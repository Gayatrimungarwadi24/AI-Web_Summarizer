from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    text: str
    format: str

# Helper Function: Sliding Window Algorithm
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    
    # If the text is short enough, just return it as a single chunk
    if len(words) <= chunk_size:
        return [text]
        
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += (chunk_size - overlap) # Step forward, but leave an overlap
        
    return chunks

# Helper Function: Send prompt to Ollama
def ask_ollama(prompt: str) -> str:
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_content(request: SummarizeRequest):
    # Step 1: Chunk the incoming text
    text_chunks = chunk_text(request.text, chunk_size=500, overlap=50)
    
    # Step 2: Map (Summarize each chunk individually to save RAM)
    chunk_summaries = []
    for index, chunk in enumerate(text_chunks):
        # We use a generic prompt for the intermediate chunks
        intermediate_prompt = f"Briefly summarize the core points of this text extract:\n\n{chunk}"
        summary = ask_ollama(intermediate_prompt)
        chunk_summaries.append(summary)
        
    # Step 3: Reduce (Combine the mini-summaries)
    combined_summaries = "\n".join(chunk_summaries)
    
    # Step 4: Final formatting based on user selection
    if request.format == "bullet":
        final_instruction = "Summarize the following text into 3 to 5 highly concise bullet points."
    elif request.format == "short":
        final_instruction = "Provide a very short, 2-sentence summary of the following text."
    else:
        final_instruction = "Provide a well-structured, detailed summary of the following text."
        
    final_prompt = f"{final_instruction}\n\nText to summarize:\n{combined_summaries}"
    
    # Generate and return the final summary
    final_summary = ask_ollama(final_prompt)
    
    return {"summary": final_summary}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)