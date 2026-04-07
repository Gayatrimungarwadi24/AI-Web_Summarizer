from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import time
import math
import json
import os
from dotenv import load_dotenv

# Load env vars from .env if present
load_dotenv()

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

# Tuning knobs for speed vs. fidelity
BASE_CHUNK_SIZE = 800
OVERLAP = 0
MAX_INPUT_WORDS = 3000
MAX_CHUNKS = 4
FAST_SINGLE_PASS = True
SINGLE_PASS_WORD_LIMIT = 2500
OLLAMA_TIMEOUT = (5, 120)  # (connect, read) seconds
OLLAMA_MAX_RETRIES = 2

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_TIMEOUT = (5, 120)
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
OPENAI_RETRY_BACKOFF = float(os.getenv("OPENAI_RETRY_BACKOFF", "1.0"))

RETRYABLE_OPENAI_STATUS = {429, 500, 502, 503, 504}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_TIMEOUT = (5, 120)
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_RETRY_BACKOFF = float(os.getenv("GEMINI_RETRY_BACKOFF", "1.0"))

RETRYABLE_GEMINI_STATUS = {429, 500, 502, 503, 504}

# Helper Function: Sliding Window Algorithm
def chunk_text(text: str, chunk_size: int = BASE_CHUNK_SIZE, overlap: int = OVERLAP) -> list:
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
        "model": "gemma2:2b",
        "prompt": prompt,
        "stream": False
    }
    
    last_error = None
    for attempt in range(OLLAMA_MAX_RETRIES + 1):
        try:
            response = requests.post(ollama_url, json=payload, timeout=OLLAMA_TIMEOUT)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise HTTPException(status_code=500, detail=str(last_error))

def _extract_openai_error_detail(response: requests.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err.get("message"))
    except ValueError:
        pass
    text = (response.text or "").strip()
    if text:
        return text[:300]
    return f"OpenAI API error (HTTP {response.status_code})."

def _extract_gemini_error_detail(response: requests.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err.get("message"))
    except ValueError:
        pass
    text = (response.text or "").strip()
    if text:
        return text[:300]
    return f"Gemini API error (HTTP {response.status_code})."

def ask_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    last_error = None
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT)
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < GEMINI_MAX_RETRIES:
                time.sleep(GEMINI_RETRY_BACKOFF * (attempt + 1))
                continue
            raise HTTPException(status_code=500, detail=str(e))

        if response.status_code in RETRYABLE_GEMINI_STATUS:
            last_error = response
            if attempt < GEMINI_MAX_RETRIES:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(float(retry_after))
                    except ValueError:
                        time.sleep(GEMINI_RETRY_BACKOFF * (attempt + 1))
                else:
                    time.sleep(GEMINI_RETRY_BACKOFF * (attempt + 1))
                continue
            detail = _extract_gemini_error_detail(response)
            raise HTTPException(status_code=response.status_code, detail=detail)

        if response.status_code >= 400:
            detail = _extract_gemini_error_detail(response)
            raise HTTPException(status_code=response.status_code, detail=detail)

        data = response.json()
        last_error = None
        break

    if last_error is not None:
        raise HTTPException(status_code=500, detail=str(last_error))

    texts = []
    for cand in data.get("candidates", []):
        content = cand.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                texts.append(text)

    if texts:
        return "\n".join(texts).strip()

    return ""

def ask_openai(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set.")

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "input": prompt,
    }

    last_error = None
    for attempt in range(OPENAI_MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=OPENAI_TIMEOUT)
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < OPENAI_MAX_RETRIES:
                time.sleep(OPENAI_RETRY_BACKOFF * (attempt + 1))
                continue
            raise HTTPException(status_code=500, detail=str(e))

        if response.status_code in RETRYABLE_OPENAI_STATUS:
            last_error = response
            if attempt < OPENAI_MAX_RETRIES:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(float(retry_after))
                    except ValueError:
                        time.sleep(OPENAI_RETRY_BACKOFF * (attempt + 1))
                else:
                    time.sleep(OPENAI_RETRY_BACKOFF * (attempt + 1))
                continue
            detail = _extract_openai_error_detail(response)
            raise HTTPException(status_code=response.status_code, detail=detail)

        if response.status_code >= 400:
            detail = _extract_openai_error_detail(response)
            raise HTTPException(status_code=response.status_code, detail=detail)

        data = response.json()
        last_error = None
        break

    if last_error is not None:
        raise HTTPException(status_code=500, detail=str(last_error))

    texts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text" and part.get("text"):
                    texts.append(part["text"])

    if texts:
        return "\n".join(texts).strip()

    # Fallback if output items aren't present
    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()

    return ""

def ask_model(prompt: str) -> str:
    # Prefer Gemini if a key is configured; then OpenAI; otherwise fall back to Ollama
    if GEMINI_API_KEY:
        return ask_gemini(prompt)
    if OPENAI_API_KEY:
        return ask_openai(prompt)
    return ask_ollama(prompt)

def build_final_instruction(fmt: str) -> str:
    if fmt == "bullet":
        return "Summarize the following text into 3 to 5 highly concise bullet points."
    if fmt == "short":
        return "Provide a very short, 2-sentence summary of the following text."
    return "Provide a well-structured, detailed summary of the following text."

@app.post("/summarize")
async def summarize_content(request: SummarizeRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided for summarization.")

    # Trim overly long pages to keep latency reasonable
    words = request.text.split()
    truncated = False
    if len(words) > MAX_INPUT_WORDS:
        words = words[:MAX_INPUT_WORDS]
        truncated = True
    trimmed_text = " ".join(words)

    # Dynamically size chunks to cap the number of calls to Ollama
    if len(words) <= BASE_CHUNK_SIZE:
        chunk_size = BASE_CHUNK_SIZE
    else:
        chunk_size = max(BASE_CHUNK_SIZE, math.ceil(len(words) / MAX_CHUNKS))

    # Fast path: single-pass summary
    if FAST_SINGLE_PASS and len(words) <= SINGLE_PASS_WORD_LIMIT:
        final_instruction = build_final_instruction(request.format)
        final_prompt = f"{final_instruction}\n\nText to summarize:\n{trimmed_text}"
        final_summary = ask_model(final_prompt)
    else:
        # Step 1: Chunk the incoming text
        text_chunks = chunk_text(trimmed_text, chunk_size=chunk_size, overlap=OVERLAP)
        
        # Step 2: Map (Summarize each chunk individually)
        chunk_summaries = []
        for index, chunk in enumerate(text_chunks):
            # We use a generic prompt for the intermediate chunks
            intermediate_prompt = f"Briefly summarize the core points of this text extract:\n\n{chunk}"
            summary = ask_model(intermediate_prompt)
            chunk_summaries.append(summary)
            
        # Step 3: Reduce (Combine the mini-summaries)
        combined_summaries = "\n".join(chunk_summaries)
        
        # Step 4: Final formatting based on user selection
        final_instruction = build_final_instruction(request.format)
        final_prompt = f"{final_instruction}\n\nText to summarize:\n{combined_summaries}"
        
        # Generate and return the final summary
        final_summary = ask_model(final_prompt)
    
    if truncated:
        final_summary = "[Note: Page was long; summary is based on the first portion of the text.]\n\n" + final_summary

    return {"summary": final_summary}

@app.post("/summarize_stream")
def summarize_content_stream(request: SummarizeRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided for summarization.")

    # Trim overly long pages to keep latency reasonable
    words = request.text.split()
    truncated = False
    if len(words) > MAX_INPUT_WORDS:
        words = words[:MAX_INPUT_WORDS]
        truncated = True
    trimmed_text = " ".join(words)

    # Dynamically size chunks to cap the number of calls to Ollama
    if len(words) <= BASE_CHUNK_SIZE:
        chunk_size = BASE_CHUNK_SIZE
    else:
        chunk_size = max(BASE_CHUNK_SIZE, math.ceil(len(words) / MAX_CHUNKS))

    def stream():
        try:
            # Fast path: single-pass summary
            if FAST_SINGLE_PASS and len(words) <= SINGLE_PASS_WORD_LIMIT:
                total_steps = 2
                yield json.dumps({"type": "meta", "total": total_steps}) + "\n"
                yield json.dumps({
                    "type": "progress",
                    "current": 1,
                    "total": total_steps,
                    "stage": "single_start"
                }) + "\n"
                final_instruction = build_final_instruction(request.format)
                final_prompt = f"{final_instruction}\n\nText to summarize:\n{trimmed_text}"
                final_summary = ask_model(final_prompt)
            else:
                text_chunks = chunk_text(trimmed_text, chunk_size=chunk_size, overlap=OVERLAP)
                total_steps = (len(text_chunks) * 2) + 1  # start + done per chunk, plus final reduce
                yield json.dumps({"type": "meta", "total": total_steps}) + "\n"

                chunk_summaries = []
                for index, chunk in enumerate(text_chunks):
                    yield json.dumps({
                        "type": "progress",
                        "current": (index * 2) + 1,
                        "total": total_steps,
                        "stage": "chunk_start"
                    }) + "\n"

                    intermediate_prompt = f"Briefly summarize the core points of this text extract:\n\n{chunk}"
                    summary = ask_model(intermediate_prompt)
                    chunk_summaries.append(summary)
                    yield json.dumps({
                        "type": "progress",
                        "current": (index * 2) + 2,
                        "total": total_steps,
                        "stage": "chunk_done"
                    }) + "\n"

                combined_summaries = "\n".join(chunk_summaries)
                final_instruction = build_final_instruction(request.format)
                final_prompt = f"{final_instruction}\n\nText to summarize:\n{combined_summaries}"
                final_summary = ask_model(final_prompt)

            if truncated:
                final_summary = "[Note: Page was long; summary is based on the first portion of the text.]\n\n" + final_summary

            yield json.dumps({
                "type": "result",
                "summary": final_summary,
                "current": total_steps,
                "total": total_steps
            }) + "\n"
        except HTTPException as e:
            yield json.dumps({
                "type": "error",
                "message": str(e.detail),
                "status": e.status_code
            }) + "\n"
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "message": str(e)
            }) + "\n"

    return StreamingResponse(stream(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
