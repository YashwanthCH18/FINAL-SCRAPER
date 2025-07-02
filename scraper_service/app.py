"""Scraper micro-service (local-first).
FastAPI app that exposes:
  • POST /scraper/profile
  • POST /scraper/topic

This service uses an asynchronous, two-Lambda architecture. The API endpoints
instantly invoke a worker Lambda to perform long-running scraping jobs, thus
preventing API Gateway timeouts.
"""

import os
import uuid
import asyncio
import json
import boto3
from typing import List, Optional, Dict

import httpx
from fastapi import FastAPI, Depends, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Third-party SDKs
from supabase import create_client, Client as SupabaseClient
from pinecone import Pinecone
from trafilatura import extract as extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=os.getenv("SCRAPER_DOTENV", ".env"))

# ---------------------------------------------------------------------------
# ENV & third-party clients
# ---------------------------------------------------------------------------
SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_KEY"]
RAPID_KEY: str = os.environ["RAPIDAPI_KEY"]
RAPID_HOST: str = os.environ["RAPIDAPI_HOST"]
FIRE_KEY: str = os.environ["FIRECRAWL_API_KEY"]
PINECONE_API_KEY: str = os.environ["PINECONE_API_KEY"]
PINECONE_ENV: str = os.environ["PINECONE_ENVIRONMENT"]
PINECONE_INDEX: str = os.environ["PINECONE_INDEX"]
PINECONE_MODEL: Optional[str] = os.getenv("PINECONE_MODEL")
BLOG_CALLBACK_URL: Optional[str] = os.getenv("BLOG_CALLBACK_URL")
BLOG_CALLBACK_TOKEN: Optional[str] = os.getenv("BLOG_CALLBACK_TOKEN")

# AWS SDK client for invoking the worker
lambda_client = boto3.client("lambda")
WORKER_LAMBDA_NAME = os.getenv("WORKER_LAMBDA_NAME")

supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pc_index = pc.Index(PINECONE_INDEX)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# ---------------------------------------------------------------------------
# Helper / middleware
# ---------------------------------------------------------------------------
async def get_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme")
    token = authorization.split(" ")[1]
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_response.user.id
    except Exception as e:
        print(f"Token validation error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TopicRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=200)
    max_pages: int = Field(5, ge=1, le=50)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="Scraper Service", version="0.1.0")

origins = [
    "http://localhost:3000",
    "http://192.168.83.157:3000",
    "http://192.168.58.157:3000",
    "http://192.168.186.157:3000",
    "http://192.168.0.203:3000",
    "https://your-app-name.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Crawling & embedding core (UNCHANGED)
# ---------------------------------------------------------------------------
async def rapid_search(topic: str, max_pages: int) -> List[str]:
    url = f"https://{RAPID_HOST}/websearch"
    payload = {"text": topic, "safesearch": "off", "timelimit": "", "region": "wt-wt", "max_results": max_pages}
    headers = {"X-RapidAPI-Key": RAPID_KEY, "X-RapidAPI-Host": RAPID_HOST, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            results = data.get('result', []) if isinstance(data, dict) else data
            urls = [res.get('href') for res in results if isinstance(res, dict) and res.get('href')]
            if not urls:
                print(f"rapid_search warning: No URLs found in response.")
            return urls
    except httpx.HTTPStatusError as exc:
        print(f"rapid_search failed with status {exc.response.status_code}: {exc.response.text}")
        return []
    except Exception as exc:
        print(f"rapid_search failed with an unexpected error: {exc}")
        return []

async def firecrawl_fetch(url: str) -> str:
    headers = {"Authorization": f"Bearer {FIRE_KEY}"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json={"url": url, "formats": ["html"]})
        r.raise_for_status()
        return r.json()["data"]["html"]

async def upsert_text_chunks(namespace: str, chunks: List[Dict]):
    pc_index.upsert_records(records=chunks, namespace=namespace)

async def crawl_and_embed(urls: List[str], user_id: str, source_type: str, topic: Optional[str] = None):
    for url in urls:
        try:
            html = await firecrawl_fetch(url)
            text = extract_text(html) or ""
            if not text.strip(): continue
            chunks = text_splitter.split_text(text)
            records = [{
                "_id": str(uuid.uuid4()), "text": chunk, "url": url,
                "source_type": source_type, "topic": topic
            } for chunk in chunks]
            await upsert_text_chunks(user_id, records)
            print(f"Successfully crawled and embedded: {url}")
        except Exception as exc:
            print(f"Error crawling {url}: {exc}")

async def crawl_topic_background(topic: str, max_pages: int, user_id: str, source_type: str):
    try:
        urls = await rapid_search(topic, max_pages)
        print(f"[Worker] Found {len(urls)} URLs for topic '{topic}'. Starting crawl.")
        await crawl_and_embed(urls, user_id, source_type, topic=topic)
        if BLOG_CALLBACK_URL and BLOG_CALLBACK_TOKEN:
            try:
                await httpx.post(
                    f"{BLOG_CALLBACK_URL}/blog/generate-from-scrape",
                    headers={"X-Internal-Token": BLOG_CALLBACK_TOKEN, "X-User-Id": user_id},
                    json={"topic": topic}, timeout=20)
            except Exception as cb_exc:
                print(f"[Worker] Blog callback failed: {cb_exc}")
    except Exception as exc:
        print(f"[Worker] crawl_topic_background failed: {exc}")

# ---------------------------------------------------------------------------
# API Routes (NEW: Asynchronous Invocation)
# ---------------------------------------------------------------------------
@app.post("/scraper/profile")
async def scrape_profile(user_id: str = Depends(get_user_id)):
    response = supabase.table("onboarding").select("question3").eq("user_id", user_id).execute()
    if not response.data or not response.data[0].get("question3"):
        raise HTTPException(status_code=404, detail="Onboarding topic not found.")
    
    topic = response.data[0].get("question3")
    payload = {"topic": topic, "max_pages": 5, "user_id": user_id, "source_type": "profile"}

    if WORKER_LAMBDA_NAME:
        lambda_client.invoke(FunctionName=WORKER_LAMBDA_NAME, InvocationType="Event", Payload=json.dumps(payload))
        print(f"Invoked worker for profile scrape. User: {user_id}")
    else:
        print("Local mode: WORKER_LAMBDA_NAME not set. Skipping async invocation.")

    return {"status": "scraping_invoked", "source": "profile"}

@app.post("/scraper/topic")
async def scrape_topic(req: TopicRequest, user_id: str = Depends(get_user_id)):
    payload = {"topic": req.topic, "max_pages": req.max_pages, "user_id": user_id, "source_type": "topic"}

    if WORKER_LAMBDA_NAME:
        lambda_client.invoke(FunctionName=WORKER_LAMBDA_NAME, InvocationType="Event", Payload=json.dumps(payload))
        print(f"Invoked worker for topic scrape. User: {user_id}, Topic: {req.topic}")
    else:
        print("Local mode: WORKER_LAMBDA_NAME not set. Skipping async invocation.")

    return {"status": "scraping_invoked", "source": "topic"}

@app.get("/healthz")
async def health():
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Worker Logic (NEW)
# ---------------------------------------------------------------------------
def worker_handler(event, context):
    print(f"Worker received event: {event}")
    topic = event.get("topic")
    max_pages = event.get("max_pages")
    user_id = event.get("user_id")
    source_type = event.get("source_type")

    if not all([topic, max_pages, user_id, source_type]):
        print("Worker error: Missing required parameters.")
        return {"status": "error", "message": "Missing parameters"}

    try:
        asyncio.run(crawl_topic_background(topic, max_pages, user_id, source_type))
        print("Worker finished successfully.")
        return {"status": "success"}
    except Exception as e:
        print(f"Worker failed with exception: {e}")
        # Optionally, you could add more robust error handling here, e.g., to a dead-letter queue.
        return {"status": "error", "message": str(e)}

# ---------------------------------------------------------------------------
# API Handler (for API Gateway)
# ---------------------------------------------------------------------------
from mangum import Mangum
handler = Mangum(app)
