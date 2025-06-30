"""Scraper micro-service (local-first).
FastAPI app that exposes:
  • POST /scraper/profile
  • POST /scraper/topic
  • GET  /status/{job_id}

Designed to run locally with `uvicorn scraper_service.app:app --reload --port 8002`
and later package into AWS Lambda via AWS SAM.
"""

import os
import uuid
import asyncio
from typing import List, Optional, Dict
from datetime import datetime

import httpx
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request, Header
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(override=True)  # Force loading from .env, overriding shell memory

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
SUPABASE_KEY: str = os.environ["SUPABASE_KEY"]  # service-role for server code

RAPID_KEY: str = os.environ["RAPIDAPI_KEY"]
RAPID_HOST: str = os.environ["RAPIDAPI_HOST"]
FIRE_KEY: str = os.environ["FIRECRAWL_API_KEY"]

PINECONE_API_KEY: str = os.environ["PINECONE_API_KEY"]
PINECONE_ENV: str = os.environ["PINECONE_ENVIRONMENT"]
PINECONE_INDEX: str = os.environ["PINECONE_INDEX"]
# Optional: override model when we need client-side embeddings
PINECONE_MODEL: Optional[str] = os.getenv("PINECONE_MODEL")

BLOG_CALLBACK_URL: Optional[str] = os.getenv("BLOG_CALLBACK_URL")
BLOG_CALLBACK_TOKEN: Optional[str] = os.getenv("BLOG_CALLBACK_TOKEN")

supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
# Ensure the index exists externally; here we just connect
pc_index = pc.Index(PINECONE_INDEX)
# Detect if the index is integrated-embedding (dimension == 0)
try:
    _index_dimension = pc.describe_index(PINECONE_INDEX).dimension  # 0 -> integrated
except Exception:
    _index_dimension = 0  # default to integrated if describe fails

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# ---------------------------------------------------------------------------
# Helper / middleware
# ---------------------------------------------------------------------------
class UserCtx(BaseModel):
    user_id: str


async def get_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme")
    token = authorization.split(" ")[1]
    try:
        # Use get_user to validate the JWT and get user details
        user_response = supabase.auth.get_user(token)
        if not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = user_response.user.id
        return user_id
    except Exception as e:
        print(f"Token validation error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TopicRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=200)
    max_pages: int = Field(10, ge=1, le=50)  # Default to 10, but allow override

class JobStatus(BaseModel):
    state: str
    progress: int = 0
    result_resource_id: Optional[str] = None
    error: Optional[str] = None

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="Scraper Service", version="0.1.0")

# ---------------------------------------------------------------------------
# DB helpers (Supabase) – with in-memory fallback for local dev
# ---------------------------------------------------------------------------

_jobs_cache: Dict[str, Dict] = {}


def insert_job(job_id: str, job_type: str, user_id: str):
    # user_id is now passed in directly
    payload = {
        "id": job_id,
        "user_id": user_id,
        "service": "scraper",
        "job_type": job_type,
        "status": "queued",
        "progress": 0,
    }
    try:
        if supabase:
            print(f"[DB_DEBUG] Attempting to insert job: {payload}")
            response = supabase.table("jobs").insert(payload).execute()
            print(f"[DB_DEBUG] Supabase insert response: {response.data}")
        else:
            print("insert_job fallback: no supabase client")
            _jobs_cache[job_id] = payload
    except Exception as exc:
        print("insert_job fallback", exc)
        _jobs_cache[job_id] = payload


def update_job(job_id: str, user_id: str, **kwargs):
    payload = kwargs.copy()
    payload["updated_at"] = datetime.now().isoformat()

    try:
        if supabase:
            print(f"[DB_DEBUG] Attempting to update job {job_id} with: {payload}")
            response = supabase.table("jobs").update(payload).eq("id", job_id).execute()
            print(f"[DB_DEBUG] Supabase update response: {response.data}")
        else:
            print("update_job fallback: no supabase client")
            if job_id in _jobs_cache:
                _jobs_cache[job_id].update(payload)
    except Exception as exc:
        print("update_job fallback", exc)
        if job_id in _jobs_cache:
            _jobs_cache[job_id].update(payload)


def fetch_job(job_id: str, user_id: str) -> Dict:
    try:
        row = (
            supabase.table("jobs")
            .select("*")
            .eq("id", job_id)
            .eq("user_id", user_id)
            .single()
            .execute()
            .data
        )
        if row:
            return row
    except Exception as exc:
        print("fetch_job fallback", exc)
    # Fallback to in-memory cache
    row = _jobs_cache.get(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return row

# ---------------------------------------------------------------------------
# Crawling & embedding core
# ---------------------------------------------------------------------------
async def rapid_search(topic: str, max_pages: int) -> List[str]:
    """Search Google using a configured RapidAPI provider."""
    url = f"https://{RAPID_HOST}/websearch"
    payload = {
        "text": topic,
        "safesearch": "off",
        "timelimit": "",
        "region": "wt-wt",
        "max_results": max_pages
    }
    headers = {
        "X-RapidAPI-Key": RAPID_KEY,
        "X-RapidAPI-Host": RAPID_HOST,
        "Content-Type": "application/json"
    }

    print(f"Calling RapidAPI at {url} with host {RAPID_HOST} and payload {payload}")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

            print(f"RapidAPI response: {data}")

            results = []
            if isinstance(data, list):
                results = data
            elif isinstance(data, dict) and 'result' in data and isinstance(data['result'], list):
                results = data['result']
            
            if not results:
                print(f"rapid_search warning: No results found in response.")
                return []

            urls = [result.get('href') for result in results if isinstance(result, dict) and result.get('href')]
            
            if not urls:
                print(f"rapid_search error: Could not find URLs in the response (expected 'href' key).")
                return []

            return urls

    except httpx.HTTPStatusError as exc:
        print(f"rapid_search failed with status {exc.response.status_code}: {exc}")
        print(f"Response body: {exc.response.text}")
        return []
    except Exception as exc:
        print(f"rapid_search failed with an unexpected error: {exc}")
        return []

async def firecrawl_fetch(url: str) -> str:
    headers = {"Authorization": f"Bearer {FIRE_KEY}"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers=headers,
            json={"url": url, "formats": ["html"]},
        )
        r.raise_for_status()
        return r.json()["data"]["html"]


async def upsert_text_chunks(namespace: str, chunks: List[Dict]):
    """
    Upserts records into the Pinecone index, allowing Pinecone to handle
    the text-to-vector embedding automatically.
    """
    pc_index.upsert_records(records=chunks, namespace=namespace)

async def crawl_and_embed(urls: List[str], job_id: str, source_type: str, user_id: str, topic: Optional[str] = None):
    total = len(urls)
    completed = 0
    for url in urls:
        try:
            html = await firecrawl_fetch(url)
            text = extract_text(html) or ""
            if not text.strip():
                continue
            chunks = text_splitter.split_text(text)
            records = [{
                "_id": str(uuid.uuid4()),
                "text": chunk,
                "url": url,
                "source_type": source_type,
                "job_id": job_id,
                "topic": topic
            } for chunk in chunks]
            # Integrated embedding: Pinecone converts chunk_text to vectors automatically.
            await upsert_text_chunks(user_id, records)
        except Exception as exc:
            # log and skip
            print(f"Error crawling {url}: {exc}")
        completed += 1
        progress = int(completed / total * 100)
        update_job(job_id, user_id, progress=progress, status="in_progress")

    update_job(job_id, user_id, progress=100, status="completed")


async def crawl_topic_background(topic: str, max_pages: int, job_id: str, user_id: str):
    update_job(job_id, user_id, status="in_progress")
    try:
        urls = await rapid_search(topic, max_pages)
        print(f"[debug] rapid_search returned {len(urls)} urls for topic '{topic}'")
        await crawl_and_embed(urls, job_id, "topic", user_id, topic=topic)
        # Notify blog service if configured, but don't fail the whole job if callback is unreachable.
        if BLOG_CALLBACK_URL and BLOG_CALLBACK_TOKEN:
            try:
                await httpx.post(
                    f"{BLOG_CALLBACK_URL}/blog/generate-from-scrape",
                    headers={
                        "X-Internal-Token": BLOG_CALLBACK_TOKEN,
                        "X-User-Id": user_id,
                    },
                    json={"topic": topic, "job_id": job_id},
                    timeout=20,
                )
            except Exception as cb_exc:
                print("blog callback failed", cb_exc)
                # We intentionally swallow the error so the scrape is still marked completed.

    except Exception as exc:
        print("crawl_topic_background failed", exc)
        update_job(job_id, user_id, status="failed", error=str(exc))

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/scraper/profile", status_code=202)
async def scrape_profile(background_tasks: BackgroundTasks, user_id: str = Depends(get_user_id)):
    job_id = f"scrp_{uuid.uuid4().hex[:8]}"
    insert_job(job_id, job_type="profile", user_id=user_id)

    # 1. fetch onboarding data to get the website URL and the topic for metadata
    onboarding = (
        supabase.table("onboarding")
        .select("question1", "question3") # Fetch both website and topic source
        .eq("user_id", user_id)
        .single()
        .execute()
        .data
    )

    if not onboarding:
        # Handle case where onboarding data doesn't exist for the user
        update_job(job_id, user_id, status="failed", error="Onboarding data not found.")
        return {"job_id": job_id, "status": "failed", "detail": "Onboarding data not found."}

    domain = onboarding.get("question1")
    # Use the answer to question3 as the consistent topic for profile data
    topic = onboarding.get("question3")

    urls = [f"https://{domain}"] if domain else []

    background_tasks.add_task(crawl_and_embed, urls, job_id, "profile", user_id, topic=topic)
    return {"job_id": job_id}


@app.post("/scraper/topic", status_code=202)
async def scrape_topic(req: TopicRequest, background_tasks: BackgroundTasks, user_id: str = Depends(get_user_id)):
    job_id = f"scrp_{uuid.uuid4().hex[:8]}"
    insert_job(job_id, job_type="topic", user_id=user_id)
    background_tasks.add_task(crawl_topic_background, req.topic, req.max_pages, job_id, user_id)
    return {"job_id": job_id}


@app.get("/status/{job_id}", response_model=JobStatus)
async def job_status(job_id: str, user_id: str = Depends(get_user_id)):
    row = fetch_job(job_id, user_id)
    if row["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {
        "state": row["status"],
        "progress": row.get("progress", 0),
        "result_resource_id": row.get("result_resource"),
        "error": row.get("error"),
    }


@app.get("/healthz")
async def health():
    return {"status": "ok"}
