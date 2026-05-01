from fastapi import FastAPI
from pydantic import BaseModel, Field

from rent_intelligence.parser.post_parser import parse_rental_post
from rent_intelligence.db.database import insert_post, fetch_posts, init_db
from fastapi import Body
app = FastAPI(title="Rental Post Parser API")
from typing import List

init_db()


class RawPostRequest(BaseModel):
    raw_text: str = Field(..., min_length=20)
    save: bool = False

class BulkRawPostRequest(BaseModel):
    raw_text: str = Field(..., min_length=20)
    separator: str = "==="
    save: bool = True


@app.get("/")
def home():
    return {
        "message": "Rental Post Parser API is running"
    }


@app.post("/rental-posts/parse")
def parse_post(request: RawPostRequest):
    parsed = parse_rental_post(request.raw_text)

    if request.save:
        post_id = insert_post(parsed)
        parsed["id"] = post_id

    return parsed


@app.get("/rental-posts")
def list_posts(limit: int = 20, status: str | None = None):
    return {
        "posts": fetch_posts(limit=limit, status=status)
    }


@app.get("/rental-posts/review")
def review_posts(limit: int = 20):
    return {
        "posts": fetch_posts(limit=limit, status="needs_review")
    }


@app.post("/rental-posts/parse-text")
def parse_post_text(
    raw_text: str = Body(..., media_type="text/plain"),
    save: bool = False
):
    # multiline text ko clean single-line bana rahe hain
    cleaned_text = " ".join(raw_text.split())

    parsed = parse_rental_post(cleaned_text)

    if save:
        post_id = insert_post(parsed)
        parsed["id"] = post_id

    return parsed

@app.post("/rental-posts/bulk-parse-text")
def bulk_parse_text(
    raw_text: str = Body(..., media_type="text/plain"),
    separator: str = "===",
    save: bool = True
):
    raw_posts = [
        post.strip()
        for post in raw_text.split(separator)
        if post.strip()
    ]

    results = []

    for index, post_text in enumerate(raw_posts, start=1):
        cleaned_text = " ".join(post_text.split())
        parsed = parse_rental_post(cleaned_text)

        if save:
            post_id = insert_post(parsed)
            parsed["id"] = post_id

        parsed["post_index"] = index
        results.append(parsed)

    return {
        "total_received": len(raw_posts),
        "saved": save,
        "results": results
    }


@app.post("/rental-posts/bulk-parse")
def bulk_parse_posts(request: BulkRawPostRequest):
    raw_posts = [
        post.strip()
        for post in request.raw_text.split(request.separator)
        if post.strip()
    ]

    results = []

    for index, post_text in enumerate(raw_posts, start=1):
        cleaned_text = " ".join(post_text.split())
        parsed = parse_rental_post(cleaned_text)

        if request.save:
            post_id = insert_post(parsed)
            parsed["id"] = post_id

        parsed["post_index"] = index
        results.append(parsed)

    return {
        "total_received": len(raw_posts),
        "saved": request.save,
        "results": results
    }