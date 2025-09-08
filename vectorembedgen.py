from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
import os
import json
import pymysql
import math
import re
import logging
import numpy as np

logger = logging.getLogger("uvicorn.error")

# --- Load environment variables ---
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")  # e.g. "mysql+pymysql://root:password@localhost:4000/test"

# --- SQLAlchemy engine + session ---
engine = create_engine(
    DB_URL,
    connect_args={"ssl": {"ca": "/etc/ssl/certs/ca-certificates.crt"}},  # tells PyMySQL to use TLS with system root CAs
    pool_recycle=3600,
    pool_pre_ping=True 
)

SessionLocal = sessionmaker(bind=engine)

# --- FastAPI app ---
app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/nli-deberta-v3-large")

BATCH_SIZE = 5

from urllib.parse import urlparse

def normalize_table_name_external(raw_searched: str) -> str:
    """
    Convert a raw searched string into a safe table name for external tables.
    Strips common filters and invalid characters, replaces spaces with underscores,
    and appends '_external' suffix.
    """
    if not raw_searched:
        return "external_table"

    # Step 1: Remove filters starting with -- or - (like --filetype=xhtml or -site=)
    base = re.split(r"\s--|\s-", raw_searched)[0].strip()

    # Step 2: Replace all non-alphanumeric characters with underscores
    base = re.sub(r"[^0-9a-zA-Z]+", "_", base)

    # Step 3: Lowercase and append _external
    table_name = f"{base.lower()}_external"

    return table_name

def normalize_table_name(url: str, suffix: str) -> str:
    """
    Convert a URL like https://sub.example.co.uk to 'example_internal'.
    Always uses the second-level domain.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or "unknown"  # e.g., 'sub.example.co.uk'
    parts = hostname.split(".")
    
    if len(parts) >= 2:
        base = parts[-2]  # 'example' from 'sub.example.co.uk'
    else:
        base = parts[0]

    return f"{base}_{suffix}".lower()

def normalize_search_keyword(keyword: str, suffix: str) -> str:
    """Convert keyword to safe table name with suffix"""
    import re
    base = re.sub(r"[^a-zA-Z0-9]", "_", keyword.lower())
    return f"{base}_{suffix}"

from fastapi import Body
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from sqlalchemy import text as sql_text

@app.post("/embed")
async def embed(chunks: list[dict] = Body(...)):
    if not chunks:
         return JSONResponse({"status": "error", "message": "No chunks provided"})

    # 1️⃣ Extract searched from first chunk's metadata
    first_meta = chunks[0].get("metadata", {})
    searched_raw = first_meta.get("searched", "").strip()
    if not searched_raw:
        return JSONResponse({"status": "error", "message": "No searched keyword in metadata"})

    # 2️⃣ Get base keyword (strip filters like --filetype, -site)
    base_keyword = searched_raw.split("--")[0].split("-")[0].strip()
    if not base_keyword:
        return JSONResponse({"status": "error", "message": "No valid base keyword found"})
    
    # 3️⃣ Normalize table name
    external_table = normalize_table_name_external(base_keyword)

    # 4️⃣ Extract texts and metadata
    texts = [c["text"] for c in chunks if "text" in c]
    metadata_list = [c.get("metadata", {}) for c in chunks]

    # 5️⃣ Encode embeddings
    vecs = model.encode(texts, show_progress_bar=False)
    inserted_count = 0

    try:
        with SessionLocal() as session:
            # --- Check if table exists ---
            table_exists = session.execute(sql_text(f"""
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
                  AND table_name = :table_name
            """), {"table_name": external_table}).fetchone()

            recreate = False
            if table_exists:
                # Check table creation time
                create_time = session.execute(sql_text(f"""
                    SELECT CREATE_TIME
                    FROM information_schema.tables
                    WHERE table_schema = DATABASE()
                      AND table_name = :table_name
                """), {"table_name": external_table}).fetchone()[0]

                if create_time < datetime.utcnow() - timedelta(days=3):
                    # Too old → drop & recreate
                    session.execute(sql_text(f"DROP TABLE IF EXISTS {external_table}"))
                    recreate = True
            else:
                # Table does not exist → create
                recreate = True

            # --- Create table if needed ---
            if recreate:
                session.execute(sql_text(f"""
                    CREATE TABLE {external_table} (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        chunk_text TEXT NOT NULL,
                        embedding VECTOR(384),
                        url TEXT,
                        retrieved_at TIMESTAMP,
                        sourcekb VARCHAR(100)
                    )
                """))
            session.commit()

            # --- Insert chunks ---
            for text_val, emb, meta in zip(texts, vecs, metadata_list):
                session.execute(sql_text(f"""
                    INSERT INTO {external_table} 
                    (chunk_text, embedding, url, retrieved_at, sourcekb)
                    VALUES (:chunk_text, CAST(:emb_str AS VECTOR(384)), :url, :retrieved_at, :sourcekb)
                    ON DUPLICATE KEY UPDATE
                        chunk_text = VALUES(chunk_text),
                        embedding = VALUES(embedding),
                        url = VALUES(url),
                        retrieved_at = VALUES(retrieved_at),
                        sourcekb = VALUES(sourcekb)
                """), {
                    "chunk_text": text_val,
                    "emb_str": json.dumps(emb.astype("float32").tolist()),
                    "url": meta.get("url"),
                    "retrieved_at": meta.get("date"),
                    "sourcekb": meta.get("sourcekb"),
                })
                inserted_count += 1

            session.commit()

            # --- Update keywords table ---
            session.execute(sql_text("""
                    UPDATE keywords
                    SET status = 'completed',
                    last_seen = CURRENT_TIMESTAMP
                    WHERE keyword = :kw
                 """), {"kw": searched_raw})
            session.commit()

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

    return JSONResponse({
        "status": "ok",
        "inserted_count": inserted_count,
        "external_table": external_table,
        "keyword_status": "completed"
    })

@app.post("/insertsearchtodb")
async def insert_search_to_db(topic: dict = Body(...)):
    searched_full = topic.get("searched", "").strip()
    if not searched_full:
        return JSONResponse({"answer": "no", "reason": "No searched text provided"})

    try:
        with SessionLocal() as session:
            # Ensure keywords table exists
            session.execute(sql_text("""
                CREATE TABLE IF NOT EXISTS keywords (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,   -- TiDB requires a proper PK
                    keyword VARCHAR(255) UNIQUE, 
                    status ENUM('pending','completed') DEFAULT 'pending',
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    searches INT DEFAULT 1
                )
            """))
            session.commit()

            # Check if keyword exists
            existing = session.execute(
                sql_text("SELECT keyword, status, searches FROM keywords WHERE keyword = :kw"),
                {"kw": searched_full}
            ).first()

            if not existing:
                # Insert new keyword with pending status
                session.execute(
                    sql_text("""
                        INSERT INTO keywords (keyword, status, last_seen, searches)
                        VALUES (:kw, 'pending', NOW(), 1)
                    """),
                    {"kw": searched_full}
                )
                session.commit()
                return JSONResponse({"answer": "no", "reason": "Keyword just inserted, status pending"})

            # Update metadata (counter + last_seen)
            session.execute(
                sql_text("""
                    UPDATE keywords
                    SET searches = searches + 1, last_seen = NOW()
                    WHERE keyword = :kw
                """),
                {"kw": searched_full}
            )
            session.commit()

            # Return based on status
            return JSONResponse({"answer": "yes" if existing.status == "completed" else "no",
                                 "status": existing.status})

    except Exception as e:
        return JSONResponse({"answer": "no", "error": str(e)})


@app.post("/embedcorporate")
async def embedcorporate(chunks: list[dict] = Body(...)):
    if not chunks:
        return JSONResponse({"status": "error", "message": "No chunks provided"})

    # Extract texts and metadata
    texts = [c.get("text") for c in chunks]
    metadata_list = [c.get("metadata") for c in chunks]

    # Validate metadata exists for all chunks
    for idx, meta in enumerate(metadata_list):
        if not meta or "url" not in meta or "retrieved_at" not in meta or "sourcekb" not in meta:
            return JSONResponse({
                "status": "error",
                "message": f"Missing required metadata in chunk index {idx}"
            })

    vecs = model.encode(texts, show_progress_bar=False)
    inserted_count = 0

    # Determine source table name from first chunk
    first_meta = metadata_list[0]
    url = first_meta["url"]
    sourcekb = first_meta["sourcekb"]
    internal_table = normalize_table_name(url, sourcekb)  # e.g., mongodb_internal

    try:
        with SessionLocal() as session:
            # 1. Create internal table if not exists
            session.execute(sql_text(f"""
                CREATE TABLE IF NOT EXISTS {internal_table} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding VECTOR(384),
                    url TEXT,
                    retrieved_at TIMESTAMP,
                    sourcekb VARCHAR(100)
                )
            """))
            session.commit()

            # 2. Insert chunks with metadata
            for text_val, emb, meta in zip(texts, vecs, metadata_list):
                session.execute(sql_text(f"""
                    INSERT INTO {internal_table} 
                    (chunk_text, embedding, url, retrieved_at, sourcekb)
                    VALUES (:chunk_text, CAST(:emb_str AS VECTOR(384)), :url, :retrieved_at, :sourcekb)
                    ON DUPLICATE KEY UPDATE
                        chunk_text = VALUES(chunk_text),
                        embedding = VALUES(embedding),
                        url = VALUES(url),
                        retrieved_at = VALUES(retrieved_at),
                        sourcekb = VALUES(sourcekb)
                """), {
                    "chunk_text": text_val,
                    "emb_str": json.dumps(emb.astype("float32").tolist()),
                    "url": meta["url"],
                    "retrieved_at": meta["retrieved_at"],
                    "sourcekb": meta["sourcekb"],
                })
                inserted_count += 1

            session.commit()

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

    return JSONResponse({
        "status": "ok",
        "inserted_count": inserted_count,
        "internal_table": internal_table
    })

def normalize_keyword(searched: str):
    return searched.split("--")[0].split("-")[0].strip()

def get_neighbor_chunks(conn, table, chunk_id):
    """
    Retrieve chunk-1, chunk, chunk+1 and concatenate.
    """
    rows = conn.execute(sql_text(f"""
        SELECT chunk_id, chunk_text FROM {table}
        WHERE chunk_id IN (:prev_id, :curr_id, :next_id)
        ORDER BY chunk_id
    """), {
        "prev_id": chunk_id - 1,
        "curr_id": chunk_id,
        "next_id": chunk_id + 1
    }).fetchall()
    return " ".join([r["chunk_text"] for r in rows])

@app.post("/searchvectordb")
async def searchvectordb(payload: dict = Body(...)):
    try:
        query_json = json.loads(payload.get("query", "{}"))
        question_obj = query_json["question"]
        question = question_obj.get("question", "").strip()
        options = question_obj.get("options", [])
        metadata = question_obj.get("metadata", {})

        if not question or not options:
            return JSONResponse({"status": "error", "message": "Missing question or options"})

        # Extract base keyword from metadata.searched
        searched_raw = metadata.get("searched", "")
        base_keyword = normalize_keyword(searched_raw)

        now = datetime.utcnow()
        internal_cutoff = now - timedelta(days=365)
        external_cutoff = now - timedelta(days=90)

        option_scores = []

        for option in options:
            query_text = f"{question} {option}"
            vec = model.encode([query_text])[0]

            results = {"internal": [], "external": []}

            # --- External search ---
            with SessionLocal() as session:
                rows = session.execute(sql_text("""
                    SELECT id, chunk_text, embedding, url, retrieved_at
                    FROM mongodb_rag_external
                    WHERE sourcekb='external' AND retrieved_at >= :cutoff
                """), {"cutoff": external_cutoff}).mappings().all()

                for row in rows:
                    emb = np.array(json.loads(row["embedding"]))
                    score = np.dot(vec, emb) / (np.linalg.norm(vec) * np.linalg.norm(emb))
                    results["external"].append({
                        "id": row["id"],
                        "text": row["chunk_text"],
                        "url": row["url"],
                        "score": float(score)
                    })

            # --- Internal search ---
            with SessionLocal() as session:
                rows = session.execute(sql_text("""
                    SELECT id, chunk_text, embedding, url, retrieved_at
                    FROM mongodb_rag_internal
                    WHERE sourcekb='internal' AND retrieved_at >= :cutoff
                """), {"cutoff": internal_cutoff}).mappings().all()

                for row in rows:
                    emb = np.array(json.loads(row["embedding"]))
                    score = np.dot(vec, emb) / (np.linalg.norm(vec) * np.linalg.norm(emb))
                    results["internal"].append({
                        "id": row["id"],
                        "text": row["chunk_text"],
                        "url": row["url"],
                        "score": float(score)
                    })

            # --- Select best scoring chunk among internal + external ---
            all_results = results["internal"] + results["external"]
            if all_results:
                best_chunk = max(all_results, key=lambda x: x["score"])
                option_scores.append({"option": option, "score": best_chunk["score"]})
            else:
                option_scores.append({"option": option, "score": 0})

        # --- Select best option ---
        best_option = max(option_scores, key=lambda x: x["score"])
        logger.info(f"Option scores: {option_scores}")
        logger.info(f"Best option selected: {best_option}")

        return JSONResponse({
            "status": "ok",
            "question": question,
            "options": options,
            "bestAnswer": best_option["option"],
            "metadata": metadata
        })

    except Exception as e:
        logger.exception("Error in /searchvectordb")
        return JSONResponse({"status": "error", "message": str(e)})
