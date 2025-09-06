from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import json
import pymysql
import math
import re

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

# --- Create table (if not exists) ---
with SessionLocal() as session:
    session.execute(sql_text("""
        CREATE TABLE IF NOT EXISTS textvectorized (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            embedding VECTOR(384)
        )
    """))
    session.commit()

BATCH_SIZE = 5

from urllib.parse import urlparse

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

@app.post("/embed")
async def embed(chunks: list[dict] = Body(...)):
    if not chunks:
        return JSONResponse({"status": "error", "message": "No chunks provided"})

    # Extract texts and metadata
    texts = [c["text"] for c in chunks if "text" in c]
    metadata_list = [c.get("metadata", {}) for c in chunks]

    # Extract searched keyword from metadata
    searched_keyword = metadata_list[0]["searched"]  # assume must exist, no fallback
    external_table = normalize_table_name(searched_keyword, "external")

    vecs = model.encode(texts, show_progress_bar=False)
    inserted_count = 0

    try:
        with SessionLocal() as session:
            # 1️⃣ Create external table if not exists
            session.execute(sql_text(f"""
                CREATE TABLE IF NOT EXISTS {external_table} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding VECTOR(384),
                    url TEXT,
                    retrieved_at TIMESTAMP,
                    sourcekb VARCHAR(100),
                    searched VARCHAR(255)
                )
            """))
            session.commit()

            # 2️⃣ Insert chunks into external table
            for text_val, emb, meta in zip(texts, vecs, metadata_list):
                session.execute(sql_text(f"""
                    INSERT INTO {external_table} 
                    (chunk_text, embedding, url, retrieved_at, sourcekb, searched)
                    VALUES (:chunk_text, CAST(:emb_str AS VECTOR(384)), :url, :retrieved_at, :sourcekb, :searched)
                    ON DUPLICATE KEY UPDATE
                        chunk_text = VALUES(chunk_text),
                        embedding = VALUES(embedding),
                        url = VALUES(url),
                        retrieved_at = VALUES(retrieved_at),
                        sourcekb = VALUES(sourcekb),
                        searched = VALUES(searched)
                """),
                {
                    "chunk_text": text_val,
                    "emb_str": json.dumps(emb.astype("float32").tolist()),
                    "url": meta["url"],
                    "retrieved_at": meta["retrieved_at"],
                    "sourcekb": meta["sourcekb"],
                    "searched": searched_keyword
                })
                inserted_count += 1

            session.commit()

            # 3️⃣ Check for corresponding internal table
            internal_table = normalize_search_keyword(searched_keyword, "internal")
            join_table = f"{internal_table}_{external_table}"

            internal_exists = session.execute(sql_text(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = '{internal_table}'
            """)).scalar()

            join_exists = session.execute(sql_text(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = '{join_table}'
            """)).scalar()

            # 4️⃣ If internal exists and join does not, create join table
            if internal_exists > 0 and join_exists == 0:
                session.execute(sql_text(f"""
                    CREATE TABLE {join_table} (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        internal_id BIGINT,
                        external_id BIGINT,
                        chunk_text TEXT,
                        internal_embedding VECTOR(384),
                        external_embedding VECTOR(384),
                        url TEXT,
                        retrieved_at TIMESTAMP,
                        sourcekb VARCHAR(100),
                        searched VARCHAR(255)
                    )
                """))
                session.commit()

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

    return JSONResponse({
        "status": "ok",
        "inserted_count": inserted_count,
        "external_table": external_table,
        "join_table_created": internal_exists > 0 and join_exists == 0
    })

@app.post("/insertsearchtodb")
async def insert_search_to_db(topic: dict = Body(...)):
    searched_full = topic.get("searched", "").strip()
    if not searched_full:
        return JSONResponse({"answer": "no", "reason": "No searched text provided"})

    # Extract first word as keyword
    keyword = searched_full.split()[0].lower()

    try:
        with SessionLocal() as session:
            # 1️⃣ Create keywords table if not exists
            session.execute(sql_text("""
                CREATE TABLE IF NOT EXISTS keywords (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    searched VARCHAR(255) UNIQUE
                )
            """))
            session.commit()

            # 2️⃣ Check if keyword exists
            existing = session.execute(
                sql_text("SELECT id FROM keywords WHERE searched = :kw"),
                {"kw": keyword}
            ).first()

            if not existing:
                # Insert new keyword
                session.execute(
                    sql_text("INSERT INTO keywords (searched) VALUES (:kw)"),
                    {"kw": keyword}
                )
                session.commit()
                return JSONResponse({"answer": "no", "reason": "Keyword just inserted"})

            # 3️⃣ Keyword exists, check if join table exists
            join_table = normalize_search_keyword(keyword, "internal_external")
            join_exists = session.execute(sql_text(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = :tbl
            """), {"tbl": join_table}).scalar()

            if join_exists > 0:
                return JSONResponse({"answer": "yes", "join_table": join_table})
            else:
                return JSONResponse({"answer": "no", "reason": "Join table does not exist"})

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
            # 1️⃣ Create internal table if not exists
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

            # 2️⃣ Insert chunks with metadata
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
                """),
                {
                    "chunk_text": text_val,
                    "emb_str": json.dumps(emb.astype("float32").tolist()),
                    "url": meta["url"],
                    "retrieved_at": meta["retrieved_at"],
                    "sourcekb": meta["sourcekb"],
                })
                inserted_count += 1

            session.commit()

            # 3️⃣ Handle external table join
            external_table = normalize_table_name(url, "external")  # e.g., mongodb_external

            # Check if external table exists
            ext_exists = session.execute(sql_text(f"""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = '{external_table}'
            """)).scalar()

            if ext_exists > 0:
                join_table = f"{internal_table}_{external_table}"  # e.g., mongodb_internal_mongodb_external

                # Drop & recreate join table
                session.execute(sql_text(f"DROP TABLE IF EXISTS {join_table}"))
                session.commit()

                session.execute(sql_text(f"""
                    CREATE TABLE {join_table} (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        internal_id BIGINT,
                        external_id BIGINT,
                        chunk_text TEXT,
                        internal_embedding VECTOR(384),
                        external_embedding VECTOR(384),
                        url TEXT,
                        retrieved_at TIMESTAMP,
                        sourcekb VARCHAR(100)
                    )
                """))
                session.commit()

                # Populate join table using vector similarity (pseudo SQL, implement actual vector distance logic if TiDB supports)
                session.execute(sql_text(f"""
                    INSERT INTO {join_table} (internal_id, external_id, chunk_text, internal_embedding, external_embedding, url, retrieved_at, sourcekb)
                    SELECT i.id, e.id, i.chunk_text, i.embedding, e.embedding, i.url, i.retrieved_at, i.sourcekb
                    FROM {internal_table} i
                    JOIN {external_table} e
                      ON 1=1  -- placeholder, replace with actual vector similarity comparison if supported
                """))
                session.commit()

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

    return JSONResponse({
        "status": "ok",
        "inserted_count": inserted_count,
        "internal_table": internal_table,
        "external_table": external_table if ext_exists > 0 else None
    })


@app.post("/searchvectordb")
async def searchvectordb(query: str = Body(...), top_k: int = Body(1)):
    # Encode query → list of floats
    query_embedding = model.encode([query], show_progress_bar=False)[0].tolist()
    emb_str = json.dumps(query_embedding)

    session = SessionLocal()
    try:
        rows = session.execute(
            sql_text("""
                SELECT id, chunk_text,
                       VEC_COSINE_DISTANCE(embedding, CAST(:query AS VECTOR(384))) AS score
                FROM textvectorized
                ORDER BY score ASC
                LIMIT :top_k
            """),
            {"query": emb_str, "top_k": top_k}
        ).mappings().all()
    finally:
        session.close()

    return {
        "query": query,
        "top_results": [dict(row) for row in rows]
    }
