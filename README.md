#  About This Package

This package functions as executor to create, insert, vectorized and search on tidb database. Shortly 
could said this is a database connector to Tidb cloud. 

🚀 How to Run It

📥 Download

```bash
docker pull ghcr.io/hendram/vectorembedgen
```

▶️ Start

```bash
docker run -it -d --network=host ghcr.io/hendram/vectorembedgen bash
```

🔍 Check Running Container

```bash
docker ps
```

```bash
CONTAINER ID   IMAGE                               NAME                STATUS
123abc456def   ghcr.io/hendram/vectorembedgen      confident_banzai    Up 5 minutes
```

📦 Enter Container

```bash
docker exec -it confident_banzai /bin/bash
```

🏃 Run the Service

```bash
cd /home
source .venv/bin/activate
uvicorn vectorembedgen:app --host 0.0.0.0 --port 8000
```

---

# 📖 **How This Works**

####  This works based on some logic received from chunkgeneratorforaimodel

📌 /embed Endpoint – Embedding and Storage Pipeline

##  The /embed and /embedcorporate API endpoint is responsible for:

📝 Extracting the searched keyword from incoming document chunks.

🧹 Normalizing a table name to store embeddings for that keyword.

🗄️ Creating or refreshing the external table (if it does not exist or is too old).

⚡ Enabling TiFlash replication for accelerated vector search queries.

🧭 Creating a vector index on the embeddings column.

📥 Inserting document chunks and their embeddings into the external table.

🔄 Updating the keyword status in the keywords table.

---

## 🔹 Endpoint Definition

###  @app.post("/embed")
async def embed(chunks: list[dict] = Body(...)):


Accepts a JSON body containing a list of chunks.

Each chunk contains:

```bash
{
   "text" → the content to embed.

   "metadata" → details such as url, date, sourcekb, and the searched keyword.

}
```
---

###  🔹 Processing Steps

#### ✅ Validate Input

Ensures the request contains chunks.

Extracts the searched keyword from the first chunk’s metadata.

If no keyword is provided → ❌ returns an error.

#### 🧹 Normalize Keyword

Strips filters like --filetype or -site.

Produces a clean base keyword used to name the external table.

external_table = normalize_table_name_external(base_keyword)l

for embedcorporate internal_table = normalize_table_name_internal


#### ➡️ Ensures each search term has its own dedicated table for embeddings.

### 🧠 Encode Embeddings

Converts chunks into embeddings with model.encode().

vecs = model.encode(texts, show_progress_bar=False)

### 🗄️ Manage External or Internal Table

####  Connects via SessionLocal.

If table exists but is older than 3 days → drop and recreate.

Else → create fresh.

```bash
CREATE TABLE {external_table} (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384),
    url TEXT,
    retrieved_at TIMESTAMP,
    sourcekb VARCHAR(100)
);
```
#### ⚡ Enable TiFlash Replication

```bash
ALTER TABLE {external_table} SET TIFLASH REPLICA 1;
```

➡️ Waits until TiFlash is ready before proceeding.

🧭 Create Vector Index

```bash
ALTER TABLE {external_table}
ADD VECTOR INDEX embedding_idx ((VEC_COSINE_DISTANCE(embedding)))
USING HNSW;
```

#### ➡️ Optimizes nearest-neighbor lookups for similarity search.

#### 📥 Insert Chunks with Embeddings

Inserts:

```bash
   chunk_text → raw text

   embedding → 384-dim vector

   url → source URL

   retrieved_at → timestamp

   sourcekb → knowledge base type
```

#### ➡️ Duplicate entries → update existing rows.

#### 🔄 Update Keywords Table

```bash
UPDATE keywords
SET status = 'completed', last_seen = CURRENT_TIMESTAMP
WHERE keyword = :kw;
```


🔹 Response

```bash
{
  "status": "ok",
  "inserted_count": 25,
  "external_table": "python_programming",
  "keyword_status": "completed"
}
```

inserted_count → number of chunks stored.

external_table → name of created table.

keyword_status → processing state.


---

## 📌 /insertsearchtodb Endpoint – Keyword Tracking

This endpoint manages the keywords table.

📝 Tracks every searched keyword.

⚙️ Stores its status (pending / completed).

🔄 Updates usage statistics.

---

## 🔹 Endpoint Definition


###  @app.post("/insertsearchtodb")


async def insert_search_to_db(topic: dict = Body(...)):


Accepts:

```bash
{ "searched": "python tutorial --filetype:pdf" }
```

### 🔹 Processing Steps

#### ✅ Validate Input

If "searched" missing →

```bash
{ "answer": "no", "reason": "No searched text provided" }
```

#### 🗄️ Ensure keywords Table Exists


```bash
CREATE TABLE IF NOT EXISTS keywords (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    keyword VARCHAR(255) UNIQUE,
    status ENUM('pending','completed') DEFAULT 'pending',
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    searches INT DEFAULT 1
);
```

#### 🔍 Check if Keyword Exists

```bash
SELECT keyword, status, searches
FROM keywords
WHERE keyword = :kw;
```

If new → insert with pending.

Else → increment counter + update timestamp.

#### 🔄 Return Based on Status

If status = completed →

```bash
{ "answer": "yes", "status": "completed" }
```

If status = pending →

```bash
{ "answer": "no", "status": "pending" }
```

---

## 📌 /searchvectordb Endpoint – Semantic Multiple-Choice Search

This endpoint performs semantic similarity search to select the best multiple-choice answer.

Takes: question + options.

###  Searches embeddings.

  Computes similarity.

✅ Picks the best-scoring option.

---

##  🔹 Endpoint Definition

###  @app.post("/searchvectordb")

async def searchvectordb(payload: dict = Body(...)):


Input:

```bash
{
  "query": {
    "question": {
      "question": "Which language is used in FastAPI?",
      "options": ["Python", "Java", "Go"],
      "metadata": { "searched": "fastapi tutorial" }
    }
  }
}
```

### 🔹 Processing Steps


#### ✅ Parse and Validate Input

Extracts:

```bash
   question

   options

   metadata.searched
```

#### 🧹 Normalize Keyword

From metadata.searched.

Used to decide which tables to query.

#### ⏳ Apply Time Cutoffs

Internal KB → 365 days.

External KB → 90 days.

#### 🧠 Encode Question + Option Pair

query_text = f"{question} {option}"

vec = model.encode([query_text])[0]

#### 🌍 Search External Knowledgebase

Table: <base_keyword>_external.

If exists → select rows (fresh within 90 days).

Compute cosine similarity.

#### 🏠 Search _Internal Table

Table: <base_keyword>_internal.

If exists → select rows (fresh within 365 days).

Compute cosine similarity.

#### 📊 Pick Best Chunk per pair Question & Options

For each option → best scoring chunk.

If none → score = 0.

#### ✅ Pick Best Answer


###  Select option with highest similarity score.

#### 🔹 Response

```bash
{
  "status": "ok",
  "question": "Which language is used in FastAPI?",
  "options": ["Python", "Java", "Go"],
  "bestAnswer": "Python",
  "metadata": { "searched": "fastapi tutorial" }
}
```

If error →

```bash
{ "status": "error", "message": "" }
```

## 🔄 Workflow Summary

/insertsearchtodb → track keyword (pending/completed).

/embed → process + store embeddings (mark completed).

/searchvectordb → retrieve best multiple-choice answer. or direct to here if /insertsearchtodb checked table already mark completed

➡️ Together, they form a full ingestion + retrieval pipeline.
