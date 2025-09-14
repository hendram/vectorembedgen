About This Package

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

###   How This Works ?

**This works based on some logic received from chunkgeneratorforaimodel**

#  📌  /embed Endpoint – Embedding and Storage Pipeline

The /embed API endpoint is responsible for:

Extracting the searched keyword from incoming document chunks.

Normalizing a table name to store embeddings for that keyword.

Creating or refreshing the external table (if it does not exist or is too old).

Enabling TiFlash replication for accelerated vector search queries.

Creating a vector index on the embeddings column.

Inserting document chunks and their embeddings into the external table.

Updating the keyword status in the keywords table.

🔹 Endpoint Definition
@app.post("/embed")
async def embed(chunks: list[dict] = Body(...)):


Accepts a JSON body containing a list of chunks, where each chunk contains:

"text" → the content to embed.

"metadata" → additional details such as url, date, sourcekb, and the searched keyword.

🔹 Processing Steps
1. Validate Input

Ensures that the request contains chunks.

Extracts the searched keyword from the first chunk’s metadata.

If no keyword is provided → returns an error.

2. Normalize Keyword

Strips extra filters such as --filetype or -site.

Produces a clean base keyword used to name the external table.

external_table = normalize_table_name_external(base_keyword)


This ensures that each unique search term has its own dedicated table for embeddings.

3. Encode Embeddings

Uses the model.encode() function to convert text chunks into vector embeddings.

Stores these vectors in memory for later database insertion.

vecs = model.encode(texts, show_progress_bar=False)

4. Manage External Table

Connects to the database via SessionLocal.

Checks if the table already exists in information_schema.tables.

If the table exists but is older than 3 days, it is dropped and recreated.

If it does not exist, it is created fresh.

The created table has the following structure:

CREATE TABLE `{external_table}` (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384),
    url TEXT,
    retrieved_at TIMESTAMP,
    sourcekb VARCHAR(100)
)

5. Enable TiFlash Replication

TiFlash is enabled for the table to allow fast analytical/vector queries.

ALTER TABLE `{external_table}` SET TIFLASH REPLICA 1;


A loop waits until TiFlash reports that the replica is ready before proceeding.

6. Create Vector Index

A vector index is created on the embedding column to optimize similarity search:

ALTER TABLE `{external_table}`
ADD VECTOR INDEX embedding_idx ((VEC_COSINE_DISTANCE(embedding))) USING HNSW;


This allows fast nearest-neighbor lookups (e.g., finding the most relevant chunks for a query).

7. Insert Chunks with Embeddings

Each chunk is inserted into the external table with:

chunk_text → raw text.

embedding → 384-dimensional vector (stored as JSON → CAST into VECTOR).

url → source URL.

retrieved_at → timestamp from metadata.

sourcekb → source knowledge base identifier.

Duplicate entries update existing rows (ON DUPLICATE KEY UPDATE).

8. Update Keywords Table

Finally, the keywords table is updated to mark the searched keyword as completed:

UPDATE keywords
SET status = 'completed',
    last_seen = CURRENT_TIMESTAMP
WHERE keyword = :kw

🔹 Response

The endpoint returns:

{
  "status": "ok",
  "inserted_count": 25,
  "external_table": "python_programming",
  "keyword_status": "completed"
}


inserted_count → Number of chunks successfully inserted.

external_table → Name of the external table used.

keyword_status → Processing status (completed).

✅ Summary

This endpoint provides a fully automated embedding pipeline that:

Accepts raw document chunks.

Generates embeddings.

Creates a keyword-specific table (auto-refreshes if outdated).

Sets up TiFlash + HNSW vector index for efficient similarity search.

Stores chunk embeddings alongside metadata.

Tracks keyword processing status.

It ensures that each search keyword has a fresh, queryable vector store optimized for retrieval.
 

📌 /insertsearchtodb Endpoint – Keyword Tracking

This endpoint manages the keywords table in the database. It ensures that every searched keyword is tracked, its status (pending / completed) is stored, and its usage statistics are updated.

It’s the entry point of the pipeline:

Keeps track of what keywords have been searched.

Marks whether embeddings for that keyword have been processed.

Prevents duplicate processing by reusing the existing keyword record.

🔹 Endpoint Definition
@app.post("/insertsearchtodb")
async def insert_search_to_db(topic: dict = Body(...)):


Accepts a JSON body with:

{
  "searched": "python tutorial --filetype:pdf"
}


Extracts the value of "searched" and uses it to manage the keywords table.

🔹 Processing Steps
1. Validate Input

Extracts the "searched" string.

If missing or empty → returns:

{"answer": "no", "reason": "No searched text provided"}

2. Ensure keywords Table Exists

Before inserting or updating, it ensures the table exists:

CREATE TABLE IF NOT EXISTS keywords (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,   -- TiDB requires a PK
    keyword VARCHAR(255) UNIQUE,
    status ENUM('pending','completed') DEFAULT 'pending',
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    searches INT DEFAULT 1
)


Columns explained:

id → unique primary key.

keyword → the searched text (unique).

status → workflow state (pending = waiting for embeddings, completed = embeddings stored).

last_seen → when this keyword was last searched.

searches → how many times this keyword has been submitted.

3. Check if Keyword Exists
SELECT keyword, status, searches 
FROM keywords 
WHERE keyword = :kw


If the keyword does not exist:

Insert it with status = 'pending' and searches = 1.

Return immediately with:

{"answer": "no", "reason": "Keyword just inserted, status pending"}

4. Update Metadata (If Keyword Exists)

If the keyword is already present:

Increment the search counter.

Update the last_seen timestamp.

UPDATE keywords
SET searches = searches + 1, last_seen = NOW()
WHERE keyword = :kw

5. Return Based on Status

If the keyword’s status = 'completed':

{"answer": "yes", "status": "completed"}


If the keyword’s status = 'pending':

{"answer": "no", "status": "pending"}


This lets the caller know whether the embeddings for that keyword are already in the system.

6. Error Handling

If any exception occurs (e.g., DB connection issue), returns:

{"answer": "no", "error": "<error message>"}

🔹 Workflow Role

This endpoint is the controller for keyword ingestion:

First-time keyword → inserts new row, marks it as pending.

Repeated keyword → increments search count, updates timestamp, and reports the current status.

Syncs with /embed →

/insertsearchtodb manages keyword tracking.

/embed updates the keyword’s status to completed once embeddings are stored.

Together, they form a two-step ingestion pipeline:

/insertsearchtodb → track keyword (create or update entry).

/embed → process keyword chunks into embeddings and mark as completed.

🔹 Example Scenarios
First-time keyword search
POST /insertsearchtodb
{ "searched": "python tutorial --filetype:pdf" }


Response

{"answer": "no", "reason": "Keyword just inserted, status pending"}

Repeat keyword search (still pending)
{"answer": "no", "status": "pending"}

Repeat keyword search (completed)
{"answer": "yes", "status": "completed"}

✅ Summary

The /insertsearchtodb endpoint:

Ensures a keywords tracking table exists.

Records new searched keywords with pending status.

Updates counters and timestamps for existing keywords.

Returns whether a keyword is already processed (completed) or still pending.

This enables a controlled pipeline for embedding ingestion, preventing duplicate work and tracking keyword lifecycle.


📌 /searchvectordb Endpoint – Semantic Multiple-Choice Search

This endpoint performs a semantic similarity search over stored embeddings (internal and external knowledgebases) to automatically select the best multiple-choice option for a given question.

It’s essentially the retrieval + scoring part of your pipeline:

Takes a question with options.

Looks up related embeddings in the vector DB.

Computes similarity scores.

Picks the option with the highest semantic similarity.

🔹 Endpoint Definition
@app.post("/searchvectordb")
async def searchvectordb(payload: dict = Body(...)):


Accepts a JSON body with a query field containing:

{
  "query": "{ \"question\": { \"question\": \"Which language is used in FastAPI?\", \"options\": [\"Python\", \"Java\", \"Go\"], \"metadata\": { \"searched\": \"fastapi tutorial\" } } }"
}

🔹 Processing Steps
1. Parse and Validate Input

Extracts:

question: the actual question string.

options: list of possible answers.

metadata: contains the original searched keyword.

If question or options is missing → return error.

2. Normalize Keyword

From metadata.searched, extracts the base keyword (e.g., "fastapi tutorial") using normalize_keyword.

This determines which database tables to query.

3. Set Time Cutoffs

Defines how fresh data must be:

Internal knowledge cutoff → 365 days old.

External knowledge cutoff → 90 days old.

This ensures only recent, relevant data is used.

4. Encode Question + Option Pair

For each option:

Concatenates the question and option → "Which language is used in FastAPI? Python".

Encodes it into an embedding vector:

vec = model.encode([query_text])[0]

5. Search External Knowledgebase

Uses table name: <base_keyword>_external.

Checks if the table exists.

If it exists, retrieves recent rows (retrieved_at >= external_cutoff) where sourcekb = 'external'.

For each row:

Loads stored embedding.

Computes cosine similarity:

score = np.dot(vec, emb) / (np.linalg.norm(vec) * np.linalg.norm(emb))


Stores text, URL, and score.

6. Search Internal Knowledgebase

Uses table name: <base_keyword>_internal.

Same logic as external, but cutoff = 365 days and sourcekb = 'internal'.

7. Pick Best Chunk per Option

Combine internal + external results.

If results exist → select the chunk with the highest similarity score.

Otherwise → assign score = 0.

Append { "option": option, "score": best_chunk_score } to option_scores.

8. Pick Best Answer

From all options, select the one with the highest score:

best_option = max(option_scores, key=lambda x: x["score"])


---

🔹 Response

Example response:

{
  "status": "ok",
  "question": "Which language is used in FastAPI?",
  "options": ["Python", "Java", "Go"],
  "bestAnswer": "Python",
  "metadata": { "searched": "fastapi tutorial" }
}


If something goes wrong (e.g., DB error), returns:

{ "status": "error", "message": "<error message>" }


---

#�   ��� Workflow Role

/insertsearchtodb → Track keyword status.

/embed → Insert embeddings into <keyword>_internal / <keyword>_external.

/searchvectordb → Query embeddings to pick best multiple-choice answer. or

/searchvectordb  ^f^r only if insertsearchtodb keyword check having status completed 

This endpoint closes the loop: after ingestion, it lets you ask questions with options and retrieves the best answer based on stored knowledge.

---

#✅ Summary

The /searchvectordb endpoint:

Accepts a multiple-choice question with options.

Encodes each option paired with the question into a vector.

Searches both internal and external embeddings (with freshness cutoffs).

Computes cosine similarity to stored chunks.

Chooses the best option with the highest similarity score.
